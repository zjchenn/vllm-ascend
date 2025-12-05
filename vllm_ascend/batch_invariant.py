# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.model_executor.layers.batch_invariant import (
    mm_batch_invariant, addmm_batch_invariant, matmul_batch_invariant,
    linear_batch_invariant, bmm_batch_invariant, _log_softmax_batch_invariant,
    softmax_batch_invariant, mean_batch_invariant)


@triton.jit
def matmul_bias_persistent_kernel(
    # 输入输出张量的指针
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 张量的步长（strides）
    stride_am, stride_ak,  # a的步长：行步长、列步长
    stride_bk, stride_bn,  # b的步长：行步长、列步长
    stride_cm, stride_cn,  # c的步长
    stride_bias,  # bias的步长（对于向量，通常为1）
    # 块大小（必须为2的幂）
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # 是否启用bias    
    HAS_BIAS: tl.constexpr,
):
    # 获取当前程序实例的ID（处理输出矩阵的哪个块）
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
                
    # 创建块指针（block pointers）用于加载a和b的块
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),  # 当前块在a中的偏移
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)  # 行主序
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),  # 当前块在b中的偏移
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),  # 当前块在c中的偏移        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
                                    
    # 初始化累加器（使用float32避免精度损失）    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                                            
    # 循环遍历K维度，分块计算矩阵乘
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr)  # 加载a的块，形状为(BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_block_ptr)  # 加载b的块，形状为(BLOCK_SIZE_K, BLOCK_SIZE_N)
        acc += tl.dot(a, b)  # 矩阵乘累加
        # 前进指针到下一个K块
        a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_SIZE_K])
        b_block_ptr = tl.advance(b_block_ptr, [BLOCK_SIZE_K, 0])
                                                                                                        
    # 如果启用bias，添加偏置    
    if HAS_BIAS:
        # 计算bias的偏移和mask
        col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_mask = col_offsets < N
                                                                                                                                            
        # 直接使用指针偏移加载bias
        bias_vals = tl.load(bias_ptr + col_offsets, mask=bias_mask, other=0.0)
                                                                                                                                                                    
        # 将bias广播到整个块并加到累加器
        acc += bias_vals[None, :]  # 广播到(BLOCK_SIZE_M, BLOCK_SIZE_N)
                                                                                                                                                                                        
    # 将结果存储到输出
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty))  # 转换类型以匹配输出


def matmul_persistent(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    使用Triton实现矩阵乘并可选添加偏置。
    参数:
        x: 输入矩阵，形状为(M, K)
        y: 输入矩阵，形状为(K, N)
        bias: 可选偏置向量，形状为(N,)。如果为None，则不添加偏置。
    返回:
        输出矩阵，形状为(M, N)
    """
    assert x.dim() == 2 and y.dim() == 2, "输入必须是2D张量"
    assert x.shape[1] == y.shape[0], f"x的列数({x.shape[1]})必须等于y的行数({y.shape[0]})"
    M, K = x.shape    
    _, N = y.shape    
    # 分配输出张量（与x同设备同数据类型）
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
                                                                            
    # 设置块大小（必须为2的幂，Triton的约束）
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
                                                                                                
    # 计算网格大小（每个输出块一个程序实例）
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
                                                                                                            
    # 获取张量的步长（假设张量是连续的）
    stride_am, stride_ak = x.stride() if x.is_contiguous() else (x.stride(0), x.stride(1))
    stride_bk, stride_bn = y.stride() if y.is_contiguous() else (y.stride(0), y.stride(1))
    stride_cm, stride_cn = c.stride()
                                                                                                                                
    # 处理bias参数
    if bias is not None:
        assert bias.dim() == 1 and bias.shape[0] == N, f"bias必须是形状为({N},)的向量，但得到{bias.shape}"
        bias_ptr = bias        
        stride_bias = bias.stride(0)  # 对于向量，通常为1
        HAS_BIAS = True    
    else:
        # 如果bias为None，传递一个虚拟指针（不会实际使用，因为HAS_BIAS=False）
        bias_ptr = x  # 任意有效指针        
        stride_bias = 1  # 虚拟值
        HAS_BIAS = False    
    # 启动Triton内核
    matmul_bias_kernel[grid](
        a_ptr=x,
        b_ptr=y,
        bias_ptr=bias_ptr,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        stride_bias=stride_bias,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=HAS_BIAS,
    )
                                                                                                                                                                                                    
    return c


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)

def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process each batch separately with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        results = []
        for i in range(a.shape[0]):
            results.append(matmul_persistent(a[i], b[i]))
        result = torch.stack(results, dim=0)

        if out is not None:
            out.copy_(result)
            return out        
        return result    
    else:
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)

def matmul_batch_invariant(a, b, *, out=None):
    # torch.matmul can handle various dimensions
    # For 2D x 2D, it's the same as matmul    
    if a.ndim == 2 and b.ndim == 2:
        result = matmul_persistent(a, b)
        if out is not None:
            out.copy_(result)
            return out        
        return result    
    elif a.ndim == 3 and b.ndim == 3:
        # Handle batched case like bmm
        return bmm_batch_invariant(a, b, out=out)
    elif a.ndim == 3 and b.ndim == 2:
        # Handle 3D x 2D: common for linear layers
        # (batch, seq, hidden) @ (hidden, out) -> (batch, seq, out)
        # Reshape to 2D, do mm, reshape back
        batch, seq, hidden = a.shape
        a_2d = a.reshape(-1, hidden)
        result_2d = matmul_persistent(a_2d, b)
        result = result_2d.reshape(batch, seq, -1)
        if out is not None:
            out.copy_(result)
            return out        
        return result    
    elif a.ndim == 2 and b.ndim == 3:
        # Handle 2D x 3D: (M, K) @ (B, K, N) -> (B, M, N)
        # By broadcasting `a` to 3D, we can reuse the batched matrix
        # multiplication logic.
        a_expanded = a.unsqueeze(0).expand(b.shape[0], -1, -1)
        return bmm_batch_invariant(a_expanded, b, out=out)
    elif a.ndim == 4 and b.ndim == 4:
        # Handle 4D attention tensors: [batch, heads, seq, dim]
        # Reshape to 3D, process, reshape back        
        batch, heads, seq_a, dim_a = a.shape        
        _, _, dim_b, seq_b = b.shape

        # Reshape to [batch*heads, seq_a, dim_a]
        a_3d = a.reshape(batch * heads, seq_a, dim_a)
        b_3d = b.reshape(batch * heads, dim_b, seq_b)

        # Do batched matmul        
        result_3d = bmm_batch_invariant(a_3d, b_3d)

        # Reshape back to [batch, heads, seq_a, seq_b]
        result = result_3d.reshape(batch, heads, seq_a, seq_b)

        if out is not None:
            out.copy_(result)
            return out        
        return result    
    else:
        raise ValueError(
            f"matmul_batch_invariant currently only supports 2D x 2D, 3D x 3D, "
            f"3D x 2D, 2D x 3D, and 4D x 4D, "
            f"got shapes {a.shape} and {b.shape}"
        )


def linear_batch_invariant(input, weight, bias=None):
    output = matmul_batch_invariant(input, weight.t())

    if bias is not None:
        output = output + bias    
    return output
            

def enable_batch_invariant_mode():

    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::_log_softmax",
                              _log_softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "NPU")

    # Also monkeypatch torch.bmm directly as a fallback
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
    _original_torch_bmm = torch.bmm
    torch.bmm = bmm_batch_invariant



@cache
def vllm_is_batch_invariant():
    env_key = "VLLM_BATCH_INVARIANT"
    is_overridden = False
    val = os.getenv(env_key, "0")
    try:
        is_overridden = int(val) != 0
    except ValueError:
        is_overridden = False
    return is_overridden


def override_envs_for_invariance():
    # TODO(Ronald) set attntion backend to deterministic mode


    # communication determinism settings
    os.environ["HCCL_DETERMINISTIC"] = "1"
    os.environ["LCCL_DETERMINISTIC"] = "1"

    # computing determinism settings
    os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "0"
    os.environ["ATB_LLM_LCOC_ENABLE"] = "0"



def init_batch_invariance():
    # this will hit all the csrc overrides as well
    if vllm_is_batch_invariant():
        override_envs_for_invariance()
        enable_batch_invariant_mode()
