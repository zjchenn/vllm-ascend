# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.batch_invariant import (
     _log_softmax_batch_invariant,
    softmax_batch_invariant)

import triton.runtime.driver as driver


@triton.jit
def matmul_bias_persistent_kernel(
    # 输入张量指针
    x_ptr, y_ptr, bias_ptr, output_ptr,
    # 矩阵维度
    M, N, K,
    # 步长信息
    stride_xm, stride_xk,  # x的步长: [M, K]
    stride_yk, stride_yn,  # y的步长: [K, N]  
    stride_bias,           # bias的步长: [N]
    stride_outm, stride_outn,  # 输出的步长: [M, N]
    # 是否使用偏置
    has_bias: tl.constexpr,
    # 分块大小（常量表达式）
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 获取程序ID（2D网格）
    pid_m = tl.program_id(0)  # 行分块ID
    pid_n = tl.program_id(1)  # 列分块ID
                
    # 计算当前块在矩阵中的起始位置
    rm_start = pid_m * BLOCK_M    
    rn_start = pid_n * BLOCK_N
    
    # 创建索引范围
    rm = rm_start + tl.arange(0, BLOCK_M)  # 行索引范围 [BLOCK_M]
    rn = rn_start + tl.arange(0, BLOCK_N)  # 列索引范围 [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)              # K维度索引范围 [BLOCK_K]
                                            
    # 初始化累加器为0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                                                        
    # 在K维度上进行循环，每次处理BLOCK_K个元素
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K        
        # 计算x的指针偏移量（行主序）
        x_ptrs = x_ptr + rm[:, None] * stride_xm + (rk[None, :] + k_start) * stride_xk        
        # 计算y的指针偏移量（行主序）  
        y_ptrs = y_ptr + (rk[:, None] + k_start) * stride_yk + rn[None, :] * stride_yn

        # 创建掩码以防止越界访问
        x_mask = (rm[:, None] < M) & ((rk[None, :] + k_start) < K)
        y_mask = ((rk[:, None] + k_start) < K) & (rn[None, :] < N)
                                                                                                                                    
        # 从全局内存加载数据块
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        y_chunk = tl.load(y_ptrs, mask=y_mask, other=0.0).to(tl.float32)
                                                                                                                                                                    
        # 计算矩阵乘法累加
        acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)
                                                                                                                                                                                        
    # 根据has_bias标志决定是否添加偏置
    if has_bias:
        # 加载偏置值（广播到所有行）
        bias_ptrs = bias_ptr + rn * stride_bias        
        bias_mask = rn < None        
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        # 将偏置加到累加器上（自动广播）
        acc += bias_vals[None, :]
                                                                                                                                                                                                                                        
    # 计算输出指针位置
    out_ptrs = output_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn    
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
                                                                                                                                                                                                                                                    
    # 将结果存储到全局内存
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)


def matmul_persistent(x, y, bias=None):
    """
    使用Triton实现矩阵乘法加可选偏置: x @ y + bias (如果bias不为None)
                
    参数:
        x: torch.Tensor, 形状为 [M, K]
        y: torch.Tensor, 形状为 [K, N] 
        bias: torch.Tensor, 形状为 [N] 或 None
                                                
    返回:
        output: torch.Tensor, 形状为 [M, N]
    """
    # 验证输入形状
    assert x.dim() == 2, "x必须是2D张量"
    assert y.dim() == 2, "y必须是2D张量" 
    assert x.shape[1] == y.shape[0], f"矩阵维度不匹配: x.shape[1]={x.shape[1]}, y.shape[0]={y.shape[0]}"
                                                                                    
    M, K = x.shape    
    _, N = y.shape    
    # 验证bias形状（如果不为None）
    if bias is not None:
        assert bias.dim() == 1, "bias必须是1D张量"
        assert y.shape[1] == bias.shape[0], f"偏置维度不匹配: y.shape[1]={y.shape[1]}, bias.shape[0]={bias.shape[0]}"
                                                                                                                        
    # 分配输出张量（与x相同的数据类型）
    output = torch.zeros((M, N), dtype=x.dtype, device=x.device)
                                                                                                                                    
    # 定义分块大小（可根据硬件调整）
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
                                                                                                                                                
    # 计算网格大小（每个分块一个线程）
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
                                                                                                                                                            
    # 处理bias为None的情况
    if bias is None:
        # 创建一个虚拟的bias张量（不会被使用，因为has_bias=False）
        dummy_bias = torch.zeros(0, dtype=x.dtype, device=x.device)
        has_bias = False
        bias_stride = 0
        bias_to_pass = dummy_bias    
    else:
        has_bias = True
        bias_stride = bias.stride(0)
        bias_to_pass = bias    
    # 启动kernel
    matmul_bias_persistent_kernel[grid](
        x, y, bias_to_pass, output,           # 输入输出张量
        M, N, K,                              # 矩阵维度
        x.stride(0), x.stride(1),             # x的步长
        y.stride(0), y.stride(1),             # y的步长  
        bias_stride,                          # bias的步长（如果bias为None则为0）
        output.stride(0), output.stride(1),   # 输出的步长
        has_bias,                             # 是否使用偏置的标志
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
                                                                                                                                                                                                                                                    
    return output


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


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // k
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return
    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
        input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert -input.ndim <= dim < input.ndim, (
        f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    )

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim
    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype
    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output

def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype = torch.float16):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {torch.float16, torch.bfloat16, torch.float32}, (
            "only float types supported for now"
        )
        if len(dim) == 0:
            dim = list(range(input.ndim))
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32).to(dtype or input.dtype) / n_elems


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,  # 新增参数：总行数
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each program handles multiple rows of the input tensor.
    """
    pid = tl.program_id(0)  # 程序ID
    n_programs = tl.num_programs(0)  # 网格大小（固定值，例如1024）

    # 计算每个程序处理的行数（向上取整）
    rows_per_program = (n_rows + n_programs - 1) // n_programs    
    start_row = pid * rows_per_program    
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    # 循环处理分配给该程序的多行
    for row_idx in range(start_row, end_row):
        row_start_ptr = input_ptr + row_idx * input_row_stride        
        output_row_start_ptr = output_ptr + row_idx * output_row_stride

        # Step 1: Compute sum of squares in float32 to avoid overflow
        sum_sq = tl.zeros([1], dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32            
            sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

        # Step 2: Compute RMS (root mean square) in float32
        mean_sq = sum_sq / n_cols        
        rms = tl.sqrt(mean_sq + eps)
        inv_rms = 1.0 / rms

        # Step 3: Normalize and apply weight
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols            
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms * weight_f32            
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
    input: torch.Tensor, 
    weight: torch.Tensor, 
    eps: float = 1e-6,
                    
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel with fixed grid size.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input.shape    
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d, dtype=input.dtype)
    BLOCK_SIZE = 1024  # 保持原有的BLOCK_SIZE
    max_grid_size = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]

    # 固定网格大小，使用min避免当n_rows较小时启动过多程序
    grid = (min(n_rows, max_grid_size),)

    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_rows,  # 传入总行数
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(
        input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.
    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)


def softmax_batch_invariant(input, dim, dtype=None):
    # Compute softmax in a deterministic way
    # First subtract max for numerical stability (standard practice)
    input_max = torch.amax(input, dim=dim, keepdim=True)
    input = input - input_max    
    exp_x = torch.exp(input)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

_batch_invariant_LIB = None

def enable_batch_invariant_mode():
    global _batch_invariant_LIB

    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant,
                              "NPU")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant,
                              "NPU")
    _batch_invariant_LIB.impl("aten::_log_softmax",
                              _log_softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant,
                              "NPU")
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant,
                              "NPU")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant,
                              "NPU")

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
    os.environ["HCCL_DETERMINISTIC"] = "true"
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
