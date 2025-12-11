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
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),  # 当前块在c中的偏移        
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )

    # 初始化累加器（使用float32避免精度损失）
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 循环遍历K维度，分块计算矩阵乘
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr).to(tl.float32)  # 加载a的块，形状为(BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_block_ptr).to(tl.float32) # 加载b的块，形状为(BLOCK_SIZE_K, BLOCK_SIZE_N)
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
        bias_vals = tl.load(bias_ptr + col_offsets, mask=bias_mask, other=0.0).to(tl.float32)

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
    # BLOCK_SIZE_M = 128
    # BLOCK_SIZE_N = 128
    # BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = min(triton.next_power_of_2(M) // 2, 64)
    BLOCK_SIZE_N = min(triton.next_power_of_2(N) // 2, 64)
    BLOCK_SIZE_K = min(triton.next_power_of_2(K) // 2, 64)

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
    matmul_bias_persistent_kernel[grid](
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

    ref_c = torch.matmul(x, y)
    if (not torch.allclose(c, ref_c, atol=1e-3, rtol=1e-3)):
        print(x.shape)
        print(y.shape)

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
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / RMS

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
        input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

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
    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
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
