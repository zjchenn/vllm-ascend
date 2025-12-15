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
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y_chunk = tl.load(y_ptrs, mask=y_mask, other=0.0)
                                                                                                                                                                    
        # 计算矩阵乘法累加
        acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)
                                                                                                                                                                                        
    # 根据has_bias标志决定是否添加偏置
    if has_bias:
        # 加载偏置值（广播到所有行）
        bias_ptrs = bias_ptr + rn * stride_bias        
        bias_mask = rn < None        
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        # 将偏置加到累加器上（自动广播）
        acc += bias_vals[None, :]
                                                                                                                                                                                                                                        
    # 计算输出指针位置
    out_ptrs = output_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn    
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
                                                                                                                                                                                                                                                    
    # 将结果存储到全局内存
    tl.store(out_ptrs, acc, mask=out_mask)


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
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
                                                                                                                                    
    # 定义分块大小（可根据硬件调整）
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
                                                                                                                                                
    # 计算网格大小（每个分块一个线程）
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
                                                                                                                                                            
    # 处理bias为None的情况
    if bias is None:
        # 创建一个虚拟的bias张量（不会被使用，因为has_bias=False）
        dummy_bias = torch.empty(0, dtype=x.dtype, device=x.device)
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

    # Register flash attention batch invariant version
    # Note: Flash attention is typically called by the attention backend,
    # so we don't need to register it as a PyTorch operator.
    # The attention backend should check VLLM_USE_BATCH_INVARIANT_ATTENTION
    # environment variable and use flash_attn_with_kvcache_batch_invariant
    # instead of the standard flash_attn_with_kvcache.




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


@cache
def use_batch_invariant_attention():
    """
    检查是否应该使用批不变性的 Flash Attention。

    返回 True 如果满足以下任一条件：
    1. VLLM_USE_BATCH_INVARIANT_ATTENTION=1
    2. VLLM_BATCH_INVARIANT=1（全局批不变性模式）
    """
    # 检查专门的 flash attention 环境变量
    fa_env = os.getenv("VLLM_USE_BATCH_INVARIANT_ATTENTION", None)
    if fa_env is not None:
        try:
            return int(fa_env) != 0
        except ValueError:
            pass

    # 回退到全局批不变性设置
    return vllm_is_batch_invariant()


def override_envs_for_invariance():
    # Attention backend determinism settings
    # Flash Attention will check use_batch_invariant_attention()
    # to decide whether to use the batch invariant version
    # Users can set VLLM_USE_BATCH_INVARIANT_ATTENTION=1 explicitly
    # or rely on VLLM_BATCH_INVARIANT=1 (global setting)

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


# ============================================================================
# Flash Attention with KVCache - Batch Invariant Implementation
# ============================================================================

@triton.jit
def _flash_attn_kvcache_batch_invariant_kernel(
    # 输入输出指针
    Q, K_cache, V_cache, Out,
    K_new, V_new,
    Cache_seqlens, Cache_batch_idx,
    # Q 的步长
    stride_qz, stride_qm, stride_qh, stride_qd,
    # K cache 的步长
    stride_kz, stride_kn, stride_kh, stride_kd,
    # V cache 的步长
    stride_vz, stride_vn, stride_vh, stride_vd,
    # Output 的步长
    stride_oz, stride_om, stride_oh, stride_od,
    # K new 的步长（如果有新 KV）
    stride_knz, stride_knn, stride_knh, stride_knd,
    # V new 的步长（如果有新 KV）
    stride_vnz, stride_vnn, stride_vnh, stride_vnd,
    # 维度参数
    Z, N_CTX_Q, N_CTX_K, N_CTX_NEW,
    H_q, H_kv, HEADDIM,
    sm_scale,
    # 编译时常量
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_CACHE_SEQLENS: tl.constexpr,
    USE_CACHE_BATCH_IDX: tl.constexpr,
    NEW_KV: tl.constexpr,
    IS_GQA: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    批不变性的 Flash Attention with KVCache Triton Kernel

    关键设计：
    1. 不使用 Split-K 并行，串行遍历所有 K/V 块
    2. Float32 累加，确保精度
    3. 禁用 TF32（通过 allow_tf32=False）
    4. 固定的遍历顺序
    5. 串行 KV 更新（只有 pid_m==0 的块负责）
    """
    # 获取程序 ID
    pid_m = tl.program_id(0)  # Query 块 ID
    pid_zh = tl.program_id(1)  # Batch * Heads 的组合 ID

    # 分解 batch 和 head 索引
    z_id = pid_zh // H_q
    hq_id = pid_zh % H_q

    # GQA: 计算对应的 KV head
    if IS_GQA:
        hk_id = hq_id // GROUP_SIZE
        hv_id = hk_id
    else:
        hk_id = hq_id
        hv_id = hq_id

    # 确定实际的 KV 序列长度
    if USE_CACHE_SEQLENS:
        cache_seqlen = tl.load(Cache_seqlens + z_id)
        if NEW_KV:
            N_CTX_K_FINAL = cache_seqlen + N_CTX_NEW
        else:
            N_CTX_K_FINAL = cache_seqlen
    else:
        N_CTX_K_FINAL = N_CTX_K

    # 确定 batch 索引映射
    if USE_CACHE_BATCH_IDX:
        cache_batch_idx = tl.load(Cache_batch_idx + z_id)
    else:
        cache_batch_idx = z_id

    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Q 指针
    q_offset = Q + z_id * stride_qz + hq_id * stride_qh
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    # K/V Cache 指针
    k_offset = K_cache + cache_batch_idx * stride_kz + hk_id * stride_kh
    v_offset = V_cache + cache_batch_idx * stride_vz + hv_id * stride_vh

    # 创建掩码
    q_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEADDIM)

    # 加载 Q（保持在 SRAM）
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 缩放 Q（使用 log2(e) 优化）
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(q.dtype)

    # === KV Cache 更新（批不变性关键：串行更新）===
    # 只有 pid_m == 0 的线程块负责更新，避免并发写入
    if NEW_KV and pid_m == 0:
        knew_base = K_new + z_id * stride_knz + hk_id * stride_knh
        vnew_base = V_new + z_id * stride_vnz + hv_id * stride_vnh

        # 确定起始位置
        if USE_CACHE_SEQLENS:
            start_idx = tl.load(Cache_seqlens + z_id)
        else:
            start_idx = N_CTX_K - N_CTX_NEW

        # 逐块复制新的 K
        for i in range(0, N_CTX_NEW, BLOCK_N):
            k_new_block = tl.load(
                knew_base +
                offs_d[:, None] * stride_knd +
                (offs_n[None, :] + i) * stride_knn,
                mask=(offs_d[:, None] < HEADDIM) & ((offs_n[None, :] + i) < N_CTX_NEW),
                other=0.0
            )
            tl.store(
                k_offset +
                offs_d[:, None] * stride_kd +
                (offs_n[None, :] + i + start_idx) * stride_kn,
                k_new_block,
                mask=(offs_d[:, None] < HEADDIM) & ((offs_n[None, :] + i) < N_CTX_NEW)
            )

        # 逐块复制新的 V
        for i in range(0, N_CTX_NEW, BLOCK_N):
            v_new_block = tl.load(
                vnew_base +
                (offs_n[:, None] + i) * stride_vnn +
                offs_d[None, :] * stride_vnd,
                mask=((offs_n[:, None] + i) < N_CTX_NEW) & (offs_d[None, :] < HEADDIM),
                other=0.0
            )
            tl.store(
                v_offset +
                (offs_n[:, None] + i + start_idx) * stride_vn +
                offs_d[None, :] * stride_vd,
                v_new_block,
                mask=((offs_n[:, None] + i) < N_CTX_NEW) & (offs_d[None, :] < HEADDIM)
            )

    # === Online Softmax Attention（批不变性关键：固定顺序）===

    # 初始化累加器（在 float32 中累加）
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 批不变性关键：固定顺序遍历，不使用 Split-K
    # 所有批次以相同的顺序访问 K/V 块
    for start_n in range(0, N_CTX_K_FINAL, BLOCK_N):
        # 计算当前块的实际大小
        curr_n_end = tl.minimum(start_n + BLOCK_N, N_CTX_K_FINAL)

        # 加载 K^T 块
        kT_ptrs = k_offset + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
        kT_mask = (offs_d[:, None] < HEADDIM) & ((start_n + offs_n)[None, :] < N_CTX_K_FINAL)
        kT = tl.load(kT_ptrs, mask=kT_mask, other=0.0)

        # 加载 V 块
        v_ptrs = v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = ((start_n + offs_n)[:, None] < N_CTX_K_FINAL) & (offs_d[None, :] < HEADDIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # 计算 QK^T（批不变性关键：禁用 TF32）
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, kT, allow_tf32=False)

        # 应用 Causal Mask
        if IS_CAUSAL:
            row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)
            # 创建 N_CTX_Q x N_CTX_K_FINAL 的因果掩码
            col_offset = N_CTX_Q - N_CTX_K_FINAL
            causal_mask = row_idx[:, None] >= (col_offset + col_idx[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # 边界掩码
        boundary_mask = (start_n + offs_n)[None, :] < N_CTX_K_FINAL
        qk = tl.where(boundary_mask, qk, float("-inf"))

        # Online Softmax（批不变性关键：精确控制累加顺序）
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))

        # 计算 alpha（重缩放因子）
        if IS_CAUSAL:
            alpha = tl.math.exp2(tl.where(m_i > float("-inf"), m_i - m_i_new, float("-inf")))
        else:
            alpha = tl.math.exp2(m_i - m_i_new)

        # 计算 P = exp2(QK - m_new)
        if IS_CAUSAL:
            qk = tl.where(qk > float("-inf"), qk - m_i_new[:, None], float("-inf"))
        else:
            qk = qk - m_i_new[:, None]

        p = tl.math.exp2(qk)

        # 更新 l_i 和 m_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        # 转换 P 的类型
        p = p.to(q.dtype)

        # 重缩放累加器并累加新值（批不变性关键：在 float32 中累加）
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32), allow_tf32=False)

    # === 最终归一化 ===
    # 避免除零
    l_i_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    # 转换回输出类型
    acc = acc.to(Out.dtype.element_ty)

    # 存储输出
    out_ptrs = Out + z_id * stride_oz + hq_id * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEADDIM)
    tl.store(out_ptrs, acc, mask=out_mask)


def flash_attn_with_kvcache_batch_invariant(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    qv: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    rotary_seqlens: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata=None,
    num_splits: int = 0,
    pack_gqa: bool | None = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
) -> torch.Tensor | tuple:
    """
    批不变性版本的 Flash Attention with KVCache（Hopper 接口）

    与 flash-attention/hopper/flash_attn_interface.py:928 的 flash_attn_with_kvcache 接口对齐。

    确保批不变性：
    flash_attn_with_kvcache_batch_invariant(q[:1], k_cache[:1], v_cache[:1])
    == flash_attn_with_kvcache_batch_invariant(q, k_cache, v_cache)[:1]

    通过以下方式保证批不变性：
    1. 串行 K/V 遍历（无 Split-K 并行）
    2. Float32 累加确保精度
    3. 禁用 TF32（allow_tf32=False）
    4. 固定的遍历顺序

    参数:
        q: (batch_size, seqlen, nheads, headdim) - Query 张量
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) - Key cache
                 或 (num_blocks, page_block_size, nheads_k, headdim) 如果使用 page_table
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) - Value cache
                 或 (num_blocks, page_block_size, nheads_k, headdim_v) 如果使用 page_table
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim) - 新的 Key
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v) - 新的 Value
        qv [optional]: (batch_size, seqlen, nheads, headdim_v) - Query for value（暂不支持）
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2) - Rotary embedding cos（暂不支持）
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2) - Rotary embedding sin（暂不支持）
        cache_seqlens: int or (batch_size,) dtype=int32 - KV cache 序列长度
        cache_batch_idx: (batch_size,) dtype=int32 - 批次索引映射
        cache_leftpad: (batch_size,) dtype=int32 - KV cache 起始索引（暂不支持）
        page_table [optional]: (batch_size, max_num_blocks_per_seq) dtype=int32 - Paged KV cache（暂不支持）
        cu_seqlens_q [optional]: 累积序列长度（varlen）（暂不支持）
        cu_seqlens_k_new [optional]: 新 K 的累积序列长度（暂不支持）
        max_seqlen_q [optional]: 最大 query 序列长度（暂不支持）
        rotary_seqlens [optional]: Rotary embedding 序列长度（暂不支持）
        q_descale [optional]: FP8 量化的 descale 因子（暂不支持）
        k_descale [optional]: FP8 量化的 descale 因子（暂不支持）
        v_descale [optional]: FP8 量化的 descale 因子（暂不支持）
        softmax_scale: float - QK^T 的缩放因子，默认 1/sqrt(headdim)
        causal: bool - 是否使用因果掩码
        window_size: (left, right) - 滑动窗口（暂不支持）
        attention_chunk: int - Attention chunk size（暂不支持）
        softcap: float - Softcapping attention（暂不支持）
        rotary_interleaved: bool - Rotary embedding 模式（暂不支持）
        scheduler_metadata: Scheduler metadata（暂不支持）
        num_splits: int - Split-K 数量（批不变性版本固定为1）
        pack_gqa: bool - 是否 pack GQA（暂不支持）
        sm_margin: int - SM margin for communication（暂不支持）
        return_softmax_lse: bool - 是否返回 log-sum-exp

    返回:
        output: (batch_size, seqlen, nheads, headdim) - 输出张量
        或 (output, softmax_lse) 如果 return_softmax_lse=True
            softmax_lse: (batch_size, nheads, seqlen)

    限制（MVP 版本）:
        - 不支持 qv
        - 不支持 rotary_cos/rotary_sin（rotary embedding 应在外部处理）
        - 不支持 page_table（paged KV cache）
        - 不支持 cache_leftpad
        - 不支持 cu_seqlens_q/cu_seqlens_k_new（varlen）
        - 不支持 window_size（滑动窗口）
        - 不支持 attention_chunk
        - 不支持 softcap
        - 不支持 FP8 量化（q_descale/k_descale/v_descale）
        - 不支持 scheduler_metadata
        - 不支持 pack_gqa
        - 不支持 sm_margin
        - num_splits 固定为 1（无 Split-K）
        - return_softmax_lse 暂返回 None
    """
    # 验证不支持的参数
    if qv is not None:
        raise NotImplementedError("qv not supported in batch_invariant version yet")
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("Rotary embedding not supported in batch_invariant version yet")
    if page_table is not None:
        raise NotImplementedError("Paged KV cache (page_table) not supported in batch_invariant version yet")
    if cache_leftpad is not None:
        raise NotImplementedError("cache_leftpad not supported in batch_invariant version yet")
    if cu_seqlens_q is not None or cu_seqlens_k_new is not None:
        raise NotImplementedError("Variable-length sequences (cu_seqlens) not supported in batch_invariant version yet")
    if max_seqlen_q is not None:
        raise NotImplementedError("max_seqlen_q not supported in batch_invariant version yet")
    if rotary_seqlens is not None:
        raise NotImplementedError("rotary_seqlens not supported in batch_invariant version yet")
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError("FP8 quantization (descale) not supported in batch_invariant version yet")
    if window_size != (-1, -1):
        raise NotImplementedError("Sliding window attention not supported in batch_invariant version yet")
    if attention_chunk != 0:
        raise NotImplementedError("attention_chunk not supported in batch_invariant version yet")
    if softcap != 0.0:
        raise NotImplementedError("Softcap not supported in batch_invariant version yet")
    if scheduler_metadata is not None:
        raise NotImplementedError("scheduler_metadata not supported in batch_invariant version yet")
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa not supported in batch_invariant version yet")
    if sm_margin != 0:
        raise NotImplementedError("sm_margin not supported in batch_invariant version yet")
    if num_splits != 0 and num_splits != 1:
        raise ValueError(f"Batch-invariant version only supports num_splits=0 or 1, got {num_splits}")

    # 验证输入
    assert q.dim() == 4, f"q 必须是 4D 张量 [batch, seqlen, nheads, headdim]，当前: {q.shape}"
    assert k_cache.dim() == 4 and v_cache.dim() == 4, "k_cache 和 v_cache 必须是 4D 张量"

    batch, seqlen_q, nheads_q, headdim = q.shape
    batch_cache, seqlen_cache, nheads_kv, _ = k_cache.shape

    # 验证 k_cache 和 v_cache 形状匹配
    assert k_cache.shape == v_cache.shape, f"k_cache 和 v_cache 形状必须相同：{k_cache.shape} vs {v_cache.shape}"

    # 处理 cache_seqlens（可以是 int 或 Tensor）
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (batch,), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    # 计算 softmax_scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (headdim ** 0.5)

    # 确定是否为 GQA
    assert nheads_q % nheads_kv == 0, f"nheads_q ({nheads_q}) 必须能被 nheads_kv ({nheads_kv}) 整除"
    group_size = nheads_q // nheads_kv
    is_gqa = group_size > 1

    # 处理新的 KV
    is_new_kv = k is not None and v is not None
    if is_new_kv:
        assert k.shape == v.shape, f"k 和 v 形状必须相同：{k.shape} vs {v.shape}"
        assert k.shape[0] == batch, f"k 的 batch 大小必须与 q 相同：{k.shape[0]} vs {batch}"
        assert k.shape[2] == nheads_kv, f"k 的 nheads 必须与 k_cache 相同：{k.shape[2]} vs {nheads_kv}"
        seqlen_new = k.shape[1]
    else:
        seqlen_new = 0
        # 创建 dummy 张量以传递给 kernel
        k = torch.empty((batch, 0, nheads_kv, headdim), dtype=q.dtype, device=q.device)
        v = torch.empty((batch, 0, nheads_kv, headdim), dtype=q.dtype, device=q.device)

    # 处理 cache_seqlens
    use_cache_seqlens = cache_seqlens is not None
    if not use_cache_seqlens:
        cache_seqlens = torch.empty(0, dtype=torch.int32, device=q.device)
    else:
        if cache_seqlens.dtype != torch.int32:
            cache_seqlens = cache_seqlens.to(torch.int32)

    # 处理 cache_batch_idx
    use_cache_batch_idx = cache_batch_idx is not None
    if not use_cache_batch_idx:
        cache_batch_idx = torch.empty(0, dtype=torch.int32, device=q.device)
    else:
        if cache_batch_idx.dtype != torch.int32:
            cache_batch_idx = cache_batch_idx.to(torch.int32)

    # 分配输出张量
    output = torch.empty_like(q)

    # 固定的块大小（批不变性的关键）
    BLOCK_M = 16  # Query 块大小
    BLOCK_N = 64  # Key/Value 块大小
    BLOCK_DMODEL = triton.next_power_of_2(headdim)

    # 计算网格大小
    grid = (
        triton.cdiv(seqlen_q, BLOCK_M),  # M 维度的块数
        batch * nheads_q,                 # Batch * Heads
    )

    # 启动 kernel
    _flash_attn_kvcache_batch_invariant_kernel[grid](
        # 输入输出
        q, k_cache, v_cache, output,
        k, v,
        cache_seqlens, cache_batch_idx,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K cache strides
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        # V cache strides
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # K new strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V new strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # 维度
        batch, seqlen_q, seqlen_cache, seqlen_new,
        nheads_q, nheads_kv, headdim,
        softmax_scale,
        # 编译时常量
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        USE_CACHE_SEQLENS=use_cache_seqlens,
        USE_CACHE_BATCH_IDX=use_cache_batch_idx,
        NEW_KV=is_new_kv,
        IS_GQA=is_gqa,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=1,
    )

    # TODO: 实现 softmax_lse 计算
    # 当前kernel不输出lse，如果需要可以在kernel中添加lse输出
    if return_softmax_lse:
        # softmax_lse shape: (batch, nheads, seqlen)
        # 暂时返回None作为占位符
        softmax_lse = None
        return output, softmax_lse
    else:
        return output
