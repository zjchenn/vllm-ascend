# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.batch_invariant import (
     _log_softmax_batch_invariant)

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


@triton.jit
def linear_persistent_kernel(
    a_ptr,  # 指针指向张量 a，形状 [M, K]
    b_ptr,  # 指针指向张量 b，形状 [N, K]
    c_ptr,  # 指针指向输出张量 c，形状 [M, N]
    M,      # 张量 a 的行数
    N,      # 张量 b 的行数（输出 c 的列数）
    K,      # 张量 a 的列数和张量 b 的列数
    stride_am,  # 张量 a 在维度 M 的步长（通常为 K）
    stride_ak,  # 张量 a 在维度 K 的步长（通常为 1）
    stride_bn,  # 张量 b 在维度 N 的步长（通常为 K）
    stride_bk,  # 张量 b 在维度 K 的步长（通常为 1）
    stride_cm,  # 张量 c 在维度 M 的步长（通常为 N）
    stride_cn,  # 张量 c 在维度 N 的步长（通常为 1）
    BLOCK_M: tl.constexpr,  # 阻塞大小 for M 维度
    BLOCK_N: tl.constexpr,  # 阻塞大小 for N 维度
    BLOCK_K: tl.constexpr,  # 阻塞大小 for K 维度
    NUM_BLOCKS_M: tl.constexpr,  # 新增：M 维度的块数
    NUM_BLOCKS_N: tl.constexpr,  # 新增：N 维度的块数
    GRID_SIZE: tl.constexpr,     # 新增：固定的一维网格大小
):
    # 获取当前程序的一维索引（一维网格）
    pid = tl.program_id(0)
    total_blocks = NUM_BLOCKS_M * NUM_BLOCKS_N  # 总输出块数
                                                                                            
    # 循环处理分配给当前 program 的多个块（类似知识片段7的循环策略）
    for block_index in range(pid, total_blocks, GRID_SIZE):
        # 将一维块索引转换为二维坐标 (m_block, n_block)
        m_block = block_index // NUM_BLOCKS_N        
        n_block = block_index % NUM_BLOCKS_N
        
        # 计算当前输出块的起始索引
        start_m = m_block * BLOCK_M        
        start_n = n_block * BLOCK_N

        # 创建当前块内的行和列索引范围
        m_indices = start_m + tl.arange(0, BLOCK_M)
        n_indices = start_n + tl.arange(0, BLOCK_N)
                                                                                                                                                                        
        # 创建掩码以处理边界
        m_mask = m_indices < M        
        n_mask = n_indices < N        

        # 初始化累加器为0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                                                                                                                                                                                                                
        # 循环遍历 K 维度，以 BLOCK_K 为步长进行阻塞
        for k_offset in range(0, K, BLOCK_K):
            k_indices = k_offset + tl.arange(0, BLOCK_K)
            k_mask = k_indices < K
                                                                                                                                                                                                                                                                        
            # 加载张量 a 的块：形状 [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + m_indices[:, None] * stride_am + k_indices[None, :] * stride_ak            
            a_vals = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                                                                                                                                                                                                                                                                                                            
            # 加载张量 b 的块：形状 [BLOCK_N, BLOCK_K]
            b_ptrs = b_ptr + n_indices[:, None] * stride_bn + k_indices[None, :] * stride_bk            
            b_vals = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
                                                                                                                                                                                                                                                                                                                                                
            # 使用 tl.trans 显式转置 b 矩阵：形状变为 [BLOCK_K, BLOCK_N]
            b_vals_transposed = tl.trans(b_vals)
                                                                                                                                                                                                                                                                                                                                                                                    
            # 计算矩阵乘法：a_vals × b_vals_transposed
            product = tl.dot(a_vals, b_vals_transposed)
            acc += product        
        # 将结果存储到输出张量 c
        c_ptrs = c_ptr + m_indices[:, None] * stride_cm + n_indices[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def linear_persistent(x, y):
    """
    使用Triton实现矩阵乘法加可选偏置: x @ y^T
    使用固定大小的一维网格
                    
    参数:
        x: torch.Tensor, 形状为 [M, K]
        y: torch.Tensor, 形状为 [N, K] 
                                                    
    返回:
        output: torch.Tensor, 形状为 [M, N]
    """
    # 验证输入形状
    assert x.dim() == 2, "x必须是2D张量"
    assert y.dim() == 2, "y必须是2D张量" 
    assert x.shape[1] == y.shape[1], f"矩阵维度不匹配: x.shape[1]={x.shape[1]}, y.shape[1]={y.shape[1]}"
                                                                                        
    M, K = x.shape    
    N, _ = y.shape
       
    # 分配输出张量（与x相同的数据类型）
    output = torch.zeros((M, N), dtype=x.dtype, device=x.device)
                                                                                                        
    # 定义分块大小（可根据硬件调整）
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
                                                                                                                    
    # 计算每个维度的块数（向上取整）
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
                                                                                                                                    
    # 设置固定的一维网格
    grid_size = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"] // 2
    grid = (grid_size,)
                                                                                                                                                    
    # 启动kernel
    linear_persistent_kernel[grid](
        a_ptr=x, b_ptr=y, c_ptr=output,
        M=M, N=N, K=K,
        stride_am=x.stride(0), stride_ak=x.stride(1),
        stride_bn=y.stride(0), stride_bk=y.stride(1),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BLOCKS_M=num_blocks_m,  # 传入M维度块数
        NUM_BLOCKS_N=num_blocks_n,  # 传入N维度块数
        GRID_SIZE=grid_size,        # 传入固定网格大小
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
    output = linear_persistent(input, weight)

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

    # _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
    # _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
    # _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant,
    #                           "NPU")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant,
                              "NPU")
    # _batch_invariant_LIB.impl("aten::_log_softmax",
    #                           _log_softmax_batch_invariant, "NPU")
    # _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant,
    #                           "NPU")
    # _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant,
    #                           "NPU")
    # _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant,
    #                           "NPU")

    # Also monkeypatch torch.bmm directly as a fallback
    # _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
    # _original_torch_bmm = torch.bmm
    # torch.bmm = bmm_batch_invariant



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

    # enabling NZ mode introduces NZ format input to the triton operator,
    # resulting in accuracy anomalies.
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"

    # communication determinism settings
    os.environ["HCCL_DETERMINISTIC"] = "true"
    os.environ["LCCL_DETERMINISTIC"] = "1"

    # computing determinism settings
    os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "0"
    os.environ["ATB_LLM_LCOC_ENABLE"] = "0"

# =============================================================================
# Batch-Invariant Flash Attention with KV Cache
# =============================================================================
#
# This section implements a batch-invariant Flash Attention operator using
# Triton for Ascend NPU. The key design principle is that each (batch, head)
# pair is processed independently to ensure batch invariance.
#
# Grid Design: (num_q_blocks, batch_size, num_heads)
# - This ensures Request A's computation never depends on Request B
#
# Block Sizes: Conservative sizes to avoid NPU UB overflow (192KB limit)
# - BLOCK_M = 64 (query block size)
# - BLOCK_N = 64 (key/value block size, adjusted based on HEAD_DIM)
# =============================================================================


def _get_block_sizes(head_dim: int) -> tuple:
    """
    Compute conservative block sizes based on head dimension to avoid NPU UB overflow.

    Approximate UB usage per block:
    - Q_block: BLOCK_M * HEAD_DIM * 2 bytes
    - K_block: BLOCK_N * HEAD_DIM * 2 bytes
    - V_block: BLOCK_N * HEAD_DIM * 2 bytes
    - O_block: BLOCK_M * HEAD_DIM * 2 bytes
    - Float32 accumulators: BLOCK_M * BLOCK_N * 4 bytes

    Args:
        head_dim: The head dimension (e.g., 64, 128, 192, 256)

    Returns:
        Tuple of (BLOCK_M, BLOCK_N)
    """
    BLOCK_M = 64  # Conservative query block size

    # Adjust BLOCK_N based on head dimension to stay within UB limits
    if head_dim <= 64:
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_N = 64
    elif head_dim <= 192:
        BLOCK_N = 32
    else:  # head_dim > 192
        BLOCK_N = 32

    return BLOCK_M, BLOCK_N


@triton.jit
def _flash_attn_with_kvcache_kernel(
    # Q, K, V pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Softmax LSE output (optional)
    LSE_ptr,
    # Cache sequence lengths pointer (for variable length sequences)
    Cache_seqlens_ptr,
    # Dimensions
    batch_size, seqlen_q, seqlen_k, num_heads_q, num_heads_k, head_dim,
    # Strides for Q: (batch, seqlen, heads, head_dim)
    stride_qb, stride_qs, stride_qh, stride_qd,
    # Strides for K: (batch, seqlen, heads, head_dim)
    stride_kb, stride_ks, stride_kh, stride_kd,
    # Strides for V: (batch, seqlen, heads, head_dim)
    stride_vb, stride_vs, stride_vh, stride_vd,
    # Strides for O: (batch, seqlen, heads, head_dim)
    stride_ob, stride_os, stride_oh, stride_od,
    # Stride for LSE: (batch, heads, seqlen)
    stride_lseb, stride_lseh, stride_lses,
    # Attention parameters
    softmax_scale,
    # Causal masking
    is_causal: tl.constexpr,
    # Softcap (0.0 means disabled)
    softcap,
    # GQA ratio (num_heads_q // num_heads_k)
    gqa_ratio: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Whether to output LSE
    WRITE_LSE: tl.constexpr,
    # Whether to use variable sequence lengths
    USE_VARLEN: tl.constexpr,
):
    """
    Batch-invariant Flash Attention forward kernel.

    Grid: (num_q_blocks, batch_size, num_heads_q)

    This kernel processes one block of queries per program, ensuring that
    each (batch, head) pair is processed independently for batch invariance.

    When USE_VARLEN=True, each sequence can have different K/V lengths,
    read from Cache_seqlens_ptr[batch_idx].
    """
    # Get program IDs - this is the key to batch invariance
    # Each (batch, head) pair gets its own independent computation
    pid_m = tl.program_id(0)  # Query block index
    pid_b = tl.program_id(1)  # Batch index
    pid_h = tl.program_id(2)  # Head index (query head)

    # Compute KV head index for GQA/MQA
    # For standard attention: gqa_ratio = 1
    # For GQA: gqa_ratio = num_heads_q // num_heads_k > 1
    kv_head_idx = pid_h // gqa_ratio

    # Get the actual seqlen_k for this batch element
    if USE_VARLEN:
        # Read sequence length for this batch element
        actual_seqlen_k = tl.load(Cache_seqlens_ptr + pid_b)
    else:
        actual_seqlen_k = seqlen_k

    # Compute starting positions
    q_start = pid_m * BLOCK_M

    # Create offset arrays
    offs_m = q_start + tl.arange(0, BLOCK_M)  # Query positions [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)            # Key/Value positions [BLOCK_N]
    offs_d = tl.arange(0, HEAD_DIM)           # Head dimension [HEAD_DIM]

    # Compute Q pointer for this (batch, head, q_block)
    Q_block_ptr = (Q_ptr +
                   pid_b * stride_qb +
                   offs_m[:, None] * stride_qs +
                   pid_h * stride_qh +
                   offs_d[None, :] * stride_qd)

    # Load Q block with masking
    q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    q = tl.load(Q_block_ptr, mask=q_mask, other=0.0).to(tl.float32)

    # Initialize output accumulator and log-sum-exp
    # Using float32 for numerical stability
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp(scores)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # Max score

    # Iterate over K/V blocks
    # For causal masking, we only need to iterate up to the diagonal
    # Note: Causal mask is aligned to bottom-right corner of attention matrix
    # This means query at position i can attend to keys at positions <= i + (seqlen_k - seqlen_q)
    causal_offset = actual_seqlen_k - seqlen_q  # Offset for bottom-right alignment

    if is_causal:
        # For causal attention, the last valid K position for query at position q_pos
        # is q_pos + causal_offset (aligned to bottom-right corner of attention matrix)
        kv_len = tl.minimum(actual_seqlen_k, q_start + BLOCK_M + causal_offset)
    else:
        kv_len = actual_seqlen_k

    num_kv_blocks = tl.cdiv(kv_len, BLOCK_N)

    for kv_block_idx in range(num_kv_blocks):
        k_start = kv_block_idx * BLOCK_N
        offs_kv = k_start + offs_n

        # Compute K pointer for this (batch, kv_head, kv_block)
        K_block_ptr = (K_ptr +
                       pid_b * stride_kb +
                       offs_kv[None, :] * stride_ks +
                       kv_head_idx * stride_kh +
                       offs_d[:, None] * stride_kd)

        # Load K block: [HEAD_DIM, BLOCK_N]
        k_mask = (offs_kv[None, :] < actual_seqlen_k) & (offs_d[:, None] < head_dim)
        k = tl.load(K_block_ptr, mask=k_mask, other=0.0).to(tl.float32)

        # Compute attention scores: Q @ K^T -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k, allow_tf32=False)
        scores = scores * softmax_scale

        # Apply softcap if enabled
        if softcap > 0.0:
            scores = softcap * tl.math.tanh(scores / softcap)

        # Apply causal mask if needed
        if is_causal:
            # Causal mask: query at position i can only attend to keys at positions <= i + offset
            # This aligns the mask to the bottom-right corner of the attention matrix
            # Example: if seqlen_q=2, seqlen_k=5, offset=3
            #   Q0 can attend to K0,K1,K2,K3 (positions <= 0+3)
            #   Q1 can attend to K0,K1,K2,K3,K4 (positions <= 1+3)
            causal_mask = (offs_m[:, None] + causal_offset) >= offs_kv[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Apply boundary mask for keys beyond actual_seqlen_k
        boundary_mask = offs_kv[None, :] < actual_seqlen_k
        scores = tl.where(boundary_mask, scores, float("-inf"))

        # Online softmax update (numerically stable)
        # m_ij = max of current block scores
        m_ij = tl.max(scores, axis=1)
        # New max
        m_new = tl.maximum(m_i, m_ij)
        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # Compute exp(scores - m_new)
        p = tl.exp(scores - m_new[:, None])
        # Update sum of exp
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Load V block: [BLOCK_N, HEAD_DIM]
        V_block_ptr = (V_ptr +
                       pid_b * stride_vb +
                       offs_kv[:, None] * stride_vs +
                       kv_head_idx * stride_vh +
                       offs_d[None, :] * stride_vd)

        v_mask = (offs_kv[:, None] < actual_seqlen_k) & (offs_d[None, :] < head_dim)
        v = tl.load(V_block_ptr, mask=v_mask, other=0.0).to(tl.float32)

        # Update output accumulator
        # acc = alpha * acc + P @ V
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v, allow_tf32=False)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output
    O_block_ptr = (O_ptr +
                   pid_b * stride_ob +
                   offs_m[:, None] * stride_os +
                   pid_h * stride_oh +
                   offs_d[None, :] * stride_od)

    o_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    tl.store(O_block_ptr, acc.to(O_block_ptr.dtype.element_ty), mask=o_mask)

    # Write log-sum-exp if requested
    if WRITE_LSE:
        lse = m_i + tl.log(l_i)
        LSE_block_ptr = (LSE_ptr +
                         pid_b * stride_lseb +
                         pid_h * stride_lseh +
                         offs_m * stride_lses)
        lse_mask = offs_m < seqlen_q
        tl.store(LSE_block_ptr, lse, mask=lse_mask)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    """
    Batch-invariant Flash Attention with KV Cache.

    This function provides a drop-in replacement for flash_attn_with_kvcache
    from flash-attention v3, ensuring batch-invariant computation on Ascend NPU.

    The key batch invariance guarantee: the output for sequence i is
    mathematically identical regardless of:
    - Its position in the batch
    - The presence or absence of other sequences in the batch
    - The sequence lengths of other sequences

    Arguments:
        q: (batch_size, seqlen_q, num_heads, head_dim)
        k_cache: (batch_size, seqlen_k, num_heads_k, head_dim) - KV cache for keys
        v_cache: (batch_size, seqlen_k, num_heads_k, head_dim) - KV cache for values
        k: Optional new keys to append (not yet supported in batch-invariant version)
        v: Optional new values to append (not yet supported in batch-invariant version)
        softmax_scale: Scale factor for attention scores. Default: 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        softcap: Softcap value (0.0 means disabled)
        return_softmax_lse: Whether to return log-sum-exp values

    Returns:
        out: (batch_size, seqlen_q, num_heads, head_dim)
        softmax_lse: (batch_size, num_heads, seqlen_q) if return_softmax_lse=True

    Note:
        - This implementation currently does not support:
          - Appending new K/V to cache (k, v parameters)
          - Rotary embeddings (rotary_cos, rotary_sin)
          - Paged KV cache (page_table)
          - Variable length sequences (cu_seqlens_q)
          - Sliding window attention (window_size)
          - FP8 quantization (q_descale, k_descale, v_descale)
        - These features may be added in future versions
    """
    # Input validation
    assert q.dim() == 4, f"q must be 4D (batch, seqlen, heads, head_dim), got {q.dim()}D"
    assert k_cache.dim() == 4, f"k_cache must be 4D, got {k_cache.dim()}D"
    assert v_cache.dim() == 4, f"v_cache must be 4D, got {v_cache.dim()}D"

    # Feature limitation warnings/assertions
    if k is not None or v is not None:
        raise NotImplementedError(
            "Appending new K/V to cache is not yet supported in batch-invariant mode. "
            "Please update the cache manually before calling this function."
        )

    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError(
            "Rotary embeddings are not yet supported in batch-invariant mode. "
            "Please apply rotary embeddings to Q/K before calling this function."
        )

    if page_table is not None:
        raise NotImplementedError(
            "Paged KV cache is not yet supported in batch-invariant mode."
        )

    if cu_seqlens_q is not None:
        raise NotImplementedError(
            "Variable length sequences (varlen mode) are not yet supported "
            "in batch-invariant mode."
        )

    if window_size != (-1, -1):
        raise NotImplementedError(
            "Sliding window attention is not yet supported in batch-invariant mode."
        )

    # Extract dimensions
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    _, seqlen_k, num_heads_k, _ = k_cache.shape

    # Handle cache_seqlens (actual sequence lengths in cache)
    # cache_seqlens can be:
    # - None: use full seqlen_k for all sequences
    # - int: all sequences have the same length
    # - Tensor: each sequence has its own length
    if cache_seqlens is not None:
        if isinstance(cache_seqlens, int):
            # All sequences have the same length
            cache_seqlens_tensor = None
            effective_seqlen_k = cache_seqlens
            variable_seqlens = False
        elif isinstance(cache_seqlens, torch.Tensor):
            # Variable lengths per sequence
            cache_seqlens_tensor = cache_seqlens.to(torch.int32)
            effective_seqlen_k = seqlen_k  # Will be overridden per-sequence
            variable_seqlens = True
        else:
            raise TypeError(f"cache_seqlens must be int or Tensor, got {type(cache_seqlens)}")
    else:
        cache_seqlens_tensor = None
        effective_seqlen_k = seqlen_k
        variable_seqlens = False

    # Validate GQA/MQA configuration
    assert num_heads_q % num_heads_k == 0, (
        f"num_heads_q ({num_heads_q}) must be divisible by num_heads_k ({num_heads_k})"
    )
    gqa_ratio = num_heads_q // num_heads_k

    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Ensure inputs are contiguous
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()

    # Allocate output tensor
    out = torch.empty_like(q)

    # Allocate LSE tensor if requested
    if return_softmax_lse:
        softmax_lse = torch.empty(
            (batch_size, num_heads_q, seqlen_q),
            dtype=torch.float32,
            device=q.device
        )
    else:
        # Create dummy tensor (won't be written to)
        softmax_lse = torch.empty(0, device=q.device)

    # Get conservative block sizes for NPU
    BLOCK_M, BLOCK_N = _get_block_sizes(head_dim)

    # Round up head_dim to power of 2 for efficiency
    HEAD_DIM_PADDED = triton.next_power_of_2(head_dim)

    # Compute grid dimensions
    # Grid: (num_q_blocks, batch_size, num_heads_q)
    # This ensures batch invariance: each (batch, head) pair is independent
    num_q_blocks = triton.cdiv(seqlen_q, BLOCK_M)
    grid = (num_q_blocks, batch_size, num_heads_q)

    # Prepare cache_seqlens tensor for kernel
    # If variable_seqlens, use the actual tensor; otherwise create a dummy
    if variable_seqlens:
        cache_seqlens_for_kernel = cache_seqlens_tensor
    else:
        # Create a dummy tensor (won't be read because USE_VARLEN=False)
        cache_seqlens_for_kernel = torch.empty(0, dtype=torch.int32, device=q.device)

    # Launch kernel - single call for entire batch
    _flash_attn_with_kvcache_kernel[grid](
        # Pointers
        q, k_cache, v_cache, out,
        softmax_lse,
        cache_seqlens_for_kernel,
        # Dimensions
        batch_size, seqlen_q, effective_seqlen_k, num_heads_q, num_heads_k, head_dim,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        # V strides
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # O strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # LSE strides
        softmax_lse.stride(0) if return_softmax_lse else 0,
        softmax_lse.stride(1) if return_softmax_lse else 0,
        softmax_lse.stride(2) if return_softmax_lse else 1,
        # Attention parameters
        softmax_scale,
        causal,
        softcap,
        gqa_ratio,
        # Block sizes
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_PADDED,
        WRITE_LSE=return_softmax_lse,
        USE_VARLEN=variable_seqlens,
    )

    if return_softmax_lse:
        return out, softmax_lse
    return out


# Alias for consistency with the spec naming
flash_attn_with_kvcache_batch_invariant = flash_attn_with_kvcache


# =============================================================================
# Integration Functions
# =============================================================================
# The following functions provide mechanisms to integrate the batch-invariant
# flash attention into vllm-ascend runtime.
# =============================================================================

_original_flash_attn_with_kvcache = None


def get_flash_attn_with_kvcache():
    """
    Get the batch-invariant flash attention function.

    This function returns the batch-invariant implementation of
    flash_attn_with_kvcache that can be used as a drop-in replacement
    for the standard flash-attention v3 implementation.

    Returns:
        The flash_attn_with_kvcache function

    Example:
        >>> from vllm_ascend.batch_invariant import get_flash_attn_with_kvcache
        >>> flash_attn = get_flash_attn_with_kvcache()
        >>> output = flash_attn(q, k_cache, v_cache, causal=True)
    """
    return flash_attn_with_kvcache


def enable_batch_invariant_flash_attention():
    """
    Enable batch-invariant flash attention by monkey-patching the flash_attn module.

    This function attempts to replace the standard flash_attn_with_kvcache
    implementation with the batch-invariant version. It should be called
    early in the application startup, before any flash attention calls.

    The monkey-patching targets the following modules (if available):
    - flash_attn.flash_attn_interface
    - vllm_ascend.attention (if applicable)

    Example:
        >>> from vllm_ascend.batch_invariant import enable_batch_invariant_flash_attention
        >>> enable_batch_invariant_flash_attention()
        >>> # Now all flash attention calls will use the batch-invariant version

    Note:
        This function is automatically called by init_batch_invariance()
        when VLLM_BATCH_INVARIANT=1 is set.
    """
    global _original_flash_attn_with_kvcache

    # Try to patch flash_attn module if available
    try:
        import flash_attn.flash_attn_interface as flash_attn_interface
        if hasattr(flash_attn_interface, 'flash_attn_with_kvcache'):
            _original_flash_attn_with_kvcache = flash_attn_interface.flash_attn_with_kvcache
            flash_attn_interface.flash_attn_with_kvcache = flash_attn_with_kvcache
    except ImportError:
        pass  # flash_attn not installed, skip patching

    # Also export via module for easy access
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'flash_attn_with_kvcache', flash_attn_with_kvcache)


def disable_batch_invariant_flash_attention():
    """
    Disable batch-invariant flash attention and restore the original implementation.

    This function restores the original flash_attn_with_kvcache implementation
    that was saved when enable_batch_invariant_flash_attention() was called.
    """
    global _original_flash_attn_with_kvcache

    if _original_flash_attn_with_kvcache is not None:
        try:
            import flash_attn.flash_attn_interface as flash_attn_interface
            flash_attn_interface.flash_attn_with_kvcache = _original_flash_attn_with_kvcache
            _original_flash_attn_with_kvcache = None
        except ImportError:
            pass


def init_batch_invariance():
    """
    Initialize batch-invariant mode for vLLM on Ascend NPU.

    This function:
    1. Sets environment variables for deterministic computation
    2. Registers batch-invariant implementations for torch operators
    3. Enables batch-invariant flash attention

    Call this function early in your application, or set VLLM_BATCH_INVARIANT=1
    environment variable to enable automatically.
    """
    if vllm_is_batch_invariant():
        override_envs_for_invariance()
        enable_batch_invariant_mode()
        enable_batch_invariant_flash_attention()

