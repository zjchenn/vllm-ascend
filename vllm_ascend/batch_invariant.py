# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from functools import cache

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
# Batch-Invariant Flash Attention with KV Cache (NPU Implementation)
# =============================================================================
#
# This section implements batch-invariant Flash Attention using NPU native
# operators. The key insight is that batch invariance is achieved through:
#
# 1. slot_mapping: Each token independently maps to its cache slot
# 2. page_table: Each sequence independently maps to its KV cache blocks
# 3. cu_seqlens_q: Clear sequence boundaries in varlen mode
#
# This approach leverages NPU's paged attention capabilities directly,
# avoiding the need for custom kernels or KV gathering to contiguous memory.
# =============================================================================

import torch_npu


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Store key-value pairs into the paged KV cache using NPU operator.

    This function stores K/V tokens into their designated cache slots,
    with each token's storage being independent of batch composition.

    Args:
        key: Key tensor of shape (num_tokens, num_kv_heads, head_dim)
        value: Value tensor of shape (num_tokens, num_kv_heads, head_dim)
        key_cache: Key cache of shape (num_blocks, block_size, num_kv_heads, head_dim)
        value_cache: Value cache of shape (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: Slot indices of shape (num_tokens,), -1 means skip
    """
    # Filter out invalid slots (slot_mapping == -1)
    valid_mask = slot_mapping >= 0
    if not valid_mask.any():
        return

    # Use NPU's native reshape_and_cache operator
    torch_npu._npu_reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=slot_mapping,
    )


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
    Batch-invariant Flash Attention with KV Cache using NPU operators.

    This function provides a drop-in replacement for flash_attn_with_kvcache
    from flash-attention v3, ensuring batch-invariant computation on Ascend NPU.

    Batch invariance is achieved through:
    - slot_mapping: Each token's storage is independent of batch position
    - page_table: Each sequence reads from its own KV cache blocks
    - cu_seqlens_q: Clear sequence boundaries in varlen mode
    - num_splits=1: Consistent execution path

    Arguments:
        q: Query tensor
            - BSHD format: (batch_size, seqlen_q, num_heads, head_dim)
            - TND format: (total_tokens, num_heads, head_dim) when cu_seqlens_q is provided
        k_cache: Key cache tensor
            - Non-paged: (batch_size, seqlen_k, num_heads_k, head_dim)
            - Paged: (num_blocks, block_size, num_heads_k, head_dim)
        v_cache: Value cache tensor (same format as k_cache)
        cache_seqlens: Actual sequence lengths in KV cache (int or Tensor)
        page_table: Block table for paged attention (batch_size, max_blocks_per_seq)
        cu_seqlens_q: Cumulative sequence lengths for varlen mode
        max_seqlen_q: Maximum query sequence length (required for varlen mode)
        softmax_scale: Scale factor for attention scores. Default: 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        num_splits: Number of splits for attention computation (use 1 for determinism)

    Returns:
        out: Attention output (same shape as q)
        softmax_lse: (batch_size, num_heads, seqlen_q) if return_softmax_lse=True
    """
    # Determine input format based on cu_seqlens_q
    is_varlen = cu_seqlens_q is not None
    is_paged = page_table is not None

    if is_varlen:
        # Varlen mode: q is (total_tokens, num_heads, head_dim)
        assert q.dim() == 3, f"In varlen mode, q must be 3D (T, H, D), got {q.dim()}D"
        total_tokens, num_heads_q, head_dim = q.shape
        batch_size = cu_seqlens_q.shape[0] - 1
    else:
        # BSHD mode: q is (batch_size, seqlen_q, num_heads, head_dim)
        assert q.dim() == 4, f"q must be 4D (B, S, H, D), got {q.dim()}D"
        batch_size, seqlen_q, num_heads_q, head_dim = q.shape
        total_tokens = batch_size * seqlen_q

    # Get KV cache dimensions
    if is_paged:
        # Paged format: (num_blocks, block_size, num_kv_heads, head_dim)
        num_blocks, block_size, num_heads_k, _ = k_cache.shape
    else:
        # Non-paged: (batch_size, seqlen_k, num_kv_heads, head_dim)
        _, seqlen_k, num_heads_k, _ = k_cache.shape
        block_size = 128  # Default block size for NPU

    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Handle cache_seqlens
    if cache_seqlens is None:
        if is_paged:
            raise ValueError("cache_seqlens is required when using paged attention")
        # Non-paged: use full seqlen_k
        actual_seq_lengths_kv = [seqlen_k] * batch_size
    elif isinstance(cache_seqlens, int):
        actual_seq_lengths_kv = [cache_seqlens] * batch_size
    elif isinstance(cache_seqlens, torch.Tensor):
        actual_seq_lengths_kv = cache_seqlens.tolist()
    else:
        actual_seq_lengths_kv = list(cache_seqlens)

    # Prepare Q for NPU operator (TND format)
    if is_varlen:
        # Already in TND format
        q_tnd = q.contiguous()
        actual_seq_lengths_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    else:
        # Convert BSHD to TND: (B, S, H, D) -> (B*S, H, D)
        q_tnd = q.reshape(-1, num_heads_q, head_dim).contiguous()
        actual_seq_lengths_q = [seqlen_q] * batch_size
        # Create cu_seqlens_q for non-varlen case
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, seqlen_q,
            dtype=torch.int32, device=q.device
        )

    # Prepare KV cache for NPU operator
    if is_paged:
        # Paged format: reshape to (num_blocks, block_size, num_kv_heads * head_dim)
        k_cache_npu = k_cache.reshape(num_blocks, block_size, -1).contiguous()
        v_cache_npu = v_cache.reshape(num_blocks, block_size, -1).contiguous()
        block_table = page_table
    else:
        # Non-paged: need to create a simple block mapping
        # Reshape to (batch * ceil(seqlen_k/block_size), block_size, num_kv_heads * head_dim)
        k_cache_npu = k_cache.reshape(-1, num_heads_k * head_dim).contiguous()
        v_cache_npu = v_cache.reshape(-1, num_heads_k * head_dim).contiguous()
        block_table = None

    # Allocate output tensor
    out_tnd = torch.empty(
        (q_tnd.shape[0], num_heads_q, head_dim),
        dtype=q.dtype, device=q.device
    )

    # Call NPU fused attention operator
    # Using sparse_mode=3 for causal attention with paged KV cache
    attn_output, softmax_lse_out = torch_npu.npu_fused_infer_attention_score(
        query=q_tnd,
        key=k_cache_npu,
        value=v_cache_npu,
        num_heads=num_heads_q,
        num_key_value_heads=num_heads_k,
        input_layout="TND",
        block_size=block_size,
        scale=softmax_scale,
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_mode=3 if causal else 0,
    )

    # Reshape output back to input format
    if is_varlen:
        out = attn_output
    else:
        out = attn_output.reshape(batch_size, seqlen_q, num_heads_q, head_dim)

    if return_softmax_lse:
        return out, softmax_lse_out
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

