# Flash Attention with KVCache - 批不变性实现

## 概述

本实现提供了 flash-attention-with-kvcache 的批不变性版本，确保不同批次大小的计算结果完全一致（bitwise相同）。

## 核心特性

- ✅ **批不变性保证**：`f(x[:1])` 和 `f(x)[:1]` 结果完全一致
- ✅ **Split=1 场景**：不使用 Split-K 并行，确保确定性
- ✅ **GQA/MQA 支持**：支持 Grouped Query Attention
- ✅ **Causal Masking**：支持因果掩码
- ✅ **增量 KV 更新**：支持动态更新 KV cache
- ✅ **Cache 索引映射**：支持 cache_seqlens 和 cache_batch_idx

## 关键设计

### 批不变性保证机制

1. **串行 K/V 遍历**：固定顺序 `for start_n in range(0, N_CTX_K, BLOCK_N)`
2. **Float32 累加**：所有中间结果在 float32 中累加
3. **禁用 TF32**：使用 `allow_tf32=False` 确保精确计算
4. **串行 KV 更新**：只有 `pid_m==0` 的线程块负责更新
5. **固定块大小**：BLOCK_M=16, BLOCK_N=64

## 使用方法

```python
from vllm_ascend.batch_invariant import flash_attn_with_kvcache_batch_invariant

# 基础用法
output = flash_attn_with_kvcache_batch_invariant(
    q=q,              # [batch, seqlen_q, nheads_q, headdim]
    k_cache=k_cache,  # [batch, seqlen_cache, nheads_kv, headdim]
    v_cache=v_cache,  # [batch, seqlen_cache, nheads_kv, headdim]
)

# 带新 KV 更新
output = flash_attn_with_kvcache_batch_invariant(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    k=k_new,          # [batch, seqlen_new, nheads_kv, headdim]
    v=v_new,          # [batch, seqlen_new, nheads_kv, headdim]
)

# 使用 causal masking
output = flash_attn_with_kvcache_batch_invariant(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    causal=True,
)

# GQA (Grouped Query Attention)
# nheads_q 必须能被 nheads_kv 整除
output = flash_attn_with_kvcache_batch_invariant(
    q=q,              # [batch, seqlen_q, 32, headdim]
    k_cache=k_cache,  # [batch, seqlen_cache, 8, headdim]
    v_cache=v_cache,  # [batch, seqlen_cache, 8, headdim]
    # group_size = 32 / 8 = 4
)
```

## 测试

### 快速验证

```bash
cd vllm-ascend
python test_flash_attn_quick.py
```

### 完整测试套件

```bash
# 运行所有批不变性测试
pytest tests/e2e/singlecard/test_batch_invariant.py::test_flash_attn_kvcache_basic_batch_invariance -v
pytest tests/e2e/singlecard/test_batch_invariant.py::test_flash_attn_kvcache_with_new_kv_batch_invariance -v
pytest tests/e2e/singlecard/test_batch_invariant.py::test_flash_attn_kvcache_with_causal_batch_invariance -v
pytest tests/e2e/singlecard/test_batch_invariant.py::test_flash_attn_kvcache_with_gqa_batch_invariance -v
```

## 性能

- **预期性能损失**：20-50%（相比非确定性版本）
- **原因**：串行遍历 K/V 块，没有 Split-K 并行
- **权衡**：牺牲性能换取批不变性和确定性

## 限制

当前实现的限制：

- ❌ **不支持 Paged KVCache**：无 block_table/page_table 支持
- ❌ **不支持滑动窗口**：window_size 参数不支持
- ❌ **Split=1 only**：不支持 Split-K 并行
- ⚠️ **Rotary Embedding**：需在外部预处理

## 文件结构

```
vllm-ascend/
├── vllm_ascend/
│   └── batch_invariant.py                    # 核心实现（新增 ~370 行）
│       ├── _flash_attn_kvcache_batch_invariant_kernel  # Triton kernel
│       └── flash_attn_with_kvcache_batch_invariant     # Python 包装函数
├── tests/e2e/singlecard/
│   └── test_batch_invariant.py               # 测试用例（新增 ~250 行）
└── test_flash_attn_quick.py                  # 快速验证脚本
```

## 未来工作

Phase 2-4 的潜在扩展（参考 [IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)）：

1. **确定性 Split-K**：固定 split 数量，保持批不变性
2. **性能优化**：块大小调优，内存访问优化
3. **Paged KVCache**：支持 block_table
4. **滑动窗口注意力**：支持 window_size

## 参考

- 实现计划：[IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)
- 参考实现：[flash-attention/flash_attn/flash_attn_triton_amd/fwd_decode.py](../../flash-attention/flash_attn/flash_attn_triton_amd/fwd_decode.py)
- 批不变性概念：[batch_invariant_ops/README.md](../../batch_invariant_ops/README.md)
