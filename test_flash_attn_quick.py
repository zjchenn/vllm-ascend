#!/usr/bin/env python3
"""
快速验证脚本：测试 flash-attention-with-kvcache 的批不变性实现

使用方法:
    python test_flash_attn_quick.py
"""

import torch
import sys
import os

# 添加 vllm_ascend 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_ascend.batch_invariant import flash_attn_with_kvcache_batch_invariant


def test_basic():
    """基础功能测试"""
    print("\n" + "=" * 80)
    print("测试 1: 基础功能（无新 KV，无 GQA，无 causal）")
    print("=" * 80)

    torch.manual_seed(42)

    batch = 4
    seqlen_q = 1
    seqlen_cache = 32
    nheads = 8
    headdim = 64

    device = 'cpu'
    dtype = torch.float32

    print(f"配置: batch={batch}, seqlen_q={seqlen_q}, seqlen_cache={seqlen_cache}")
    print(f"       nheads={nheads}, headdim={headdim}")

    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k_cache = torch.randn(batch, seqlen_cache, nheads, headdim, dtype=dtype, device=device)
    v_cache = torch.randn(batch, seqlen_cache, nheads, headdim, dtype=dtype, device=device)

    try:
        out1 = flash_attn_with_kvcache_batch_invariant(
            q[:1], k_cache[:1], v_cache[:1]
        )

        out2 = flash_attn_with_kvcache_batch_invariant(
            q, k_cache, v_cache
        )[:1]

        assert out1.shape == out2.shape, f"形状不匹配: {out1.shape} vs {out2.shape}"

        if torch.equal(out1, out2):
            print("✓ 批不变性：完全一致（bitwise）")
        else:
            diff = (out1 - out2).abs().max().item()
            print(f"✓ 批不变性：数值接近（最大差异: {diff:.2e}）")

            if diff > 1e-3:
                print(f"  ⚠️ 警告：差异较大 ({diff:.2e} > 1e-3)")
                return False

        print("✓ 测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_with_new_kv():
    """测试带新 KV 更新"""
    print("=" * 80)
    print("测试 2: 带新 KV 更新")
    print("=" * 80)

    torch.manual_seed(42)

    batch = 2
    seqlen_q = 1
    seqlen_cache = 32
    seqlen_new = 4
    nheads = 8
    headdim = 64

    device = 'cpu'
    dtype = torch.float32

    print(f"配置: batch={batch}, seqlen_new={seqlen_new}")

    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k_cache = torch.randn(batch, seqlen_cache, nheads, headdim, dtype=dtype, device=device)
    v_cache = torch.randn(batch, seqlen_cache, nheads, headdim, dtype=dtype, device=device)
    k_new = torch.randn(batch, seqlen_new, nheads, headdim, dtype=dtype, device=device)
    v_new = torch.randn(batch, seqlen_new, nheads, headdim, dtype=dtype, device=device)

    try:
        k_cache1 = k_cache[:1].clone()
        v_cache1 = v_cache[:1].clone()
        k_cache_full = k_cache.clone()
        v_cache_full = v_cache.clone()

        out1 = flash_attn_with_kvcache_batch_invariant(
            q[:1], k_cache1, v_cache1,
            k=k_new[:1], v=v_new[:1]
        )

        out2 = flash_attn_with_kvcache_batch_invariant(
            q, k_cache_full, v_cache_full,
            k=k_new, v=v_new
        )[:1]

        if torch.equal(out1, out2):
            print("✓ 批不变性：完全一致（bitwise）")
        else:
            diff = (out1 - out2).abs().max().item()
            print(f"✓ 批不变性：数值接近（最大差异: {diff:.2e}）")

            if diff > 1e-3:
                print(f"  ⚠️ 警告：差异较大 ({diff:.2e} > 1e-3)")
                return False

        print("✓ 测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_causal():
    """测试 causal masking"""
    print("=" * 80)
    print("测试 3: Causal Masking")
    print("=" * 80)

    torch.manual_seed(42)

    batch = 2
    seqlen_q = 4
    seqlen_cache = 32
    nheads = 8
    headdim = 64

    device = 'cpu'
    dtype = torch.float32

    print(f"配置: batch={batch}, seqlen_q={seqlen_q} (causal=True)")

    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k_cache = torch.randn(batch, seqlen_cache, nheads, headdim, dtype=dtype, device=device)
    v_cache = torch.randn(batch, seqlen_cache, nheads, headdim, dtype=dtype, device=device)

    try:
        out1 = flash_attn_with_kvcache_batch_invariant(
            q[:1], k_cache[:1], v_cache[:1],
            causal=True
        )

        out2 = flash_attn_with_kvcache_batch_invariant(
            q, k_cache, v_cache,
            causal=True
        )[:1]

        if torch.equal(out1, out2):
            print("✓ 批不变性：完全一致（bitwise）")
        else:
            diff = (out1 - out2).abs().max().item()
            print(f"✓ 批不变性：数值接近（最大差异: {diff:.2e}）")

            if diff > 1e-3:
                print(f"  ⚠️ 警告：差异较大 ({diff:.2e} > 1e-3)")
                return False

        print("✓ 测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gqa():
    """测试 GQA"""
    print("=" * 80)
    print("测试 4: GQA (Grouped Query Attention)")
    print("=" * 80)

    torch.manual_seed(42)

    batch = 2
    seqlen_q = 1
    seqlen_cache = 32
    nheads_q = 16
    nheads_kv = 4
    headdim = 64

    device = 'cpu'
    dtype = torch.float32

    print(f"配置: nheads_q={nheads_q}, nheads_kv={nheads_kv}")
    print(f"       group_size={nheads_q // nheads_kv}")

    q = torch.randn(batch, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
    k_cache = torch.randn(batch, seqlen_cache, nheads_kv, headdim, dtype=dtype, device=device)
    v_cache = torch.randn(batch, seqlen_cache, nheads_kv, headdim, dtype=dtype, device=device)

    try:
        out1 = flash_attn_with_kvcache_batch_invariant(
            q[:1], k_cache[:1], v_cache[:1]
        )

        out2 = flash_attn_with_kvcache_batch_invariant(
            q, k_cache, v_cache
        )[:1]

        if torch.equal(out1, out2):
            print("✓ 批不变性：完全一致（bitwise）")
        else:
            diff = (out1 - out2).abs().max().item()
            print(f"✓ 批不变性：数值接近（最大差异: {diff:.2e}）")

            if diff > 1e-3:
                print(f"  ⚠️ 警告：差异较大 ({diff:.2e} > 1e-3)")
                return False

        print("✓ 测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print(" Flash Attention with KVCache - 批不变性快速验证")
    print("=" * 80)

    results = []

    results.append(("基础功能", test_basic()))
    results.append(("新 KV 更新", test_with_new_kv()))
    results.append(("Causal Masking", test_causal()))
    results.append(("GQA", test_gqa()))

    print("\n" + "=" * 80)
    print(" 测试结果汇总")
    print("=" * 80)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")

    all_passed = all(passed for _, passed in results)

    print("=" * 80)
    if all_passed:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
