#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""
Test suite for batch-invariant Flash Attention with KV Cache.

This module tests the batch-invariant properties of the flash attention
implementation, ensuring that:
1. The output is numerically correct compared to a reference implementation
2. The output for a specific sequence is identical regardless of batch composition
3. The implementation handles various head dimensions without UB overflow
"""

import pytest
import torch
import math


def reference_attention(q, k, v, causal=False, softmax_scale=None, softcap=0.0):
    """
    Reference implementation of attention using PyTorch.

    Args:
        q: (batch_size, seqlen_q, num_heads, head_dim)
        k: (batch_size, seqlen_k, num_heads_k, head_dim)
        v: (batch_size, seqlen_k, num_heads_k, head_dim)
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for attention scores
        softcap: Softcap value (0.0 means disabled)

    Returns:
        out: (batch_size, seqlen_q, num_heads, head_dim)
    """
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    _, seqlen_k, num_heads_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Handle GQA/MQA by expanding K and V
    gqa_ratio = num_heads_q // num_heads_k
    if gqa_ratio > 1:
        k = k.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1)
        k = k.reshape(batch_size, seqlen_k, num_heads_q, head_dim)
        v = v.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1)
        v = v.reshape(batch_size, seqlen_k, num_heads_q, head_dim)

    # Transpose to (batch, heads, seqlen, dim) for attention
    q = q.transpose(1, 2)  # (batch, heads, seqlen_q, dim)
    k = k.transpose(1, 2)  # (batch, heads, seqlen_k, dim)
    v = v.transpose(1, 2)  # (batch, heads, seqlen_k, dim)

    # Compute attention scores
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * softmax_scale

    # Apply softcap if enabled
    if softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)

    # Apply causal mask
    if causal:
        # Create causal mask (aligned to bottom-right corner)
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool),
            diagonal=seqlen_k - seqlen_q + 1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention
    out = torch.matmul(attn_weights, v.float())

    # Transpose back to (batch, seqlen, heads, dim)
    out = out.transpose(1, 2).to(q.dtype)

    return out


class TestFlashAttnBatchInvariant:
    """Test class for batch-invariant Flash Attention."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        # Skip if NPU not available
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        self.device = torch.device("npu:0")
        self.dtype = torch.float16

    def _create_random_tensors(
        self,
        batch_size,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_heads_k,
        head_dim,
        seed=42
    ):
        """Create random Q, K, V tensors for testing."""
        torch.manual_seed(seed)
        q = torch.randn(
            batch_size, seqlen_q, num_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        k = torch.randn(
            batch_size, seqlen_k, num_heads_k, head_dim,
            device=self.device, dtype=self.dtype
        )
        v = torch.randn(
            batch_size, seqlen_k, num_heads_k, head_dim,
            device=self.device, dtype=self.dtype
        )
        return q, k, v

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_correctness_vs_reference(self, head_dim, causal):
        """
        Test that batch-invariant flash attention produces correct results
        compared to a reference PyTorch implementation.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        num_heads_k = 8

        q, k, v = self._create_random_tensors(
            batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_dim
        )

        # Compute using batch-invariant implementation
        out_bi = flash_attn_with_kvcache(
            q, k, v,
            causal=causal,
        )

        # Compute using reference implementation
        out_ref = reference_attention(q, k, v, causal=causal)

        # Check numerical correctness (allow some tolerance for float16)
        torch.testing.assert_close(
            out_bi, out_ref,
            rtol=1e-2, atol=1e-2,
            msg=f"Mismatch with head_dim={head_dim}, causal={causal}"
        )

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_batch_invariance(self, head_dim, causal):
        """
        Test that the output for a specific sequence is identical regardless
        of batch composition.

        This is the core batch invariance test:
        - Compute attention for sequence A alone (batch=[A])
        - Compute attention for sequences A and B together (batch=[A, B])
        - Verify that output[A] is bitwise identical in both cases
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        num_heads_k = 8

        # Create sequence A with a fixed seed
        torch.manual_seed(42)
        q_a = torch.randn(
            1, seqlen_q, num_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        k_a = torch.randn(
            1, seqlen_k, num_heads_k, head_dim,
            device=self.device, dtype=self.dtype
        )
        v_a = torch.randn(
            1, seqlen_k, num_heads_k, head_dim,
            device=self.device, dtype=self.dtype
        )

        # Create sequence B with a different seed
        torch.manual_seed(123)
        q_b = torch.randn(
            1, seqlen_q, num_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        k_b = torch.randn(
            1, seqlen_k, num_heads_k, head_dim,
            device=self.device, dtype=self.dtype
        )
        v_b = torch.randn(
            1, seqlen_k, num_heads_k, head_dim,
            device=self.device, dtype=self.dtype
        )

        # Compute output for A alone
        out_a_alone = flash_attn_with_kvcache(
            q_a, k_a, v_a,
            causal=causal,
        )

        # Batch A and B together
        q_ab = torch.cat([q_a, q_b], dim=0)
        k_ab = torch.cat([k_a, k_b], dim=0)
        v_ab = torch.cat([v_a, v_b], dim=0)

        # Compute output for A and B together
        out_ab = flash_attn_with_kvcache(
            q_ab, k_ab, v_ab,
            causal=causal,
        )

        # Extract output for A from the batched computation
        out_a_batched = out_ab[0:1]

        # CRITICAL: The output for A must be IDENTICAL regardless of batch
        # Using exact equality (not approximate) to verify batch invariance
        assert torch.equal(out_a_alone, out_a_batched), (
            f"Batch invariance violated! "
            f"Output for sequence A differs when computed alone vs. in a batch. "
            f"Max diff: {(out_a_alone - out_a_batched).abs().max().item()}"
        )

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_batch_invariance_different_positions(self, head_dim):
        """
        Test batch invariance when the sequence of interest is at different
        positions in the batch.

        Verifies: output[A] is the same whether A is at position 0, 1, or 2
        in the batch.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        num_heads_k = 8

        # Create three sequences
        torch.manual_seed(42)
        q_a = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_a = torch.randn(1, seqlen_k, num_heads_k, head_dim, device=self.device, dtype=self.dtype)
        v_a = torch.randn(1, seqlen_k, num_heads_k, head_dim, device=self.device, dtype=self.dtype)

        torch.manual_seed(123)
        q_b = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_b = torch.randn(1, seqlen_k, num_heads_k, head_dim, device=self.device, dtype=self.dtype)
        v_b = torch.randn(1, seqlen_k, num_heads_k, head_dim, device=self.device, dtype=self.dtype)

        torch.manual_seed(456)
        q_c = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_c = torch.randn(1, seqlen_k, num_heads_k, head_dim, device=self.device, dtype=self.dtype)
        v_c = torch.randn(1, seqlen_k, num_heads_k, head_dim, device=self.device, dtype=self.dtype)

        # A at position 0: [A, B, C]
        q_abc = torch.cat([q_a, q_b, q_c], dim=0)
        k_abc = torch.cat([k_a, k_b, k_c], dim=0)
        v_abc = torch.cat([v_a, v_b, v_c], dim=0)
        out_abc = flash_attn_with_kvcache(q_abc, k_abc, v_abc, causal=True)
        out_a_pos0 = out_abc[0:1]

        # A at position 1: [B, A, C]
        q_bac = torch.cat([q_b, q_a, q_c], dim=0)
        k_bac = torch.cat([k_b, k_a, k_c], dim=0)
        v_bac = torch.cat([v_b, v_a, v_c], dim=0)
        out_bac = flash_attn_with_kvcache(q_bac, k_bac, v_bac, causal=True)
        out_a_pos1 = out_bac[1:2]

        # A at position 2: [B, C, A]
        q_bca = torch.cat([q_b, q_c, q_a], dim=0)
        k_bca = torch.cat([k_b, k_c, k_a], dim=0)
        v_bca = torch.cat([v_b, v_c, v_a], dim=0)
        out_bca = flash_attn_with_kvcache(q_bca, k_bca, v_bca, causal=True)
        out_a_pos2 = out_bca[2:3]

        # Verify all outputs for A are identical
        assert torch.equal(out_a_pos0, out_a_pos1), (
            f"Batch invariance violated: A at pos 0 vs pos 1 differ. "
            f"Max diff: {(out_a_pos0 - out_a_pos1).abs().max().item()}"
        )
        assert torch.equal(out_a_pos0, out_a_pos2), (
            f"Batch invariance violated: A at pos 0 vs pos 2 differ. "
            f"Max diff: {(out_a_pos0 - out_a_pos2).abs().max().item()}"
        )

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_various_head_dims(self, head_dim):
        """
        Test that the implementation works correctly with various head dimensions.

        This tests the block size computation logic that adjusts BLOCK_N based
        on head_dim to avoid NPU UB overflow.

        Note: head_dim=192 and 256 are excluded due to NPU UB memory constraints.
        With the current BLOCK sizes, these larger head dimensions would cause
        UB overflow errors.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 32
        seqlen_k = 64
        num_heads = 4
        num_heads_k = 4

        q, k, v = self._create_random_tensors(
            batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_dim
        )

        # Should not raise any errors (especially UB overflow)
        out = flash_attn_with_kvcache(q, k, v, causal=True)

        # Basic sanity check
        assert out.shape == q.shape
        assert not torch.isnan(out).any(), f"NaN values in output for head_dim={head_dim}"
        assert not torch.isinf(out).any(), f"Inf values in output for head_dim={head_dim}"

    @pytest.mark.parametrize("gqa_ratio", [1, 2, 4, 8])
    def test_gqa_mqa(self, gqa_ratio):
        """
        Test Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).

        GQA ratio determines how many query heads share a single KV head.
        - gqa_ratio=1: Standard attention (num_heads_q == num_heads_k)
        - gqa_ratio=2: Each KV head is shared by 2 query heads
        - gqa_ratio=8: MQA (single KV head for all query heads)
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 32
        seqlen_k = 64
        head_dim = 64
        num_heads_q = 8
        num_heads_k = num_heads_q // gqa_ratio

        q, k, v = self._create_random_tensors(
            batch_size, seqlen_q, seqlen_k, num_heads_q, num_heads_k, head_dim
        )

        # Compute using batch-invariant implementation
        out_bi = flash_attn_with_kvcache(q, k, v, causal=True)

        # Compute using reference implementation
        out_ref = reference_attention(q, k, v, causal=True)

        # Check numerical correctness
        torch.testing.assert_close(
            out_bi, out_ref,
            rtol=1e-2, atol=1e-2,
            msg=f"Mismatch with gqa_ratio={gqa_ratio}"
        )

    @pytest.mark.skip(reason="Softcap not supported by NPU fused attention operator")
    def test_softcap(self):
        """
        Test softcap functionality.

        Softcap limits the range of attention logits by applying tanh:
        scores = softcap * tanh(scores / softcap)

        Note: This is currently not supported by the NPU fused attention operator.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        q, k, v = self._create_random_tensors(
            batch_size, seqlen_q, seqlen_k, num_heads, num_heads, head_dim
        )

        softcap = 50.0

        # Compute using batch-invariant implementation
        out_bi = flash_attn_with_kvcache(q, k, v, causal=True, softcap=softcap)

        # Compute using reference implementation
        out_ref = reference_attention(q, k, v, causal=True, softcap=softcap)

        # Check numerical correctness
        torch.testing.assert_close(
            out_bi, out_ref,
            rtol=1e-2, atol=1e-2,
            msg="Mismatch with softcap"
        )

    def test_return_softmax_lse(self):
        """
        Test that log-sum-exp values are returned correctly when requested.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        q, k, v = self._create_random_tensors(
            batch_size, seqlen_q, seqlen_k, num_heads, num_heads, head_dim
        )

        # Test with return_softmax_lse=True
        result = flash_attn_with_kvcache(
            q, k, v,
            causal=True,
            return_softmax_lse=True
        )

        assert isinstance(result, tuple), "Expected tuple when return_softmax_lse=True"
        assert len(result) == 2, "Expected (out, softmax_lse) tuple"

        out, softmax_lse = result
        assert out.shape == q.shape
        # Note: NPU operator may return different shape for softmax_lse
        assert softmax_lse is not None, "softmax_lse should not be None"


class TestFlashAttnIntegration:
    """Test integration and edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        self.device = torch.device("npu:0")
        self.dtype = torch.float16

    def test_paged_attention(self):
        """
        Test paged attention mode with page_table parameter.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 1
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64
        block_size = 128
        num_blocks = 16
        max_blocks_per_seq = 4

        torch.manual_seed(42)
        # Varlen format for Q: (total_tokens, num_heads, head_dim)
        q = torch.randn(batch_size, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)

        # Paged KV cache: (num_blocks, block_size, num_kv_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                             device=self.device, dtype=self.dtype)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                             device=self.device, dtype=self.dtype)

        # Page table: (batch_size, max_blocks_per_seq)
        page_table = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq),
                                   device=self.device, dtype=torch.int32)

        # Cache sequence lengths
        cache_seqlens = torch.tensor([64, 128], device=self.device, dtype=torch.int32)

        # cu_seqlens_q for varlen mode
        cu_seqlens_q = torch.arange(0, batch_size + 1, device=self.device, dtype=torch.int32)

        # Should work without errors
        out = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            causal=True
        )

        assert out.shape == q.shape
        assert not torch.isnan(out).any(), "NaN values in paged attention output"
        assert not torch.isinf(out).any(), "Inf values in paged attention output"

    def test_varlen_mode(self):
        """
        Test variable length mode with cu_seqlens_q parameter.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 3
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64

        # Variable query lengths: [16, 32, 24]
        query_lens = [16, 32, 24]
        total_q_tokens = sum(query_lens)
        max_seqlen_q = max(query_lens)

        # KV cache length (same for all sequences in this test)
        seqlen_k = 64

        torch.manual_seed(42)
        # Varlen format: (total_tokens, num_heads, head_dim)
        q = torch.randn(total_q_tokens, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)
        k = torch.randn(batch_size, seqlen_k, num_kv_heads, head_dim,
                       device=self.device, dtype=self.dtype)
        v = torch.randn(batch_size, seqlen_k, num_kv_heads, head_dim,
                       device=self.device, dtype=self.dtype)

        # cu_seqlens_q: cumulative sequence lengths
        cu_seqlens_q = torch.tensor([0, 16, 48, 72], device=self.device, dtype=torch.int32)
        cache_seqlens = torch.tensor([seqlen_k] * batch_size, device=self.device, dtype=torch.int32)

        # Should work without errors
        out = flash_attn_with_kvcache(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            causal=True
        )

        assert out.shape == q.shape
        assert not torch.isnan(out).any(), "NaN values in varlen output"
        assert not torch.isinf(out).any(), "Inf values in varlen output"

    def test_paged_attention_batch_invariance(self):
        """
        Test that paged attention maintains batch invariance.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        num_heads = 8
        num_kv_heads = 8
        head_dim = 64
        block_size = 128
        num_blocks = 16
        max_blocks_per_seq = 4

        torch.manual_seed(42)
        # Create sequence A
        q_a = torch.randn(1, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_cache_a = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        v_cache_a = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        page_table_a = torch.randint(0, num_blocks, (1, max_blocks_per_seq),
                                     device=self.device, dtype=torch.int32)
        cache_seqlen_a = torch.tensor([64], device=self.device, dtype=torch.int32)

        torch.manual_seed(123)
        # Create sequence B
        q_b = torch.randn(1, num_heads, head_dim, device=self.device, dtype=self.dtype)
        page_table_b = torch.randint(0, num_blocks, (1, max_blocks_per_seq),
                                     device=self.device, dtype=torch.int32)
        cache_seqlen_b = torch.tensor([96], device=self.device, dtype=torch.int32)

        # Compute A alone
        cu_seqlens_a = torch.tensor([0, 1], device=self.device, dtype=torch.int32)
        out_a_alone = flash_attn_with_kvcache(
            q_a, k_cache_a, v_cache_a,
            page_table=page_table_a,
            cache_seqlens=cache_seqlen_a,
            cu_seqlens_q=cu_seqlens_a,
            max_seqlen_q=1,
            causal=True
        )

        # Compute A and B together
        q_ab = torch.cat([q_a, q_b], dim=0)
        page_table_ab = torch.cat([page_table_a, page_table_b], dim=0)
        cache_seqlens_ab = torch.cat([cache_seqlen_a, cache_seqlen_b], dim=0)
        cu_seqlens_ab = torch.tensor([0, 1, 2], device=self.device, dtype=torch.int32)

        out_ab = flash_attn_with_kvcache(
            q_ab, k_cache_a, v_cache_a,  # Using same cache for simplicity
            page_table=page_table_ab,
            cache_seqlens=cache_seqlens_ab,
            cu_seqlens_q=cu_seqlens_ab,
            max_seqlen_q=1,
            causal=True
        )
        out_a_batched = out_ab[0:1]

        # Verify batch invariance
        assert torch.equal(out_a_alone, out_a_batched), (
            f"Paged attention batch invariance violated! "
            f"Max diff: {(out_a_alone - out_a_batched).abs().max().item()}"
        )

    def test_cache_seqlens(self):
        """
        Test that cache_seqlens parameter works correctly.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        batch_size = 2
        seqlen_q = 1
        seqlen_k_max = 128  # Max cache size
        seqlen_k_used = 64  # Actually used
        num_heads = 8
        head_dim = 64

        torch.manual_seed(42)
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k = torch.randn(batch_size, seqlen_k_max, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v = torch.randn(batch_size, seqlen_k_max, num_heads, head_dim, device=self.device, dtype=self.dtype)

        # Use only first seqlen_k_used positions
        out = flash_attn_with_kvcache(
            q, k, v,
            cache_seqlens=seqlen_k_used,
            causal=True
        )

        # Compare with manual slicing
        k_sliced = k[:, :seqlen_k_used, :, :]
        v_sliced = v[:, :seqlen_k_used, :, :]
        out_ref = flash_attn_with_kvcache(
            q, k_sliced, v_sliced,
            causal=True
        )

        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# New Test Classes for Enhanced Coverage
# =============================================================================


class TestNativeImplementationBatchVariance:
    """
    Test that native (non-batch-invariant) implementations FAIL batch invariance tests.

    This validates our test methodology: if native implementations pass these tests,
    then our tests are not actually testing batch invariance. Native implementations
    should show variance when batch composition changes.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        self.device = torch.device("npu:0")
        self.dtype = torch.float16

    def _native_attention(self, q, k, v, causal=False):
        """
        Native PyTorch attention implementation (NOT batch-invariant).
        Uses standard matmul which may have non-deterministic accumulation order.
        """
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        _, seqlen_k, _, _ = k.shape

        softmax_scale = head_dim ** (-0.5)

        # Transpose to (batch, heads, seqlen, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use standard matmul (may have non-deterministic behavior)
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

        if causal:
            mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool),
                diagonal=seqlen_k - seqlen_q + 1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        return out.transpose(1, 2)

    def test_native_attention_is_not_batch_invariant(self):
        """
        Verify that native PyTorch attention is NOT batch-invariant.

        This test is expected to show that the native implementation produces
        different results for the same sequence when batch composition changes.
        If this test passes (native impl shows variance), it validates our testing approach.
        """
        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        # Create sequence A
        torch.manual_seed(42)
        q_a = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_a = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v_a = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)

        # Create sequence B
        torch.manual_seed(123)
        q_b = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_b = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v_b = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)

        # Compute A alone multiple times to check for variance
        results_alone = []
        for _ in range(5):
            out = self._native_attention(q_a, k_a, v_a, causal=True)
            results_alone.append(out.clone())

        # Compute A in batch with B
        q_ab = torch.cat([q_a, q_b], dim=0)
        k_ab = torch.cat([k_a, k_b], dim=0)
        v_ab = torch.cat([v_a, v_b], dim=0)

        results_batched = []
        for _ in range(5):
            out_ab = self._native_attention(q_ab, k_ab, v_ab, causal=True)
            results_batched.append(out_ab[0:1].clone())

        # Check if there's any variance in native implementation
        # We expect native implementations may show variance across runs or batch configs
        all_equal_alone = all(torch.equal(results_alone[0], r) for r in results_alone[1:])
        all_equal_batched = all(torch.equal(results_batched[0], r) for r in results_batched[1:])
        alone_vs_batched_equal = torch.equal(results_alone[0], results_batched[0])

        # Log the results for analysis
        print(f"\n[Native Attention Analysis]")
        print(f"  All 'alone' runs identical: {all_equal_alone}")
        print(f"  All 'batched' runs identical: {all_equal_batched}")
        print(f"  'Alone' vs 'Batched' identical: {alone_vs_batched_equal}")

        if not alone_vs_batched_equal:
            max_diff = (results_alone[0] - results_batched[0]).abs().max().item()
            print(f"  Max diff (alone vs batched): {max_diff}")

        # Note: This test documents behavior rather than asserting failure
        # Native implementations may or may not be deterministic depending on hardware

    def test_our_implementation_is_batch_invariant(self):
        """
        Verify that OUR implementation IS batch-invariant (control test).

        This confirms that our flash_attn_with_kvcache produces identical
        results regardless of batch composition.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        # Create sequence A
        torch.manual_seed(42)
        q_a = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_a = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v_a = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)

        # Create sequence B
        torch.manual_seed(123)
        q_b = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_b = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v_b = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)

        # Compute A alone
        out_a_alone = flash_attn_with_kvcache(q_a, k_a, v_a, causal=True)

        # Compute A in batch with B
        q_ab = torch.cat([q_a, q_b], dim=0)
        k_ab = torch.cat([k_a, k_b], dim=0)
        v_ab = torch.cat([v_a, v_b], dim=0)
        out_ab = flash_attn_with_kvcache(q_ab, k_ab, v_ab, causal=True)
        out_a_batched = out_ab[0:1]

        # Our implementation MUST be batch-invariant
        assert torch.equal(out_a_alone, out_a_batched), (
            f"OUR implementation failed batch invariance! "
            f"Max diff: {(out_a_alone - out_a_batched).abs().max().item()}"
        )

    def test_compare_native_vs_ours_batch_invariance(self):
        """
        Direct comparison: native implementation should fail batch invariance
        while our implementation should pass.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        # Create sequences
        torch.manual_seed(42)
        q_a = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_a = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v_a = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)

        torch.manual_seed(123)
        q_b = torch.randn(1, seqlen_q, num_heads, head_dim, device=self.device, dtype=self.dtype)
        k_b = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)
        v_b = torch.randn(1, seqlen_k, num_heads, head_dim, device=self.device, dtype=self.dtype)

        # Batch tensors
        q_ab = torch.cat([q_a, q_b], dim=0)
        k_ab = torch.cat([k_a, k_b], dim=0)
        v_ab = torch.cat([v_a, v_b], dim=0)

        # Native implementation
        native_alone = self._native_attention(q_a, k_a, v_a, causal=True)
        native_batched = self._native_attention(q_ab, k_ab, v_ab, causal=True)[0:1]
        native_is_invariant = torch.equal(native_alone, native_batched)

        # Our implementation
        ours_alone = flash_attn_with_kvcache(q_a, k_a, v_a, causal=True)
        ours_batched = flash_attn_with_kvcache(q_ab, k_ab, v_ab, causal=True)[0:1]
        ours_is_invariant = torch.equal(ours_alone, ours_batched)

        print(f"\n[Batch Invariance Comparison]")
        print(f"  Native implementation is batch-invariant: {native_is_invariant}")
        print(f"  Our implementation is batch-invariant: {ours_is_invariant}")

        if not native_is_invariant:
            print(f"  Native max diff: {(native_alone - native_batched).abs().max().item()}")

        # Our implementation MUST pass
        assert ours_is_invariant, "Our implementation must be batch-invariant!"

        # Document that native may or may not be invariant
        # (depends on hardware/driver determinism)


class TestBatchSizeScaling:
    """
    Test batch size scaling from small to large values.

    Tests various batch sizes to verify:
    1. Correctness at all batch sizes
    2. Batch invariance holds at all batch sizes
    3. No numerical issues (NaN/Inf) at large batch sizes
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        self.device = torch.device("npu:0")
        self.dtype = torch.float16

    # Typical batch sizes covering small, medium, and large scales
    # Reduced max values to avoid OOM on NPU devices
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

    # Extended batch sizes for stress testing (optional, may require more memory)
    EXTENDED_BATCH_SIZES = [256, 512, 1024]

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_correctness_at_batch_size(self, batch_size):
        """
        Test numerical correctness at various batch sizes.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        torch.manual_seed(42)
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)
        k = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)
        v = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)

        # Compute using our implementation
        out = flash_attn_with_kvcache(q, k, v, causal=True)

        # Basic sanity checks
        assert out.shape == q.shape, f"Shape mismatch at batch_size={batch_size}"
        assert not torch.isnan(out).any(), f"NaN at batch_size={batch_size}"
        assert not torch.isinf(out).any(), f"Inf at batch_size={batch_size}"

        # Compare with reference for first few elements
        out_ref = reference_attention(q[:min(4, batch_size)],
                                      k[:min(4, batch_size)],
                                      v[:min(4, batch_size)],
                                      causal=True)
        torch.testing.assert_close(
            out[:min(4, batch_size)], out_ref,
            rtol=1e-2, atol=1e-2,
            msg=f"Correctness check failed at batch_size={batch_size}"
        )

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_batch_invariance_at_batch_size(self, batch_size):
        """
        Test that batch invariance holds at various batch sizes.

        Method: Pick a random sequence, compute it alone and in a batch of size N,
        verify results are identical.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 32
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        # Create the target sequence
        torch.manual_seed(42)
        q_target = torch.randn(1, seqlen_q, num_heads, head_dim,
                              device=self.device, dtype=self.dtype)
        k_target = torch.randn(1, seqlen_k, num_heads, head_dim,
                              device=self.device, dtype=self.dtype)
        v_target = torch.randn(1, seqlen_k, num_heads, head_dim,
                              device=self.device, dtype=self.dtype)

        # Compute target alone
        out_alone = flash_attn_with_kvcache(q_target, k_target, v_target, causal=True)

        # Create filler sequences for the batch
        torch.manual_seed(123)
        q_fillers = torch.randn(batch_size - 1, seqlen_q, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        k_fillers = torch.randn(batch_size - 1, seqlen_k, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        v_fillers = torch.randn(batch_size - 1, seqlen_k, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)

        # Put target at position 0
        q_batch = torch.cat([q_target, q_fillers], dim=0)
        k_batch = torch.cat([k_target, k_fillers], dim=0)
        v_batch = torch.cat([v_target, v_fillers], dim=0)

        out_batch = flash_attn_with_kvcache(q_batch, k_batch, v_batch, causal=True)
        out_from_batch = out_batch[0:1]

        # Must be exactly equal
        assert torch.equal(out_alone, out_from_batch), (
            f"Batch invariance failed at batch_size={batch_size}! "
            f"Max diff: {(out_alone - out_from_batch).abs().max().item()}"
        )

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_variable_seqlens_at_batch_size(self, batch_size):
        """
        Test variable sequence lengths at various batch sizes.

        Each sequence in the batch has a different KV cache length.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 1  # Decode scenario
        seqlen_k_max = 128
        num_heads = 8
        head_dim = 64

        torch.manual_seed(42)
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)
        k = torch.randn(batch_size, seqlen_k_max, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)
        v = torch.randn(batch_size, seqlen_k_max, num_heads, head_dim,
                       device=self.device, dtype=self.dtype)

        # Create variable sequence lengths (between 32 and 128)
        cache_seqlens = torch.randint(32, seqlen_k_max + 1, (batch_size,),
                                      device=self.device, dtype=torch.int32)

        # Should not raise errors
        out = flash_attn_with_kvcache(
            q, k, v,
            cache_seqlens=cache_seqlens,
            causal=True
        )

        assert out.shape == q.shape
        assert not torch.isnan(out).any(), f"NaN with variable seqlens at batch_size={batch_size}"
        assert not torch.isinf(out).any(), f"Inf with variable seqlens at batch_size={batch_size}"

    @pytest.mark.parametrize("batch_size", [1, 16, 64, 256])
    def test_batch_invariance_with_variable_seqlens(self, batch_size):
        """
        Test batch invariance with variable sequence lengths.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 1
        seqlen_k_max = 128
        num_heads = 8
        head_dim = 64

        # Create target sequence with specific length
        torch.manual_seed(42)
        q_target = torch.randn(1, seqlen_q, num_heads, head_dim,
                              device=self.device, dtype=self.dtype)
        k_target = torch.randn(1, seqlen_k_max, num_heads, head_dim,
                              device=self.device, dtype=self.dtype)
        v_target = torch.randn(1, seqlen_k_max, num_heads, head_dim,
                              device=self.device, dtype=self.dtype)
        target_seqlen = torch.tensor([64], device=self.device, dtype=torch.int32)

        # Compute alone
        out_alone = flash_attn_with_kvcache(
            q_target, k_target, v_target,
            cache_seqlens=target_seqlen,
            causal=True
        )

        # Create batch with different sequence lengths
        torch.manual_seed(123)
        q_fillers = torch.randn(batch_size - 1, seqlen_q, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        k_fillers = torch.randn(batch_size - 1, seqlen_k_max, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        v_fillers = torch.randn(batch_size - 1, seqlen_k_max, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
        filler_seqlens = torch.randint(32, seqlen_k_max + 1, (batch_size - 1,),
                                       device=self.device, dtype=torch.int32)

        q_batch = torch.cat([q_target, q_fillers], dim=0)
        k_batch = torch.cat([k_target, k_fillers], dim=0)
        v_batch = torch.cat([v_target, v_fillers], dim=0)
        batch_seqlens = torch.cat([target_seqlen, filler_seqlens], dim=0)

        out_batch = flash_attn_with_kvcache(
            q_batch, k_batch, v_batch,
            cache_seqlens=batch_seqlens,
            causal=True
        )
        out_from_batch = out_batch[0:1]

        assert torch.equal(out_alone, out_from_batch), (
            f"Batch invariance with varlen failed at batch_size={batch_size}! "
            f"Max diff: {(out_alone - out_from_batch).abs().max().item()}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", EXTENDED_BATCH_SIZES)
    def test_large_batch_sizes(self, batch_size):
        """
        Stress test with very large batch sizes.

        Marked as 'slow' - run with: pytest -m slow
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        seqlen_q = 1  # Decode scenario to reduce memory
        seqlen_k = 64
        num_heads = 8
        head_dim = 64

        try:
            torch.manual_seed(42)
            q = torch.randn(batch_size, seqlen_q, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)
            k = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)
            v = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)

            out = flash_attn_with_kvcache(q, k, v, causal=True)

            assert out.shape == q.shape
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping batch_size={batch_size} due to OOM")
            raise


class TestPerformanceBenchmark:
    """
    Performance benchmarks for batch-invariant flash attention.

    These tests measure execution time and throughput at various batch sizes.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        self.device = torch.device("npu:0")
        self.dtype = torch.float16

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_decode_performance(self, batch_size):
        """
        Benchmark decode performance (seqlen_q=1) at various batch sizes.

        Note: Parameters reduced to avoid OOM on NPU devices.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache
        import time

        seqlen_q = 1
        seqlen_k = 128  # Reduced from 512
        num_heads = 8   # Reduced from 32
        head_dim = 64   # Reduced from 128

        try:
            torch.manual_seed(42)
            q = torch.randn(batch_size, seqlen_q, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)
            k = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)
            v = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)

            # Warmup
            for _ in range(3):
                _ = flash_attn_with_kvcache(q, k, v, causal=True)
            torch.npu.synchronize()

            # Benchmark
            num_iterations = 100
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = flash_attn_with_kvcache(q, k, v, causal=True)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start

            avg_time_ms = (elapsed / num_iterations) * 1000
            throughput = batch_size / (elapsed / num_iterations)

            print(f"\n[Decode Performance] batch_size={batch_size}")
            print(f"  Avg time: {avg_time_ms:.3f} ms")
            print(f"  Throughput: {throughput:.1f} sequences/sec")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping batch_size={batch_size} due to OOM")
            raise

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_prefill_performance(self, batch_size):
        """
        Benchmark prefill performance (longer seqlen_q) at various batch sizes.

        Note: Parameters reduced to avoid OOM on NPU devices.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache
        import time

        seqlen_q = 64   # Reduced from 256
        seqlen_k = 64   # Reduced from 256
        num_heads = 8   # Reduced from 32
        head_dim = 64   # Reduced from 128

        try:
            torch.manual_seed(42)
            q = torch.randn(batch_size, seqlen_q, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)
            k = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)
            v = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                           device=self.device, dtype=self.dtype)

            # Warmup
            for _ in range(3):
                _ = flash_attn_with_kvcache(q, k, v, causal=True)
            torch.npu.synchronize()

            # Benchmark
            num_iterations = 20
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = flash_attn_with_kvcache(q, k, v, causal=True)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start

            avg_time_ms = (elapsed / num_iterations) * 1000
            tokens_per_sec = (batch_size * seqlen_q) / (elapsed / num_iterations)

            print(f"\n[Prefill Performance] batch_size={batch_size}")
            print(f"  Avg time: {avg_time_ms:.3f} ms")
            print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping batch_size={batch_size} due to OOM")
            raise

    @pytest.mark.benchmark
    def test_batch_scaling_efficiency(self):
        """
        Measure how performance scales with batch size.

        Ideally, throughput should increase linearly with batch size
        until we hit memory bandwidth limits.

        Note: Parameters are reduced to avoid OOM on NPU devices.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache
        import time

        seqlen_q = 1
        seqlen_k = 64  # Reduced from 256
        num_heads = 8  # Reduced from 32
        head_dim = 64  # Reduced from 128

        batch_sizes = [1, 2, 4, 8, 16, 32, 64]  # Reduced max from 256
        results = []

        for batch_size in batch_sizes:
            try:
                torch.manual_seed(42)
                q = torch.randn(batch_size, seqlen_q, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
                k = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)
                v = torch.randn(batch_size, seqlen_k, num_heads, head_dim,
                               device=self.device, dtype=self.dtype)

                # Warmup
                for _ in range(3):
                    _ = flash_attn_with_kvcache(q, k, v, causal=True)
                torch.npu.synchronize()

                # Benchmark
                num_iterations = 50
                start = time.perf_counter()
                for _ in range(num_iterations):
                    _ = flash_attn_with_kvcache(q, k, v, causal=True)
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start

                avg_time_ms = (elapsed / num_iterations) * 1000
                throughput = batch_size / (elapsed / num_iterations)
                results.append((batch_size, avg_time_ms, throughput))

                # Clean up to free memory
                del q, k, v
                torch.npu.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Skipping batch_size={batch_size} due to OOM")
                    break
                raise

        print("\n[Batch Scaling Efficiency]")
        print("  Batch Size | Avg Time (ms) | Throughput (seq/s) | Efficiency")
        print("  " + "-" * 60)

        base_throughput = results[0][2]
        for batch_size, avg_time, throughput in results:
            efficiency = throughput / (batch_size * base_throughput) * 100
            print(f"  {batch_size:>10} | {avg_time:>13.3f} | {throughput:>18.1f} | {efficiency:>8.1f}%")
