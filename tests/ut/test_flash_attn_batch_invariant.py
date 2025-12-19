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

    @pytest.mark.parametrize("head_dim", [64, 128, 192, 256])
    def test_various_head_dims(self, head_dim):
        """
        Test that the implementation works correctly with various head dimensions.

        This tests the block size computation logic that adjusts BLOCK_N based
        on head_dim to avoid NPU UB overflow.
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

    def test_softcap(self):
        """
        Test softcap functionality.

        Softcap limits the range of attention logits by applying tanh:
        scores = softcap * tanh(scores / softcap)
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
        assert softmax_lse.shape == (batch_size, num_heads, seqlen_q)
        assert softmax_lse.dtype == torch.float32


class TestFlashAttnIntegration:
    """Test integration and edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        self.device = torch.device("npu:0")
        self.dtype = torch.float16

    def test_unsupported_features_raise_errors(self):
        """
        Test that unsupported features raise appropriate NotImplementedError.
        """
        from vllm_ascend.batch_invariant import flash_attn_with_kvcache

        q = torch.randn(1, 32, 8, 64, device=self.device, dtype=self.dtype)
        k = torch.randn(1, 64, 8, 64, device=self.device, dtype=self.dtype)
        v = torch.randn(1, 64, 8, 64, device=self.device, dtype=self.dtype)

        # Test k/v appending not supported
        k_new = torch.randn(1, 1, 8, 64, device=self.device, dtype=self.dtype)
        with pytest.raises(NotImplementedError, match="Appending new K/V"):
            flash_attn_with_kvcache(q, k, v, k=k_new, v=k_new)

        # Test rotary not supported
        rotary_cos = torch.randn(64, 32, device=self.device, dtype=self.dtype)
        with pytest.raises(NotImplementedError, match="Rotary embeddings"):
            flash_attn_with_kvcache(q, k, v, rotary_cos=rotary_cos, rotary_sin=rotary_cos)

        # Test paged KV cache not supported
        page_table = torch.zeros(1, 4, device=self.device, dtype=torch.int32)
        with pytest.raises(NotImplementedError, match="Paged KV cache"):
            flash_attn_with_kvcache(q, k, v, page_table=page_table)

        # Test varlen mode not supported
        cu_seqlens = torch.tensor([0, 32], device=self.device, dtype=torch.int32)
        with pytest.raises(NotImplementedError, match="Variable length"):
            flash_attn_with_kvcache(q, k, v, cu_seqlens_q=cu_seqlens)

        # Test sliding window not supported
        with pytest.raises(NotImplementedError, match="Sliding window"):
            flash_attn_with_kvcache(q, k, v, window_size=(128, 128))

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
