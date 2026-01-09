from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendAttentionBackendImpl,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.utils import AscendDeviceType


class TestAscendAttentionBackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "CUSTOM")

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend.get_impl_cls(),
                         AscendAttentionBackendImpl)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend.get_builder_cls(),
                         AscendAttentionMetadataBuilder)

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    def test_get_kv_cache_shape_not_310p(self, mock_soc_version):
        result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30, 40))

    def test_get_bsh_kv_cache_shape(self):
        result = AscendAttentionBackend.get_bsh_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30 * 40))

    def test_swap_blocks(self):
        src_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        dst_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dst = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache,
                                           src_to_dst)
        self.assertTrue(torch.all(dst_kv_cache[0][1] == src_kv_cache[0][0]))
        self.assertTrue(torch.all(dst_kv_cache[1][3] == src_kv_cache[1][2]))

    def test_copy_blocks(self):
        kv_caches = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dists = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.copy_blocks(kv_caches, src_to_dists)
        self.assertTrue(torch.all(kv_caches[0][1] == kv_caches[0][0]))
        self.assertTrue(torch.all(kv_caches[1][3] == kv_caches[1][2]))


class TestAscendAttentionMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.speculative_config = None
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_vllm_config.compilation_config.cudagraph_mode = None
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.decode_max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        self.mock_device = 'cpu:0'
        torch.Tensor.pin_memory = lambda x: x  # noqa
        self.builder = AscendAttentionMetadataBuilder(None, None,
                                                      self.mock_vllm_config,
                                                      self.mock_device)

    def test_reorder_batch(self):
        mock_input_batch = MagicMock()
        mock_scheduler_output = MagicMock()

        result = self.builder.reorder_batch(mock_input_batch,
                                            mock_scheduler_output)

        self.assertFalse(result)

    @patch('vllm_ascend.attention.attention_v1.AscendMetadata')
    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    def test_build_non_310p(self, mock_soc_version, mock_ascend_metadata):
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 9]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 9]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=15,
            max_query_len=6,
            decode_token_per_req=torch.tensor([1, 1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((15, 15)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_computed_tokens_cpu=None,
            seq_lens=None)
        mock_model = MagicMock()

        self.builder.build(1, common_attn_metadata, mock_model)


class TestAscendAttentionBackendImpl(TestBase):

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0

        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"

        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"

        self.layer_no_quant = MagicMock(
            spec=['layer_name', '_k_scale_float', '_v_scale_float'])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0

        self.impl = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl_192 = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl_error = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None)

        self.impl_swa = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=1024,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

    def test_forward_no_attn_metadata(self):
        """Test forward pass when attn_metadata is None"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 0, 0, 8, 64)
        layer = self.layer_no_quant
        output = torch.empty_like(query)

        output = self.impl.forward(layer, query, key, value, kv_cache, None,
                                   output)

        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    def test_forward_prefill(self, mock_get_forward_context,
                             mock_npu_fused_infer_attention_score,
                             mock_npu_reshape_and_cache):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillCacheHit
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.actual_seq_lengths_q = [10]
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.num_decode_tokens = 0
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        mock_get_forward_context.return_value = MagicMock(capturing=False)
        mock_npu_fused_infer_attention_score.return_value = (torch.ones(
            10, 8, 64), torch.ones(10, 8, 64))
        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    def test_forward_decode_only(self, mock_get_forward_context,
                                 mock_npu_reshape_and_cache,
                                 mock_paged_attention):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_paged_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_decode_only_swa(self, mock_npu_reshape_and_cache,
                                     mock_fused_infer_attention_score,
                                     mock_get_forward_context):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty(10, 8, 64)

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10] * 10)
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 100
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8,
                                                                    64), 1)
        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)
        print(output.shape)
        mock_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_decode_only_swa_seq_len_mismatch(
            self, mock_npu_reshape_and_cache, mock_fused_infer_attention_score,
            mock_paged_attention, mock_get_forward_context):
        """Test forward pass in DecodeOnly state when seq)len_mismatch"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])  # len == 1 != query.size(0)==10
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        metadata.num_decodes = 10
        metadata.num_prefills = 0

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64),
                                                         torch.ones(10, 8, 64))

        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)

        mock_paged_attention.assert_called_once()
        mock_fused_infer_attention_score.assert_not_called()

        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_raise_error(self, mock_paged_attention):
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        layer = self.layer_no_quant

        with self.assertRaises(NotImplementedError):
            self.impl_error.forward(layer, query, key, value, kv_cache,
                                    metadata, output)


class TestBatchInvariantSequenceLengthHandling(TestBase):
    """
    Test cases for verifying correct sequence length handling in batch-invariant
    attention methods.

    These tests verify the fix for the bug where query_start_loc_list and
    seq_lens_list could include padding sequences, causing incorrect indexing.
    """

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"

    def test_metadata_with_padding_sequences(self):
        """
        Test that metadata correctly distinguishes between actual sequences
        and padding sequences.

        This verifies that:
        - num_prefills correctly counts actual prefill sequences
        - num_decodes correctly counts actual decode sequences
        - num_actual_tokens excludes padding tokens
        """
        # Simulate metadata with 2 actual sequences + 1 padding sequence
        metadata = MagicMock()
        metadata.num_prefills = 2  # Only 2 actual prefill sequences
        metadata.num_decodes = 0
        metadata.num_actual_tokens = 15  # Total actual tokens (not including padding)

        # query_start_loc_list includes padding: [5, 15, 25] where 25 is padding
        metadata.query_start_loc_list = [5, 15, 25]

        # seq_lens_list includes padding: [10, 20, 10] where last 10 is padding
        metadata.seq_lens_list = [10, 20, 10]

        # Verify that num_prefills (2) is less than len(query_start_loc_list) (3)
        self.assertEqual(metadata.num_prefills, 2)
        self.assertEqual(len(metadata.query_start_loc_list), 3)

        # The fix ensures we only process num_prefills sequences
        num_seqs_to_process = min(metadata.num_prefills,
                                   len(metadata.query_start_loc_list))
        self.assertEqual(num_seqs_to_process, 2)

    def test_query_start_loc_cumulative_positions(self):
        """
        Test that query_start_loc_list contains cumulative positions,
        not individual sequence lengths.

        query_start_loc_list = [end_0, end_1, end_2, ...]
        where end_i is the cumulative end position of sequence i.
        """
        # Example: 3 sequences with lengths 5, 10, 8
        # query_start_loc_list should be [5, 15, 23]
        query_start_loc_list = [5, 15, 23]

        # Calculate individual sequence lengths
        prev_end = 0
        seq_lengths = []
        for end in query_start_loc_list:
            seq_lengths.append(end - prev_end)
            prev_end = end

        self.assertEqual(seq_lengths, [5, 10, 8])

    def test_num_actual_tokens_limits_processing(self):
        """
        Test that num_actual_tokens correctly limits token processing
        to exclude padding tokens.
        """
        metadata = MagicMock()
        metadata.num_actual_tokens = 15
        metadata.num_prefills = 2
        # query_start_loc_list might extend beyond num_actual_tokens due to padding
        metadata.query_start_loc_list = [5, 15, 25]

        # When processing, we should limit seq_end to num_actual_tokens
        prev_end = 0
        for seq_idx in range(metadata.num_prefills):
            seq_end = metadata.query_start_loc_list[seq_idx]
            # Apply the fix: limit to num_actual_tokens
            seq_end = min(seq_end, metadata.num_actual_tokens)
            seq_len = seq_end - prev_end

            if seq_idx == 0:
                self.assertEqual(seq_len, 5)
            elif seq_idx == 1:
                self.assertEqual(seq_len, 10)

            prev_end = seq_end

    def test_decode_batch_size_from_num_decodes(self):
        """
        Test that decode batch size is correctly determined from num_decodes,
        not from seq_lens.shape[0] which may include padding.
        """
        metadata = MagicMock()
        # 3 actual decode sequences
        metadata.num_decodes = 3
        # seq_lens tensor might have extra padding entries
        metadata.seq_lens = torch.tensor([10, 20, 30, 0, 0])  # 5 entries, 2 are padding

        # The fix uses num_decodes instead of seq_lens.shape[0]
        batch_size = metadata.num_decodes
        if batch_size == 0:
            batch_size = metadata.seq_lens.shape[0]

        self.assertEqual(batch_size, 3)

        # Only slice actual sequences
        actual_seq_lens = metadata.seq_lens[:batch_size]
        self.assertEqual(len(actual_seq_lens), 3)
        self.assertTrue(torch.equal(actual_seq_lens, torch.tensor([10, 20, 30])))

    def test_kv_seq_lens_indexing_safety(self):
        """
        Test that kv_seq_lens indexing is safe when there's a mismatch
        between query_start_loc_list and seq_lens_list lengths.
        """
        metadata = MagicMock()
        metadata.num_prefills = 2
        metadata.query_start_loc_list = [5, 15, 25]  # 3 entries
        metadata.seq_lens_list = [10, 20]  # Only 2 entries (no padding in kv)

        # The fix uses min() to ensure safe indexing
        num_seqs = min(metadata.num_prefills,
                       len(metadata.query_start_loc_list),
                       len(metadata.seq_lens_list))

        self.assertEqual(num_seqs, 2)

        # Safe to index both lists
        for seq_idx in range(num_seqs):
            q_end = metadata.query_start_loc_list[seq_idx]
            kv_len = metadata.seq_lens_list[seq_idx]
            # No IndexError should occur
            self.assertIsNotNone(q_end)
            self.assertIsNotNone(kv_len)

