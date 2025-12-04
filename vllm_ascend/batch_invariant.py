# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.model_executor.layers.batch_invariant import (
    mm_batch_invariant, addmm_batch_invariant, matmul_batch_invariant,
    linear_batch_invariant, bmm_batch_invariant, _log_softmax_batch_invariant,
    softmax_batch_invariant, mean_batch_invariant)


def enable_batch_invariant_mode():

    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::_log_softmax",
                              _log_softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "NPU")

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
    os.environ["HCCL_DETERMINISTIC"] = "1"
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
