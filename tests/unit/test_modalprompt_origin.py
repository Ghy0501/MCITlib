# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Examine v3's detached snapshot with an actual HF resize and DS free_param."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

pytestmark = pytest.mark.env("llava-modalprompt-ds126")


def test_detached_snapshot_is_alias_before_and_survives_real_zero_free():
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus, free_param

    config = LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config)
    model.resize_token_embeddings(14)
    weight = model.get_input_embeddings().weight
    origin = weight.detach()  # Exact v3 expression.
    assert origin.data_ptr() == weight.data_ptr()
    expected = origin.clone()

    # Before DS partitions the parameter, the detached tensor is an alias.
    with torch.no_grad():
        weight[0, 0] += 1
    assert origin[0, 0] == weight[0, 0]
    with torch.no_grad():
        weight.copy_(expected)

    weight.ds_active_sub_modules = set()
    weight.ds_status = ZeroParamStatus.AVAILABLE
    free_param(weight)  # Actual installed DeepSpeed storage-release function.
    assert weight.numel() == 0
    assert torch.equal(origin, expected)
    weight.data = torch.full_like(expected, 99)
    assert torch.equal(origin, expected)
