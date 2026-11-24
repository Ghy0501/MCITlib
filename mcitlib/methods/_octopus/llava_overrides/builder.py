# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Chronological FP32 merge and FP16 generation without a merged-model copy."""
from __future__ import annotations

from pathlib import Path

import torch

from ..state import load_manifest


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False,
                          device_map='auto', device='cuda', **kwargs):
    if model_base is None or load_8bit or load_4bit:
        raise ValueError('Octopus evaluation requires an unquantized base and an adapter checkpoint')
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
    from llava.model import LlavaLlamaForCausalLM
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer

    current = Path(model_path).resolve()
    history = load_manifest(current)
    config = AutoConfig.from_pretrained(current)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    # Upstream evaluate.py loads without a dtype argument (FP32 default),
    # applies every incremental adapter in discovery order, then vLLM uses fp16.
    # This port chooses deterministic chronological discovery.
    model = LlavaLlamaForCausalLM.from_pretrained(model_base, config=config, torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True).to(device)
    for checkpoint in [*history, current]:
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    model = model.to(dtype=torch.float16).eval()
    if getattr(model.config, 'mm_use_im_patch_token', False):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, 'mm_use_im_start_end', False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model()
    tower.to(device=device, dtype=torch.float16)
    return tokenizer, model, tower.image_processor, getattr(model.config, 'max_sequence_length', 2048)
