# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""NB-6 reference config injection and adapter forwarding."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import transformers

from mcitlib.backbones.llava.adapter import LlavaAdapter
from mcitlib.backbones.llava.model_config import inject_text_tower_config, normalize_generation_config


def test_text_tower_injection_matches_reference(repo_root: Path) -> None:
    reference = json.loads((repo_root / "examples/llava-v1.5-7b/config.json").read_text())
    config = SimpleNamespace(mm_vision_tower="/models/clip")
    result = inject_text_tower_config(config, None)
    assert result.mm_text_tower == "/models/clip"
    assert result.mm_text_select_layer == reference["mm_text_select_layer"] == -1
    assert vars(result)["mm_vision_tower"] == "/models/clip"


def test_generation_config_matches_reference_without_sampling_defaults(repo_root: Path) -> None:
    reference = json.loads((repo_root / "examples/llava-v1.5-7b/generation_config.json").read_text())
    model = SimpleNamespace(generation_config=transformers.GenerationConfig(
        bos_token_id=1, eos_token_id=2, pad_token_id=0, max_length=4096,
        temperature=0.7, top_p=0.9,
    ))
    values = normalize_generation_config(model).generation_config.to_dict()
    assert "temperature" not in values and "top_p" not in values
    for key, expected in reference.items():
        if key != "transformers_version":
            assert values[key] == expected


def test_adapter_forwards_resolved_text_tower_to_private_builder() -> None:
    expected = (Mock(), Mock(), Mock(), 2048)
    with patch("mcitlib.methods._hide.llava_overrides.builder.load_pretrained_model", return_value=expected) as load:
        result = LlavaAdapter().load_base({
            "model_name": "llava", "vision_tower": "/models/clip", "device": "cpu",
        }, "hide")
    assert result == expected
    assert load.call_args.kwargs["text_tower"] == "/models/clip"
