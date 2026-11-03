# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Tests for command parsing and dump metadata invariants."""
from __future__ import annotations

import json
from pathlib import Path

from tools.migrate_v3_configs import command_details


def test_deepspeed_command_parts() -> None:
    details = command_details([
        "deepspeed", "--include", "localhost:0,1", "--master_port", "9001",
        "llava/train/train_mem.py", "--lora_r", "16", "--mm_vision_select_layer", "-2",
    ])
    assert details["entry"] == "llava/train/train_mem.py"
    assert details["launcher"] == {"include": "localhost:0,1", "master_port": "9001"}
    assert details["args"] == {"lora_r": "16", "mm_vision_select_layer": "-2"}


def test_dump_metadata_and_sequence_repairs(repo_root: Path) -> None:
    root = repo_root / "tests/fixtures/v3_commands/llava"
    sefe = json.loads((root / "sefe/ucit.json").read_text())
    assert sefe["config_aliases"] == {"configs/train_configs/SEFE": "configs/train_configs/RegLoRA"}
    smolora = json.loads((root / "smolora/mllm_dcl.json").read_text())
    assert smolora["sequence_repairs"] == [1]
    modal = json.loads((root / "modalprompt/mllm_acl.json").read_text())
    assert modal["sequence_repairs"] == [1, 2, 3, 4]
    for path in root.glob("mrlora/*_router.json"):
        payload = json.loads(path.read_text())
        assert payload["config_placeholders"]
        assert payload["script_aliases"]
        assert payload["runner_exit_code"] == 0
