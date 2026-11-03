# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T3: LoRA-FT v4 command generation preserves v3 argv."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcitlib.backbones.llava.adapter import LlavaAdapter
from mcitlib.backbones.llava.args import ENTRY_MODULE
from mcitlib.config.loader import load_layered_config
from mcitlib.config.schema import RunConfig, TaskSpec
from mcitlib.registry import BENCHMARKS


def normalise(value: Any) -> Any:
    if isinstance(value, list):
        return [normalise(item) for item in value]
    if isinstance(value, str):
        for source, target in (
            ("/your_model_path", "${model_root}"),
            ("/your_data_path", "${data_root}"),
            ("/your_ckpts_path", "${ckpt_root}"),
            ("/your_path/MCITlib_v3", "${repo_root}"),
            ("__V3_ROOT__", "${repo_root}"),
        ):
            if value == source:
                return target
            if value.startswith(source + "/"):
                return target + value[len(source):]
        return value
    return value


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_lora_ft_command_matches_v3(repo_root: Path, benchmark: Any) -> None:
    payload = json.loads((repo_root / "tests/fixtures/v3_commands/llava/lora_ft" / f"{benchmark.id}.json").read_text())
    adapter = LlavaAdapter()
    method = SimpleNamespace(train_entry="standard")
    for task_id, task_name in enumerate(benchmark.tasks, start=1):
        values = load_layered_config(repo_root / "configs", "llava", benchmark.id, "lora_ft", task_id=task_id)
        task = TaskSpec(
            benchmark=benchmark.id,
            task_id=task_id,
            name=task_name,
            model_config="model_configs/llava.json",
            data_config="",
            train_config="",
        )
        run_cfg = RunConfig("llava", "lora_ft", benchmark.id, task=task, values={**values, "repo_root": str(repo_root)})
        command = adapter.build_command(run_cfg, method, task)
        expected = next(c for c in payload["commands"] if c["kind"] == "train" and c["task_id"] == task_id)["argv"]
        entry_index = next(i for i, value in enumerate(expected) if value.endswith(".py"))
        expected = expected[:entry_index] + ["python", "-m", ENTRY_MODULE] + expected[entry_index + 1:]
        assert command.argv == normalise(expected)
        assert command.env["MCITLIB_TRAIN_ENTRY"] == "standard"
        assert command.kind == "train"


def test_moe_entry_is_selected_without_importing_torch(repo_root: Path) -> None:
    values = load_layered_config(repo_root / "configs", "llava", "ucit", "moelora", task_id=1)
    task = TaskSpec("ucit", 1, "ImageNet-R", "model_configs/llava.json", "", "")
    command = LlavaAdapter().build_command(
        RunConfig("llava", "moelora", "ucit", task=task, values={**values, "repo_root": str(repo_root)}),
        SimpleNamespace(train_entry="moe"),
        task,
    )
    assert command.env["MCITLIB_TRAIN_ENTRY"] == "moe"
    assert command.metadata["source_entry"] == "llava/train/train_mem_MOE.py"
