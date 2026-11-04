# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T3: migrated LLaVA methods preserve v3 command sequences."""
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
from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands


def normalise(value: Any) -> Any:
    if isinstance(value, list):
        return [normalise(item) for item in value]
    if isinstance(value, str):
        asset_prefix = "/your_ckpts_path/MCITlib_v3/LLaVA/SMoLoRA/ins_emb_multi_"
        if value.startswith(asset_prefix):
            return "${repo_root}/assets/smolora/ins_emb_multi_" + value[len(asset_prefix):]
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


@pytest.mark.parametrize("method_spec", [m for m in METHODS if m.id in {"lora_ft", "replay", "moelora", "mrlora", "clmoe", "smolora", "olora", "hide"}], ids=lambda item: item.id)
@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_prototype_commands_match_v3(repo_root: Path, benchmark: Any, method_spec: Any) -> None:
    payload = json.loads((repo_root / "tests/fixtures/v3_commands/llava" / method_spec.id / f"{benchmark.id}.json").read_text())
    adapter = LlavaAdapter()
    method = SimpleNamespace(train_entry=method_spec.train_entry, model_override="hide" if method_spec.id == "hide" else None)
    for task_id, task_name in enumerate(benchmark.tasks, start=1):
        values = load_layered_config(repo_root / "configs", "llava", benchmark.id, method_spec.id, task_id=task_id)
        task = TaskSpec(
            benchmark=benchmark.id,
            task_id=task_id,
            name=task_name,
            model_config="model_configs/llava.json",
            data_config="",
            train_config="",
        )
        run_cfg = RunConfig("llava", method_spec.id, benchmark.id, task=task, values={**values, "repo_root": str(repo_root)})
        command = adapter.build_command(run_cfg, method, task)
        expected = next(c for c in payload["commands"] if c["kind"] == "train" and c["task_id"] == task_id)["argv"]
        entry_index = next(i for i, value in enumerate(expected) if value.endswith(".py"))
        expected = expected[:entry_index] + ["python", "-m", ENTRY_MODULE] + expected[entry_index + 1:]
        assert command.argv == normalise(expected)
        assert command.env["MCITLIB_TRAIN_ENTRY"] == method_spec.train_entry
        if method_spec.id == "hide":
            assert command.env["MCITLIB_MODEL_OVERRIDE"] == "hide"
        assert command.kind == "train"

    def signature(commands: list[Any]) -> list[tuple[Any, ...]]:
        return [(c.kind, c.task_id, normalise(c.argv)) for c in commands]

    expected_commands = []
    for item in payload["commands"]:
        if item["kind"] == "install":
            continue  # NB-5: installation is environment setup, not a task step.
        argv = list(item["argv"])
        if item["kind"] == "train":
            entry_index = next(i for i, value in enumerate(argv) if value.endswith(".py"))
            argv = argv[:entry_index] + ["python", "-m", ENTRY_MODULE] + argv[entry_index + 1:]
        expected_commands.append(SimpleNamespace(kind=item["kind"], task_id=item["task_id"], argv=argv))
    actual_commands = build_sequence_commands(repo_root / "configs", "llava", method_spec.id, benchmark.id)
    assert signature(actual_commands) == signature(expected_commands)

    for task_id in range(1, len(benchmark.tasks) + 1):
        expected_eval = [c for c in expected_commands if c.kind == "eval" and c.task_id == task_id]
        actual_eval = build_sequence_commands(
            repo_root / "configs", "llava", method_spec.id, benchmark.id, task_id=task_id, only_eval=True
        )
        assert signature(actual_eval) == signature(expected_eval)


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


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_mrlora_router_dry_run_matches_v3(repo_root: Path, benchmark: Any) -> None:
    fixture = repo_root / "tests/fixtures/v3_commands/llava/mrlora" / f"{benchmark.id}_router.json"
    payload = json.loads(fixture.read_text())
    expected = []
    for item in payload["commands"]:
        if item["kind"] == "install":
            continue
        argv = list(item["argv"])
        if item["kind"] == "train":
            index = next(i for i, value in enumerate(argv) if value.endswith(".py"))
            argv = argv[:index] + ["python", "-m", ENTRY_MODULE] + argv[index + 1:]
        expected.append((item["kind"], item["task_id"], normalise(argv)))
    actual = build_sequence_commands(
        repo_root / "configs", "llava", "mrlora", benchmark.id, sequence="router"
    )
    assert [(c.kind, c.task_id, c.argv) for c in actual] == expected
    assert all(c.sequence == "router" for c in actual)

    for task_id in range(1, len(benchmark.tasks) + 1):
        actual_eval = build_sequence_commands(
            repo_root / "configs", "llava", "mrlora", benchmark.id,
            task_id=task_id, only_eval=True, sequence="router",
        )
        expected_eval = [item for item in expected if item[0] == "eval" and item[1] == task_id]
        assert [(c.kind, c.task_id, c.argv) for c in actual_eval] == expected_eval
