# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T0: generated YAML preserves v3 JSON and shell command values."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcitlib.config.loader import load_layered_config, read_yaml
from mcitlib.registry import BENCHMARKS, METHODS

PREFIXES = (
    ("/your_model_path", "${model_root}"),
    ("/your_data_path", "${data_root}"),
    ("/your_ckpts_path", "${ckpt_root}"),
    ("/your_path/MCITlib_v3", "${repo_root}"),
    ("__V3_ROOT__", "${repo_root}"),
)

LEGACY_PREFIXES = (
    (r"^/home/[^/]+/[^/]+/clip-vit-large-patch14-336(?=/|$)", "${model_root}/clip-vit-large-patch14-336"),
    (r"^/home/[^/]+/[^/]+/MCITlib/checkpoints(?=/|$)", "${ckpt_root}"),
    (r"^/home/[^/]+/[^/]+/MCITlib(?=/|$)", "${repo_root}"),
    (r"^/mnt/cache/[^/]+/miniconda3/envs/coin/bin/python(?=/|$)", "${python_root}"),
)

LEGACY_METHODS = [item for item in METHODS if item.id not in {'dgg', 'octopus', 'pclr', 'same', 'sigma'}]


def normalise(value: Any) -> Any:
    if isinstance(value, str):
        asset_prefix = "/your_ckpts_path/MCITlib_v3/LLaVA/SMoLoRA/ins_emb_multi_"
        if value.startswith(asset_prefix):
            return "${repo_root}/assets/smolora/ins_emb_multi_" + value[len(asset_prefix):]
        for pattern, target in LEGACY_PREFIXES:
            match = re.match(pattern, value)
            if match:
                return target + value[match.end():]
        for source, target in PREFIXES:
            if value == source:
                return target
            if value.startswith(source + "/"):
                return target + value[len(source):]
        return value
    if isinstance(value, list):
        return [normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: normalise(item) for key, item in value.items()}
    return value


def source_paths(method: Any, benchmark: Any, task_name: str, task_id: int, router: bool = False) -> dict[str, str]:
    config_method = "RegLoRA" if method.id == "sefe" else method.v3_dir
    train_split = "train_router" if router else "train"
    eval_split = "eval_router" if router else "eval"
    return {
        "model_config": "model_configs/llava.json",
        "data_config": f"data_configs/{benchmark.v3_name}/{task_name}.json",
        "train_config": f"train_configs/{config_method}/LLaVA/{benchmark.v3_name}/{train_split}/task{task_id}.json",
        "eval_config": f"train_configs/{config_method}/LLaVA/{benchmark.v3_name}/{eval_split}/task{task_id}.json",
    }


def assert_task_matches(v3_root: Path, payload: dict[str, Any], actual: dict[str, Any], method: Any, benchmark: Any, task_name: str, task_id: int, router: bool = False) -> None:
    for key, relative in source_paths(method, benchmark, task_name, task_id, router).items():
        expected = normalise(json.loads((v3_root / "configs" / relative).read_text(encoding="utf-8")))
        if relative in payload.get("config_placeholders", {}):
            expected.update(payload["config_placeholders"][relative])
        assert actual[key] == expected, f"{method.id}/{benchmark.id}/task{task_id}: {key}"
        assert actual["source"][key] == relative

    commands = [c for c in payload["commands"] if c["task_id"] == task_id]
    for kind, actual_key in (
        ("train", "train_command"),
        ("eval", "eval_commands"),
        ("extra", "extra_commands"),
        ("other", "other_commands"),
    ):
        expected_commands = [normalise(c["argv"]) for c in commands if c["kind"] == kind]
        if kind == "train":
            assert len(expected_commands) == 1
            assert actual[actual_key]["argv"] == expected_commands[0]
        else:
            actual_commands = actual.get(actual_key, []) if kind == "other" else actual[actual_key]
            assert [item["argv"] for item in actual_commands] == expected_commands


@pytest.mark.parametrize("method", LEGACY_METHODS, ids=lambda item: item.id)
@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_t0_all_36_sequences(repo_root: Path, v3_root: Path, method: Any, benchmark: Any) -> None:
    fixture = repo_root / "tests/fixtures/v3_commands/llava" / method.id / f"{benchmark.id}.json"
    assert fixture.is_file(), f"missing legacy fixture: {fixture}"
    payload = json.loads(fixture.read_text())
    merged = load_layered_config(repo_root / "configs", "llava", benchmark.id, method.id)
    tasks = merged["overrides"]["llava"][benchmark.id]
    assert set(tasks) - {"pre_commands"} == {f"task_{i}" for i in range(1, len(benchmark.tasks) + 1)}
    pre_commands = tasks.get("pre_commands", [])
    expected_pre = [normalise(c["argv"]) for c in payload["commands"] if c["task_id"] == 0]
    assert [item["argv"] for item in pre_commands] == expected_pre
    for task_id, task_name in enumerate(benchmark.tasks, start=1):
        # Load again with the task layer enabled: this checks the actual
        # defaults < backbone < benchmark < method < task merge, not just raw YAML.
        actual = load_layered_config(repo_root / "configs", "llava", benchmark.id, method.id, task_id=task_id)
        assert actual["task_id"] == task_id
        assert_task_matches(v3_root, payload, actual, method, benchmark, task_name, task_id)


def test_t0_mr_lora_router_sequences(repo_root: Path, v3_root: Path) -> None:
    values = read_yaml(repo_root / "configs/methods/mrlora.yaml")
    for benchmark in BENCHMARKS:
        payload = json.loads((repo_root / "tests/fixtures/v3_commands/llava/mrlora" / f"{benchmark.id}_router.json").read_text())
        actual_benchmark = values["router"]["benchmarks"][benchmark.id]
        assert actual_benchmark["config_placeholders"] == payload["config_placeholders"]
        assert actual_benchmark["script_aliases"] == payload["script_aliases"]
        for task_id, task_name in enumerate(benchmark.tasks, start=1):
            assert_task_matches(v3_root, payload, actual_benchmark["tasks"][f"task_{task_id}"], values and next(m for m in METHODS if m.id == "mrlora"), benchmark, task_name, task_id, router=True)


def test_all_sequence_task_coverage(repo_root: Path) -> None:
    for payload_path in (repo_root / "tests/fixtures/v3_commands/llava").glob("*/*.json"):
        payload = json.loads(payload_path.read_text())
        benchmark = next(item for item in BENCHMARKS if item.id == payload["benchmark"])
        train_ids = [c["task_id"] for c in payload["commands"] if c["kind"] == "train"]
        assert train_ids == list(range(1, len(benchmark.tasks) + 1)), payload_path
        for command in payload["commands"]:
            assert "" not in command["argv"], payload_path
            assert "localhost:0,-1" not in command["argv"], payload_path


def test_generated_values_do_not_contain_machine_specific_paths(repo_root: Path) -> None:
    def check(value: Any) -> None:
        if isinstance(value, str):
            assert "/home/" not in value
            assert "/mnt/" not in value
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for item in value.values():
                check(item)

    for path in (repo_root / "configs/methods").glob("*.yaml"):
        check(yaml.safe_load(path.read_text()))
        check(read_yaml(path))
    for path in (repo_root / "configs/common").glob("*.yaml"):
        check(yaml.safe_load(path.read_text()))
