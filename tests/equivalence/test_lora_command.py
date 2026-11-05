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
from mcitlib.backbones.llava.args import LAUNCHER_ENTRY
from mcitlib.config.loader import load_layered_config
from mcitlib.config.schema import DataSource, RunConfig, TaskSpec
from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands


def normalise(value: Any, method_id: str | None = None) -> Any:
    if isinstance(value, list):
        return [normalise(item, method_id) for item in value]
    if isinstance(value, str):
        for filename in ("zero2.json", "zero3_offload.json"):
            if value == f"./scripts/{filename}" or value.endswith(f"/configs/deepspeed/{filename}"):
                return f"${{deepspeed_config}}/{filename}"
        merge_target = "mcitlib/methods/_keeplora/merge.py" if method_id == "keeplora" else "mcitlib/methods/_sefe/merge.py"
        script_aliases = {
            "key_elements_identification.py": "mcitlib/methods/_sefe/key_elements.py",
            "llava.train.extract_gradients": "mcitlib.methods._keeplora.extract_gradients",
            "llava.train.extract_weights": "mcitlib.methods._keeplora.extract_weights",
            "scripts/merge_lora_weights.py": merge_target,
            "scripts/UCIT_extract_router_result.py": "mcitlib/methods/_mrlora/router_results/ucit.py",
            "scripts/DCL_extract_router_result.py": "mcitlib/methods/_mrlora/router_results/dcl.py",
            "scripts/ACL_extract_router_result.py": "mcitlib/methods/_mrlora/router_results/acl.py",
        }
        if value in script_aliases:
            return script_aliases[value]
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


def without_merge_steps(commands: list[Any]) -> list[Any]:
    """The only v4-added step: explicit chunk merging (v3's shell ``cat``); verified in test_chunk_merge."""
    return [c for c in commands if not getattr(c, "metadata", {}).get("merge_chunks")]


def command_signature(command: Any, method_id: str) -> tuple[Any, ...]:
    argv = normalise(command.argv, method_id)
    metadata = getattr(command, "metadata", {})
    if metadata.get("eval_module_alias"):
        index = argv.index("-m")
        assert argv[index + 1] == "mcitlib.backbones.llava.eval_runners.entry"
        assert argv[index + 2] == metadata["eval_target"]
        argv = argv[:index + 1] + [metadata["source_eval_module"]] + argv[index + 3:]
    return command.kind, command.task_id, argv


@pytest.mark.parametrize("method_spec", METHODS, ids=lambda item: item.id)
@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_prototype_commands_match_v3(repo_root: Path, benchmark: Any, method_spec: Any) -> None:
    payload = json.loads((repo_root / "tests/fixtures/v3_commands/llava" / method_spec.id / f"{benchmark.id}.json").read_text())
    adapter = LlavaAdapter()
    if method_spec.id == "replay":
        from mcitlib.methods.replay import ReplayMethod
        method = ReplayMethod()
    else:
        method = SimpleNamespace(
            train_entry=method_spec.train_entry,
            model_override=method_spec.model_override,
        )

    def expected_training_argv(item):
        argv = list(item["argv"])
        entry_index = next(i for i, value in enumerate(argv) if value.endswith(".py"))
        argv = argv[:entry_index] + LAUNCHER_ENTRY + argv[entry_index + 1:]
        if method_spec.id == "replay":
            values = load_layered_config(
                repo_root / "configs", "llava", benchmark.id, method_spec.id, task_id=item["task_id"]
            )
            source = DataSource.from_dict(values["data_config"])
            argv[argv.index("--data_path") + 1] = source.replay_path
            argv[argv.index("--image_folder") + 1] = source.router_folder
        return argv

    for task_id, task_name in enumerate(benchmark.tasks, start=1):
        values = load_layered_config(repo_root / "configs", "llava", benchmark.id, method_spec.id, task_id=task_id)
        task = TaskSpec(
            benchmark=benchmark.id,
            task_id=task_id,
            name=task_name,
            model_config="model_configs/llava.json",
            data_config="",
            train_config="",
            data=DataSource.from_dict(values["data_config"]),
        )
        run_cfg = RunConfig("llava", method_spec.id, benchmark.id, task=task, values={**values, "repo_root": str(repo_root)})
        command = adapter.build_command(run_cfg, method, task)
        item = next(c for c in payload["commands"] if c["kind"] == "train" and c["task_id"] == task_id)
        expected = expected_training_argv(item)
        assert normalise(command.argv, method_spec.id) == normalise(expected, method_spec.id)
        deepspeed_path = Path(command.argv[command.argv.index("--deepspeed") + 1])
        assert deepspeed_path.is_absolute()
        assert deepspeed_path == repo_root / "configs/deepspeed" / f"{method_spec.deepspeed_config}.json"
        assert command.env["MCITLIB_TRAIN_ENTRY"] == method_spec.train_entry
        assert command.env["MCITLIB_METHOD"] == method_spec.id
        if method_spec.model_override:
            assert command.env["MCITLIB_MODEL_OVERRIDE"] == method_spec.model_override
        assert command.kind == "train"

    def signature(commands: list[Any]) -> list[tuple[Any, ...]]:
        return [command_signature(command, method_spec.id) for command in commands]

    expected_commands = []
    for item in payload["commands"]:
        if item["kind"] == "install":
            continue  # NB-5: installation is environment setup, not a task step.
        argv = list(item["argv"])
        if item["kind"] == "train":
            argv = expected_training_argv(item)
        expected_commands.append(SimpleNamespace(
            kind=item["kind"], task_id=item["task_id"], argv=argv,
            env={key: value for key, value in item.get("env", {}).items() if key == "CUDA_VISIBLE_DEVICES"},
        ))
    actual_commands = without_merge_steps(
        build_sequence_commands(repo_root / "configs", "llava", method_spec.id, benchmark.id)
    )
    expected_train = [
        c for c in expected_commands
        if not method_spec.separate_train_and_infer or c.kind != "eval"
    ]
    assert signature(actual_commands) == signature(expected_train)
    if method_spec.id == "keeplora":
        assert sum(c.task_id == 0 for c in actual_commands) == 1
        for actual, expected in zip(actual_commands, expected_train):
            if actual.kind != "train":
                assert actual.env.get("CUDA_VISIBLE_DEVICES") == expected.env.get("CUDA_VISIBLE_DEVICES")

    for task_id in range(1, len(benchmark.tasks) + 1):
        expected_eval = [c for c in expected_commands if c.kind == "eval" and c.task_id == task_id]
        actual_eval = without_merge_steps(build_sequence_commands(
            repo_root / "configs", "llava", method_spec.id, benchmark.id, task_id=task_id, only_eval=True
        ))
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
            argv = argv[:index] + LAUNCHER_ENTRY + argv[index + 1:]
        expected.append((item["kind"], item["task_id"], normalise(argv)))
    actual = without_merge_steps(build_sequence_commands(
        repo_root / "configs", "llava", "mrlora", benchmark.id, sequence="router"
    ))
    assert [command_signature(c, "mrlora") for c in actual] == expected
    assert all(c.sequence == "router" for c in actual)
    for task_id in range(1, len(benchmark.tasks) + 1):
        actual_eval = without_merge_steps(build_sequence_commands(
            repo_root / "configs", "llava", "mrlora", benchmark.id,
            task_id=task_id, only_eval=True, sequence="router",
        ))
        expected_eval = [item for item in expected if item[0] == "eval" and item[1] == task_id]
        assert [command_signature(c, "mrlora") for c in actual_eval] == expected_eval


@pytest.mark.parametrize("name,source", [
    ("zero2.json", "LLaVA/LoRA-FT/scripts/zero2.json"),
    ("zero3_offload.json", "LLaVA/ModalPrompt/scripts/zero3_offload.json"),
])
def test_deepspeed_configs_are_byte_identical(repo_root: Path, v3_root: Path, name: str, source: str) -> None:
    assert (repo_root / "configs/deepspeed" / name).read_bytes() == (v3_root / source).read_bytes()
