# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build ordered continual-learning command sequences without executing them."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config.loader import deep_merge, load_layered_config
from .config.schema import Command, DataSource, RunConfig, TaskSpec
from .registry import get_backbone, get_benchmark, get_method


def _task(benchmark: Any, task_id: int, name: str, values: dict[str, Any], router: bool = False) -> TaskSpec:
    source = values["source"]
    return TaskSpec(
        benchmark=benchmark.id,
        task_id=task_id,
        name=name,
        model_config=source["model_config"],
        data_config=source["data_config"],
        train_config=source["train_config"],
        eval_config=source.get("eval_config"),
        router=router,
        data=DataSource.from_dict(values["data_config"]),
    )


def _command_from_record(item: dict[str, Any], task_id: int, kind: str, sequence: str) -> Command:
    return Command(
        argv=list(item["argv"]),
        cwd=item.get("cwd", "."),
        env=dict(item.get("env", {})),
        kind=kind,
        task_id=task_id,
        sequence=sequence,
        metadata={"order": item.get("order", -1), "source": "v3-command-dump"},
    )


def _recorded_commands(values: dict[str, Any], task: TaskSpec, include_extra: bool = True, sequence: str = "train") -> list[Command]:
    result: list[Command] = []
    kinds = [("eval_commands", "eval")]
    if include_extra:
        kinds.extend((("extra_commands", "extra"), ("other_commands", "other")))
    for key, kind in kinds:
        for item in values.get(key, []):
            result.append(_command_from_record(item, task.task_id, kind, sequence))
    return result


def _apply_script_aliases(commands: list[Command], method: Any) -> list[Command]:
    aliases = dict(method.script_aliases)
    for command in commands:
        command.argv = [aliases.get(value, value) for value in command.argv]
    return commands


def build_sequence_commands(
    config_root: str | Path,
    backbone_id: str,
    method_id: str,
    benchmark_id: str,
    paths: Any = None,
    task_id: int | None = None,
    only_eval: bool = False,
    sequence: str = "train",
) -> list[Command]:
    """Build v3-ordered commands; NB-5 install commands are never emitted."""
    backbone = get_backbone(backbone_id)
    if not backbone.v4_supported:
        raise ValueError(f"backbone is not migrated yet: {backbone_id}")
    method = get_method(method_id)
    benchmark = get_benchmark(benchmark_id)
    if sequence not in {"train", "router"}:
        raise ValueError(f"unknown sequence: {sequence}")
    if sequence == "router" and method.id != "mrlora":
        raise ValueError(f"method {method.id} has no router sequence")
    if backbone_id != "llava":
        raise ValueError(f"unsupported backbone: {backbone_id}")
    from .backbones.llava.adapter import LlavaAdapter
    adapter = LlavaAdapter()
    commands: list[Command] = []
    if not only_eval and task_id in (None, 0, 1):
        base_values = load_layered_config(config_root, backbone_id, benchmark.id, method.id)
        benchmark_values = base_values.get("overrides", {}).get(backbone_id, {}).get(benchmark.id, {})
        commands.extend(_command_from_record(item, 0, item.get("kind", "extra"), sequence)
                        for item in benchmark_values.get("pre_commands", []))
        _apply_script_aliases(commands, method)
    for current_id, task_name in enumerate(benchmark.tasks, start=1):
        if task_id is not None and current_id != task_id:
            continue
        values = load_layered_config(config_root, backbone_id, benchmark.id, method.id, task_id=current_id)
        if sequence == "router":
            router_task = values["router"]["benchmarks"][benchmark.id]["tasks"][f"task_{current_id}"]
            values = deep_merge(values, router_task)
        values = {**values, "repo_root": str(Path(config_root).resolve().parent)}
        task = _task(benchmark, current_id, task_name, values, router=sequence == "router")
        if not only_eval:
            run_cfg = RunConfig(backbone_id, method.id, benchmark.id, task=task, values=values, paths=paths)
            train = adapter.build_command(run_cfg, method, task)
            train.metadata["order"] = values["train_command"].get("order", -1)
            train.sequence = sequence
            commands.append(train)
        if only_eval or not method.separate_train_and_infer:
            recorded = _recorded_commands(values, task, include_extra=not only_eval, sequence=sequence)
            commands.extend(_apply_script_aliases(recorded, method))
    commands.sort(key=lambda command: (command.task_id or 0, command.metadata.get("order", -1)))
    return commands


def build_commands(run_cfg: RunConfig, tasks: list[TaskSpec], dry_run: bool = True) -> list[Command]:
    builder: Any = run_cfg.values.get("command_builder")
    if builder is None:
        raise NotImplementedError("use build_sequence_commands with the LLaVA adapter")
    return list(builder(run_cfg, tasks, dry_run=dry_run))


def run(*args: Any, **kwargs: Any) -> list[Command]:
    if not kwargs.get("dry_run", True):
        raise NotImplementedError("actual training execution is installed in the GPU runner phase")
    return build_commands(*args, **kwargs)
