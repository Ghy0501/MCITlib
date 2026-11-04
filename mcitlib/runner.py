# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build and execute ordered continual-learning command sequences."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config.loader import deep_merge, load_layered_config
from .config.schema import Command, DataSource, RunConfig, TaskSpec
from .methods import create_method
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
            plugin = create_method(method.id, values)
            train = adapter.build_command(run_cfg, plugin, task)
            train.metadata["order"] = values["train_command"].get("order", -1)
            train.sequence = sequence
            commands.append(train)
        if only_eval or not method.separate_train_and_infer:
            recorded = _recorded_commands(values, task, include_extra=not only_eval, sequence=sequence)
            commands.extend(_apply_script_aliases(recorded, method))
    if paths is not None:
        for command in commands:
            command.argv = [paths.resolve_templates(str(value)) for value in command.argv]
            command.cwd = paths.resolve_templates(command.cwd)
            command.env = {key: paths.resolve_templates(str(value)) for key, value in command.env.items()}
    commands.sort(key=lambda command: (command.task_id or 0, command.metadata.get("order", -1)))
    return commands


def _step_identity(command: Command) -> str:
    payload = json.dumps(command.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_commands(
    commands: list[Command],
    run_dir: str | Path,
    repo_root: str | Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Execute commands sequentially with durable logs and resume markers."""
    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    markers = run_dir / ".mcitlib" / "steps"
    logs = run_dir / ".mcitlib" / "logs"
    results = []
    for index, command in enumerate(commands):
        identity = _step_identity(command)
        marker = markers / f"{identity}.json"
        if resume and marker.is_file():
            completed = json.loads(marker.read_text(encoding="utf-8"))
            if completed.get("status") == "completed":
                results.append({"index": index, "id": identity, "status": "skipped"})
                continue
        argv = list(command.argv)
        if argv and argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        if not argv:
            raise ValueError(f"empty command at sequence index {index}")
        cwd = Path(command.cwd)
        if not cwd.is_absolute():
            cwd = repo_root / cwd
        env = os.environ.copy()
        env.update(command.env)
        pythonpath = [str(repo_root)]
        if command.env.get("PYTHONPATH"):
            pythonpath.append(command.env["PYTHONPATH"])
        elif env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        log = logs / f"{index:05d}-{identity}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                sys.stdout.write(line)
            return_code = process.wait()
        finished = datetime.now(timezone.utc).isoformat()
        status = "completed" if return_code == 0 else "failed"
        _write_marker(marker, {
            "id": identity, "index": index, "status": status,
            "return_code": return_code, "command": command.as_dict(),
            "log": str(log), "started_at": started, "finished_at": finished,
        })
        results.append({"index": index, "id": identity, "status": status, "return_code": return_code})
        if return_code != 0:
            raise RuntimeError(f"command {index} failed with exit {return_code}; log: {log}")
    return results


def build_commands(run_cfg: RunConfig, tasks: list[TaskSpec], dry_run: bool = True) -> list[Command]:
    builder: Any = run_cfg.values.get("command_builder")
    if builder is None:
        raise NotImplementedError("use build_sequence_commands with the LLaVA adapter")
    return list(builder(run_cfg, tasks, dry_run=dry_run))


def run(run_cfg: RunConfig, tasks: list[TaskSpec], dry_run: bool = True,
        run_dir: str | Path | None = None, resume: bool = False) -> list[Command]:
    """Compatibility entry point for builders that return command objects."""
    commands = build_commands(run_cfg, tasks, dry_run=dry_run)
    if not dry_run:
        if run_dir is None:
            raise ValueError("run_dir is required for actual execution")
        execute_commands(commands, run_dir, run_cfg.values["repo_root"], resume=resume)
    return commands
