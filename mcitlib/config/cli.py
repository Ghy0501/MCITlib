# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Inspect and validate method configurations without loading model libraries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .loader import deep_merge, load_layered_config, read_yaml


def add_parser(sub: Any) -> None:
    parser = sub.add_parser("config", help="validate or inspect expanded method configuration")
    add_subcommands(parser)


def add_subcommands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="config_command", required=True)
    validate = commands.add_parser("validate", help="validate every task, router and command reference")
    validate.add_argument("--config-root", type=Path, default=None)
    validate.add_argument("--backbone", default="llava")
    validate.add_argument("--json", action="store_true")
    expand = commands.add_parser("expand", help="show one expanded task as JSON")
    expand.add_argument("--config-root", type=Path, default=None)
    expand.add_argument("--backbone", default="llava")
    expand.add_argument("--method", required=True)
    expand.add_argument("--benchmark", required=True)
    expand.add_argument("--task", type=int, required=True)
    expand.add_argument("--sequence", choices=("train", "router"), default="train")


def _root(path: Path | None) -> Path:
    project = Path(__file__).resolve().parents[2]
    candidate = (path or Path("configs")).expanduser()
    root = (candidate if candidate.is_absolute() else project / candidate).resolve()
    if not root.is_dir():
        raise ValueError(f"configuration directory does not exist: {root}")
    return root


def _check_task(task: dict[str, Any], name: str, task_id: int) -> int:
    if task.get("task_id") != task_id or task.get("task_name") != name:
        raise ValueError(f"invalid task identity: expected {task_id}/{name}")
    for field in ("source", "model_config", "data_config", "train_config", "eval_config"):
        if not isinstance(task.get(field), dict):
            raise ValueError(f"task {task_id}/{name} lacks a {field} mapping")
    count = 0
    for field in ("train_command", "eval_commands", "extra_commands", "other_commands"):
        if field not in task:
            continue
        values = [task[field]] if field == "train_command" else task[field]
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("argv"), list) or not value["argv"]:
                raise ValueError(f"task {task_id}/{name} has an invalid {field} record")
            if any(not isinstance(part, str) or not part or "@{" in part for part in value["argv"]):
                raise ValueError(f"task {task_id}/{name} has an invalid or unexpanded argument")
            if not isinstance(value.get("order"), int) or not isinstance(value.get("kind"), str):
                raise ValueError(f"task {task_id}/{name} has an invalid command order or kind")
            count += 1
    if "train_command" not in task:
        raise ValueError(f"task {task_id}/{name} lacks a training command")
    return count


def validate_config(root: str | Path, backbone: str = "llava") -> dict[str, int]:
    """Validate tasks, router stages and recorded command references."""
    from ..registry import BENCHMARKS, METHODS, benchmark_supported, compatibility, get_backbone

    root = _root(Path(root))
    backbone = get_backbone(backbone).id
    if backbone == "internvl":
        total_tasks = total_commands = 0
        benchmarks = [bench for bench in BENCHMARKS if bench.id in {"ucit", "mllm_dcl"}]
        compatible_methods = [method for method in METHODS if compatibility(method.id, backbone)[0]]
        for method in compatible_methods:
            for benchmark in benchmarks:
                for index, name in enumerate(benchmark.tasks, 1):
                    try:
                        values = load_layered_config(root, backbone, benchmark.id, method.id, task_id=index)
                        total_commands += _check_task(values, name, index)
                        total_tasks += 1
                        if method.id == "mrlora":
                            router = values["router"]["benchmarks"][benchmark.id]["tasks"][f"task_{index}"]
                            total_commands += _check_task(deep_merge(values, router), name, index)
                            total_tasks += 1
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"{backbone}/{method.id}/{benchmark.id}/task_{index}: {error}") from error
        return {"methods": len(compatible_methods), "benchmarks": len(benchmarks), "tasks": total_tasks,
                "recorded_commands": total_commands}
    if backbone != "llava":
        raise ValueError(f"backbone is not supported: {backbone}")
    total_tasks = total_commands = 0
    for method in METHODS:
        path = root / "methods" / f"{method.id}.yaml"
        if not path.is_file():
            raise ValueError(f"missing method configuration: {path}")
        values = read_yaml(path)
        if values.get("id") != method.id:
            raise ValueError(f"method ID mismatch in {path}")
        for benchmark in BENCHMARKS:
            if not benchmark_supported(method.id, benchmark.id):
                continue
            tasks = values.get("overrides", {}).get("llava", {}).get(benchmark.id, {})
            expected = {f"task_{index}" for index in range(1, len(benchmark.tasks) + 1)}
            if set(tasks) - {"pre_commands"} != expected:
                raise ValueError(f"wrong task coverage: {method.id}/{benchmark.id}")
            for index, name in enumerate(benchmark.tasks, 1):
                try:
                    total_commands += _check_task(tasks[f"task_{index}"], name, index)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{method.id}/{benchmark.id}/task_{index}: {error}") from error
                total_tasks += 1
            total_commands += len(tasks.get("pre_commands", []))
            if method.id == "mrlora":
                router = values["router"]["benchmarks"][benchmark.id]["tasks"]
                if set(router) != expected:
                    raise ValueError(f"wrong router task coverage: {benchmark.id}")
                for index, name in enumerate(benchmark.tasks, 1):
                    total_commands += _check_task(router[f"task_{index}"], name, index)
                    total_tasks += 1
    return {"methods": len(METHODS), "benchmarks": len(BENCHMARKS), "tasks": total_tasks, "recorded_commands": total_commands}


def expand_task(root: str | Path, method: str, benchmark: str, task: int, sequence: str = "train",
                backbone: str = "llava") -> dict[str, Any]:
    from ..registry import benchmark_supported, compatibility, get_backbone, get_benchmark, get_method

    root = _root(Path(root))
    method_spec, benchmark_spec, backbone_spec = get_method(method), get_benchmark(benchmark), get_backbone(backbone)
    supported, reason = compatibility(method_spec.id, backbone_spec.id)
    if not supported:
        raise ValueError(f"unsupported {backbone_spec.id}/{method_spec.id}: {reason}")
    if not benchmark_supported(method_spec.id, benchmark_spec.id):
        raise ValueError(f"{method_spec.id} is not configured for {benchmark_spec.id}")
    if benchmark_spec.id == "general":
        raise ValueError("general evaluation uses MLLM-DCL task configuration; select mllm_dcl")
    if not 1 <= task <= len(benchmark_spec.tasks):
        raise ValueError(f"task must be between 1 and {len(benchmark_spec.tasks)}")
    if sequence == "router" and method_spec.id != "mrlora":
        raise ValueError("router sequence is available only for mrlora")
    values = load_layered_config(root, backbone_spec.id, benchmark_spec.id, method_spec.id, task_id=task)
    if sequence == "router":
        router = values["router"]["benchmarks"][benchmark_spec.id]["tasks"][f"task_{task}"]
        values = deep_merge(values, router)
    return {key: value for key, value in values.items() if key not in {"overrides", "router"}}


def run(args: argparse.Namespace) -> int:
    try:
        root = _root(args.config_root)
        if args.config_command == "validate":
            payload = validate_config(root, args.backbone)
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else
                  "Validated {methods} methods, {benchmarks} benchmarks, {tasks} tasks and {recorded_commands} recorded commands.".format(**payload))
        else:
            print(json.dumps(expand_task(root, args.method, args.benchmark, args.task, args.sequence, args.backbone),
                             ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError) as error:
        print(f"mcitlib config: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mcitlib.config.cli", description=__doc__)
    add_subcommands(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
