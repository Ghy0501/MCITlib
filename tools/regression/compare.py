# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Prepare/run isolated v3-v4 GPU regressions and strictly compare loss logs."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcitlib.config.loader import load_layered_config
from mcitlib.config.schema import DataSource
from mcitlib.registry import BENCHMARKS, METHODS, get_benchmark, get_method

LOSS = re.compile(r"(?:loss|Loss)[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")


@dataclass(frozen=True)
class ProcessOutcome:
    exit_code: int
    stopped_by_tool: bool = False

    @property
    def acceptable(self) -> bool:
        return self.exit_code == 0 or self.stopped_by_tool


def losses(path: Path) -> list[float]:
    return [float(match.group(1)) for match in LOSS.finditer(path.read_text(encoding="utf-8", errors="replace"))]


def compare_logs(
    v3_path: Path,
    v4_path: Path,
    rtol: float,
    expected_steps: int | None = None,
    v3_outcome: ProcessOutcome | None = None,
    v4_outcome: ProcessOutcome | None = None,
) -> dict[str, Any]:
    v3, v4 = losses(v3_path), losses(v4_path)
    count = min(len(v3), len(v4))
    if count == 0:
        raise ValueError(f"no loss values found: {v3_path}, {v4_path}")
    differences = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(v3[:count], v4[:count])]
    counts_match = len(v3) == len(v4)
    if expected_steps is not None:
        counts_match = counts_match and len(v3) == expected_steps and len(v4) == expected_steps
    exits_match = True
    if v3_outcome is not None or v4_outcome is not None:
        exits_match = bool(
            v3_outcome and v4_outcome and v3_outcome.acceptable and v4_outcome.acceptable
        )
    result = {
        "v3_losses": v3,
        "v4_losses": v4,
        "count": count,
        "v3_count": len(v3),
        "v4_count": len(v4),
        "expected_steps": expected_steps,
        "max_relative_difference": max(differences),
        "rtol": rtol,
        "counts_match": counts_match,
        "exits_acceptable": exits_match,
        "v3_exit": v3_outcome.exit_code if v3_outcome else None,
        "v4_exit": v4_outcome.exit_code if v4_outcome else None,
        "v3_stopped_by_tool": v3_outcome.stopped_by_tool if v3_outcome else False,
        "v4_stopped_by_tool": v4_outcome.stopped_by_tool if v4_outcome else False,
    }
    result["passed"] = bool(
        max(differences) <= rtol and counts_match and exits_match
    )
    return result


def _absolute_root(value: str, repo_root: Path) -> Path:
    path = Path(os.path.expanduser(value))
    return (path if path.is_absolute() else repo_root / path).resolve()


def substitute(
    argv: list[str],
    v3_root: Path,
    repo_root: Path,
    ckpt_root: Path | None = None,
) -> list[str]:
    model_root = _absolute_root(os.environ.get("MCITLIB_MODEL_ROOT", "./models"), repo_root)
    data_root = _absolute_root(os.environ.get("MCITLIB_DATA_ROOT", "./data"), repo_root)
    shared_ckpt = ckpt_root or _absolute_root(os.environ.get("MCITLIB_CKPT_ROOT", "./ckpts"), repo_root)
    roots = {
        "/your_model_path": str(model_root),
        "/your_data_path": str(data_root),
        "/your_ckpts_path": str(shared_ckpt),
        "/your_path/MCITlib_v3": str(v3_root),
        "__V3_ROOT__": str(v3_root),
        "${model_root}": str(model_root),
        "${data_root}": str(data_root),
        "${ckpt_root}": str(shared_ckpt),
        "${repo_root}": str(repo_root),
    }
    result = []
    for value in argv:
        for source, target in roots.items():
            if value == source:
                value = target
                break
            if value.startswith(source + "/"):
                value = target + value[len(source):]
                break
        result.append(value)
    return result


def run_process(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
    max_steps: int | None,
) -> ProcessOutcome:
    log.parent.mkdir(parents=True, exist_ok=True)
    stopped_by_tool = False
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        count = 0
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            count += len(LOSS.findall(line))
            if max_steps is not None and count >= max_steps:
                stopped_by_tool = True
                process.terminate()
                break
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            stopped_by_tool = False
    return ProcessOutcome(code, stopped_by_tool)


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _set_option(argv: list[str], name: str, value: Path) -> None:
    argv[argv.index(name) + 1] = str(value)


def _isolate_output(argv: list[str], original: list[str], output_root: Path) -> None:
    original_output = _option(original, "--output_dir")
    suffix = original_output
    for prefix in ("/your_ckpts_path", "${ckpt_root}"):
        if suffix == prefix:
            suffix = ""
            break
        if suffix.startswith(prefix + "/"):
            suffix = suffix[len(prefix) + 1:]
            break
    _set_option(argv, "--output_dir", output_root / suffix)


def _repair_training_argv(
    argv: list[str], repo_root: Path, method: Any, benchmark_id: str, task_id: int,
) -> list[str]:
    result = list(argv)
    if method.id == "replay":
        values = load_layered_config(repo_root / "configs", "llava", benchmark_id, method.id, task_id=task_id)
        source = DataSource.from_dict(values["data_config"])
        result[result.index("--data_path") + 1] = source.replay_path
        result[result.index("--image_folder") + 1] = source.router_folder
    return result


def _v4_training_argv(original: list[str], repo_root: Path, method: Any) -> list[str]:
    entry_index = next(i for i, value in enumerate(original) if value.endswith(".py"))
    result = original[:entry_index] + ["--module", "mcitlib.backbones.llava.train_entry"] + original[entry_index + 1:]
    deepspeed = repo_root / "configs/deepspeed" / f"{method.deepspeed_config}.json"
    _set_option(result, "--deepspeed", deepspeed)
    return result


def _prepare_shared_inputs(
    payload: dict[str, Any],
    task_id: int,
    v3_dir: Path,
    v3_root: Path,
    repo_root: Path,
    shared_root: Path,
    destination: Path,
    base_env: dict[str, str],
) -> None:
    marker = destination / f"prepared-task{task_id}.json"
    if marker.is_file():
        return
    current_train = next(
        command for command in payload["commands"]
        if command["kind"] == "train" and command["task_id"] == task_id
    )
    selected = []
    for command in payload["commands"]:
        kind = command["kind"]
        current_task = command["task_id"]
        if kind not in {"train", "extra"}:
            continue
        if current_task < task_id or (
            current_task == task_id and kind == "extra" and command["order"] < current_train["order"]
        ):
            selected.append(command)
    selected.sort(key=lambda command: (command["task_id"], command["order"]))
    method = get_method(payload["method"])
    for command in selected:
        raw_argv = command["argv"]
        if command["kind"] == "train":
            raw_argv = _repair_training_argv(raw_argv, repo_root, method, payload["benchmark"], command["task_id"])
        argv = substitute(raw_argv, v3_root, repo_root, shared_root)
        env = dict(base_env)
        env.update({key: value for key, value in command.get("env", {}).items() if key == "CUDA_VISIBLE_DEVICES"})
        log = destination / "prepare" / f"{command['task_id']}-{command['order']}-{command['kind']}.log"
        outcome = run_process(argv, v3_dir, env, log, None)
        if outcome.exit_code != 0:
            raise RuntimeError(f"shared preparation failed: {' '.join(argv)}; see {log}")
    marker.write_text(json.dumps({"task_id": task_id, "commands": len(selected)}, indent=2) + "\n", encoding="utf-8")


def run_combo(
    repo_root: Path,
    v3_root: Path,
    method_id: str,
    benchmark_id: str,
    task_id: int,
    out: Path,
    execute: bool,
    max_steps: int | None,
) -> dict[str, Any]:
    method, benchmark = get_method(method_id), get_benchmark(benchmark_id)
    repo_root = repo_root.resolve()
    v3_root = v3_root.resolve()
    out = out.resolve()
    fixture = repo_root / "tests/fixtures/v3_commands/llava" / method.id / f"{benchmark.id}.json"
    payload = json.loads(fixture.read_text())
    original = next(
        command["argv"] for command in payload["commands"]
        if command["kind"] == "train" and command["task_id"] == task_id
    )
    original = _repair_training_argv(original, repo_root, method, benchmark.id, task_id)
    destination = out / method.id / benchmark.id / f"task{task_id}"
    shared_root = out / "shared-checkpoints" / method.id / benchmark.id
    v3_output_root = destination / "v3-checkpoints"
    v4_output_root = destination / "v4-checkpoints"
    v3_input_root = shared_root if task_id > 1 else v3_output_root
    v4_input_root = shared_root if task_id > 1 else v4_output_root
    v3_argv = substitute(original, v3_root, repo_root, v3_input_root)
    v4_original = _v4_training_argv(original, repo_root, method)
    v4_argv = substitute(v4_original, repo_root, repo_root, v4_input_root)
    _isolate_output(v3_argv, original, v3_output_root)
    _isolate_output(v4_argv, original, v4_output_root)

    v3_dir = v3_root / "LLaVA" / method.v3_dir
    model_root = _absolute_root(os.environ.get("MCITLIB_MODEL_ROOT", "./models"), repo_root)
    data_root = _absolute_root(os.environ.get("MCITLIB_DATA_ROOT", "./data"), repo_root)
    v3_env = os.environ.copy()
    v3_env["PYTHONPATH"] = os.pathsep.join((str(v3_dir), str(v3_root / "LLaVA/LoRA-FT")))
    v3_env.update({"MCITLIB_MODEL_ROOT": str(model_root), "MCITLIB_DATA_ROOT": str(data_root), "MCITLIB_CKPT_ROOT": str(v3_input_root)})
    v4_env = os.environ.copy()
    v4_env["PYTHONPATH"] = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    v4_env["MCITLIB_TRAIN_ENTRY"] = method.train_entry
    v4_env["MCITLIB_METHOD"] = method.id
    v4_env.update({"MCITLIB_MODEL_ROOT": str(model_root), "MCITLIB_DATA_ROOT": str(data_root), "MCITLIB_CKPT_ROOT": str(v4_input_root)})
    if method.model_override:
        v4_env["MCITLIB_MODEL_OVERRIDE"] = method.model_override

    print(f"[{method.id}/{benchmark.id}/task{task_id}] v3: {' '.join(v3_argv)}")
    print(f"[{method.id}/{benchmark.id}/task{task_id}] v4: {' '.join(v4_argv)}")
    if not execute:
        return {
            "status": "dry-run", "method": method.id, "benchmark": benchmark.id, "task": task_id,
            "v3_output_dir": _option(v3_argv, "--output_dir"),
            "v4_output_dir": _option(v4_argv, "--output_dir"),
            "shared_input_root": str(shared_root) if task_id > 1 else None,
        }
    if task_id > 1:
        _prepare_shared_inputs(
            payload, task_id, v3_dir, v3_root, repo_root,
            shared_root, destination, v3_env,
        )
    v3_outcome = run_process(v3_argv, v3_dir, v3_env, destination / "v3.log", max_steps)
    v4_outcome = run_process(v4_argv, repo_root, v4_env, destination / "v4.log", max_steps)
    result = compare_logs(
        destination / "v3.log", destination / "v4.log", 1e-3 if max_steps else 0.005,
        expected_steps=max_steps, v3_outcome=v3_outcome, v4_outcome=v4_outcome,
    )
    result.update({"method": method.id, "benchmark": benchmark.id, "task": task_id})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--v3-log", type=Path, required=True)
    compare.add_argument("--v4-log", type=Path, required=True)
    compare.add_argument("--rtol", type=float, default=1e-3)
    run = sub.add_parser("run")
    run.add_argument("--combo", action="append", help="method:benchmark; may be repeated")
    run.add_argument("--tasks", default="1,2")
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--full", action="store_true")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--v3-root", type=Path, default=os.environ.get("MCITLIB_V3_ROOT", "../MCITlib_v3_ref"))
    run.add_argument("--out", type=Path, default=Path("regression_out"))
    args = parser.parse_args()
    if args.command == "compare":
        result = compare_logs(args.v3_log, args.v4_log, args.rtol)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    repo_root = Path(__file__).resolve().parents[2]
    combos = args.combo or [f"{method.id}:{benchmark.id}" for method in METHODS for benchmark in BENCHMARKS]
    results = []
    for combo in combos:
        method_id, benchmark_id = combo.split(":", 1)
        for task_text in args.tasks.split(","):
            results.append(run_combo(
                repo_root, args.v3_root.resolve(), method_id, benchmark_id,
                int(task_text), args.out, args.execute, None if args.full else args.max_steps,
            ))
    print(json.dumps(results, indent=2))
    return 0 if all(item.get("passed", True) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
