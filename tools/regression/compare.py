# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Prepare/run v3-v4 GPU regressions and compare loss logs.

Examples:
    python tools/regression/compare.py compare --v3-log v3.log --v4-log v4.log
    python tools/regression/compare.py run --combo lora_ft:ucit --tasks 1,2 --execute
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from mcitlib.registry import BENCHMARKS, METHODS, get_benchmark, get_method

LOSS = re.compile(r"(?:loss|Loss)[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")


def losses(path: Path) -> list[float]:
    return [float(match.group(1)) for match in LOSS.finditer(path.read_text(encoding="utf-8", errors="replace"))]


def compare_logs(v3_path: Path, v4_path: Path, rtol: float) -> dict:
    v3, v4 = losses(v3_path), losses(v4_path)
    count = min(len(v3), len(v4))
    if count == 0:
        raise ValueError(f"no loss values found: {v3_path}, {v4_path}")
    differences = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(v3[:count], v4[:count])]
    result = {
        "v3_losses": v3[:count],
        "v4_losses": v4[:count],
        "count": count,
        "max_relative_difference": max(differences),
        "rtol": rtol,
        "passed": max(differences) <= rtol,
    }
    return result


def substitute(argv: list[str], v3_root: Path, repo_root: Path) -> list[str]:
    roots = {
        "/your_model_path": os.environ.get("MCITLIB_MODEL_ROOT", "./models"),
        "/your_data_path": os.environ.get("MCITLIB_DATA_ROOT", "./data"),
        "/your_ckpts_path": os.environ.get("MCITLIB_CKPT_ROOT", "./ckpts"),
        "/your_path/MCITlib_v3": str(v3_root),
        "__V3_ROOT__": str(v3_root),
        "${model_root}": os.environ.get("MCITLIB_MODEL_ROOT", "./models"),
        "${data_root}": os.environ.get("MCITLIB_DATA_ROOT", "./data"),
        "${ckpt_root}": os.environ.get("MCITLIB_CKPT_ROOT", "./ckpts"),
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


def run_process(argv: list[str], cwd: Path, env: dict[str, str], log: Path, max_steps: int | None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        count = 0
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            count += len(LOSS.findall(line))
            if max_steps is not None and count >= max_steps:
                process.terminate()
                break
        try:
            return process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def run_combo(repo_root: Path, v3_root: Path, method_id: str, benchmark_id: str, task_id: int, out: Path, execute: bool, max_steps: int | None) -> dict:
    method, benchmark = get_method(method_id), get_benchmark(benchmark_id)
    fixture = repo_root / "tests/fixtures/v3_commands/llava" / method.id / f"{benchmark.id}.json"
    payload = json.loads(fixture.read_text())
    original = next(c for c in payload["commands"] if c["kind"] == "train" and c["task_id"] == task_id)["argv"]
    v3_argv = substitute(original, v3_root, repo_root)
    entry_index = next(i for i, value in enumerate(original) if value.endswith(".py"))
    v4_original = original[:entry_index] + ["python", "-m", "mcitlib.backbones.llava.train_entry"] + original[entry_index + 1:]
    v4_argv = substitute(v4_original, repo_root, repo_root)
    v3_dir = v3_root / "LLaVA" / method.v3_dir
    v3_env = os.environ.copy()
    v3_env["PYTHONPATH"] = os.pathsep.join((str(v3_dir), str(v3_root / "LLaVA/LoRA-FT")))
    v4_env = os.environ.copy()
    v4_env["PYTHONPATH"] = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    v4_env["MCITLIB_TRAIN_ENTRY"] = method.train_entry
    if method.model_override:
        v4_env["MCITLIB_MODEL_OVERRIDE"] = method.model_override
    destination = out / method.id / benchmark.id / f"task{task_id}"
    print(f"[{method.id}/{benchmark.id}/task{task_id}] v3: {' '.join(v3_argv)}")
    print(f"[{method.id}/{benchmark.id}/task{task_id}] v4: {' '.join(v4_argv)}")
    if not execute:
        return {"status": "dry-run", "method": method.id, "benchmark": benchmark.id, "task": task_id}
    v3_code = run_process(v3_argv, v3_dir, v3_env, destination / "v3.log", max_steps)
    v4_code = run_process(v4_argv, repo_root, v4_env, destination / "v4.log", max_steps)
    result = compare_logs(destination / "v3.log", destination / "v4.log", 1e-3 if max_steps else 0.005)
    result.update({"method": method.id, "benchmark": benchmark.id, "task": task_id, "v3_exit": v3_code, "v4_exit": v4_code})
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
            results.append(run_combo(repo_root, args.v3_root.resolve(), method_id, benchmark_id, int(task_text), args.out, args.execute, None if args.full else args.max_steps))
    print(json.dumps(results, indent=2))
    return 0 if all(item.get("passed", True) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
