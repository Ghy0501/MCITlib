# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Compare complete runner command plans from two checkouts without executing them.

Each checkout is imported in a separate worker. Only that checkout's absolute
project root is normalized; argv, cwd, env, kind, task, sequence and metadata
are compared in order. Requires the dependencies needed to construct all
method plans, but never starts training, inference or GPU work.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def normalize(value: Any, project: str) -> Any:
    if isinstance(value, str):
        return value.replace(project, "__PROJECT__")
    if isinstance(value, list):
        return [normalize(item, project) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item, project) for key, item in value.items()}
    return value


def produce(project: Path, output: Path) -> None:
    project = project.resolve()
    sys.path.insert(0, str(project))
    from mcitlib.config.schema import PathSet
    from mcitlib.paths import PathResolver
    from mcitlib.registry import BENCHMARKS, METHODS
    from mcitlib.runner import build_sequence_commands

    paths = PathResolver(PathSet(project / "audit-models", project / "audit-data", project / "audit-ckpts", project,
                                 project / "audit-results", project / "logs/audit"))
    cases: list[tuple[str, str, str, dict[str, Any]]] = []
    for profile in ("original", "resolved_gpu"):
        flags = {} if profile == "original" else {"paths": paths, "gpus": "2,3", "master_port": 29873}
        for method in METHODS:
            for benchmark in BENCHMARKS:
                for sequence in (("train", "router") if method.id == "mrlora" else ("train",)):
                    for task in (None, *range(1, len(benchmark.tasks) + 1)):
                        for only_eval in (False, True):
                            opts = dict(flags, sequence=sequence, task_id=task, only_eval=only_eval)
                            case = f"{profile}/{method.id}/{benchmark.id}/{sequence}/{task}/eval={only_eval}"
                            cases.append((case, method.id, benchmark.id, opts))
            for task in (None, 1, 2, 3, 4, 5):
                opts = dict(flags, task_id=task, only_eval=True)
                if profile == "resolved_gpu":
                    opts.update(model_path="${ckpt_root}/audit", model_base="${model_root}/audit", num_chunks=1)
                cases.append((f"{profile}/{method.id}/general/{task}", method.id, "general", opts))
        for task in (None, 2):
            cases.append((f"{profile}/modalprompt/ucit/fix/{task}", "modalprompt", "ucit",
                          dict(flags, task_id=task, validation_fixes=True)))
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for case, method, benchmark, opts in cases:
            commands = build_sequence_commands(project / "configs", "llava", method, benchmark, **opts)
            payload = {"case": case, "commands": normalize([command.as_dict() for command in commands], str(project))}
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def summary(lines: list[str]) -> dict[str, Any]:
    records = []
    for line in lines:
        item = json.loads(line)
        records.append({"case": item["case"], "commands": len(item["commands"]),
                        "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()})
    return {"case_count": len(records), "command_count": sum(record["commands"] for record in records),
            "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()}


def run_worker(project: Path, output: Path) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-root", str(project), "--worker-output", str(output)]
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"plan generation failed for {project}:\n{result.stdout}{result.stderr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="checkout with the original expanded configs")
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root is not None:
        if args.worker_output is None:
            parser.error("worker output is required")
        produce(args.worker_root, args.worker_output)
        return 0
    if args.baseline is None:
        parser.error("--baseline is required")
    baseline, candidate = args.baseline.expanduser().resolve(), args.candidate.expanduser().resolve()
    for root in (baseline, candidate):
        if not (root / "configs/methods").is_dir() or not (root / "mcitlib/runner.py").is_file():
            parser.error(f"not an MCITlib checkout: {root}")
    try:
        with tempfile.TemporaryDirectory(prefix="mcitlib-command-comparison-") as tmp:
            old_file, new_file = Path(tmp) / "baseline.jsonl.gz", Path(tmp) / "candidate.jsonl.gz"
            print(f"Generating baseline plans from {baseline}", file=sys.stderr, flush=True)
            run_worker(baseline, old_file)
            print(f"Generating candidate plans from {candidate}", file=sys.stderr, flush=True)
            run_worker(candidate, new_file)
            with gzip.open(old_file, "rt", encoding="utf-8") as handle:
                old = [line.rstrip("\n") for line in handle]
            with gzip.open(new_file, "rt", encoding="utf-8") as handle:
                new = [line.rstrip("\n") for line in handle]
    except (OSError, RuntimeError) as error:
        print(f"command comparison: {error}", file=sys.stderr)
        return 2
    differences = []
    for index, (before, after) in enumerate(itertools.zip_longest(old, new)):
        if before != after:
            before_case = json.loads(before)["case"] if before is not None else None
            after_case = json.loads(after)["case"] if after is not None else None
            differences.append({"index": index, "baseline_case": before_case, "candidate_case": after_case})
    result = {"equivalent": not differences, "baseline": summary(old), "candidate": summary(new),
              "differing_cases": len(differences), "first_differences": differences[:10]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
