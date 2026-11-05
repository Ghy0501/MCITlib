# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""The ``mcitlib`` command-line interface (standard-library argparse)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .registry import BACKBONES, BENCHMARKS, METHODS, get_benchmark, matrix


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", default="llava")
    parser.add_argument("--method", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--config-root", type=Path, default=None, help="default: <checkout>/configs")
    parser.add_argument("--sequence", choices=("train", "router"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip commands with completed markers")
    parser.add_argument("--run-dir", type=Path, help="override checkpoint/log/marker directory")
    parser.add_argument("--gpus", help="comma-separated GPU ids to use (default: the first gpu_num, or MCITLIB_GPUS)")
    parser.add_argument("--master-port", type=int, help="deepspeed master port (default 9001, or MCITLIB_MASTER_PORT)")


# Folder names the v3 evaluation scripts use below <results>/ (their result_path values).
RESULT_DIRS = {"ucit": "UCIT", "mllm_dcl": "DCL", "mllm_acl": "ACL"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcitlib", description="MCITlib continual learning toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print the method/backbone/benchmark compatibility matrix")
    train = sub.add_parser("train", help="build or run a continual-learning sequence")
    _add_common(train)
    train.add_argument("--task", type=int)
    evaluate = sub.add_parser("eval", help="build or run evaluation")
    _add_common(evaluate)
    evaluate.add_argument("--after-task", type=int)
    evaluate.add_argument("--model-path", help="override the checkpoint for general evaluation")
    evaluate.add_argument("--model-base")
    evaluate.add_argument("--num-chunks", type=int)
    doctor = sub.add_parser("doctor", help="check environment, GPUs, imports, inputs and prerequisites before a run")
    doctor.add_argument("--method", required=True)
    doctor.add_argument("--benchmark", required=True)
    doctor.add_argument("--paths", type=Path)
    doctor.add_argument("--sequence", choices=("train", "router"), default="train")
    doctor.add_argument("--skip-gpu", action="store_true", help="do not require CUDA (CPU-only machines)")
    doctor.add_argument("--skip-entrypoints", action="store_true", help="do not start each entry point (faster)")
    doctor.add_argument("--json", action="store_true")
    metrics = sub.add_parser("metrics", help="compute continual-learning metrics")
    metrics.add_argument("--run-dir", type=Path, help="matrix JSON or CoIN results directory")
    metrics.add_argument("--method", help="with --benchmark and --paths: use <results_root>/<method>/<benchmark>")
    metrics.add_argument("--paths", type=Path)
    metrics.add_argument("--benchmark", choices=tuple(item.id for item in BENCHMARKS))
    metrics.add_argument("--output", type=Path, help="also write the complete result as JSON")
    return parser


def print_list() -> None:
    print("method\tbackbone\tbenchmark\tsupported\tenvironment\tnote")
    for row in matrix():
        print("\t".join(str(row[key]) for key in ("method", "backbone", "benchmark", "supported", "environment", "note")))
    print(f"\n{len(METHODS)} methods; {len(BACKBONES)} backbones; {len(BENCHMARKS)} benchmarks")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print_list()
        return 0
    if args.command == "doctor":
        from .doctor import main_doctor
        from .paths import bootstrap_sys_path

        try:
            bootstrap_sys_path()
        except RuntimeError as error:
            print(f"mcitlib: {error}", file=sys.stderr)
            return 2
        return main_doctor(args)
    if args.command == "metrics":
        from .eval.metrics import metrics_from_path

        try:
            tasks = get_benchmark(args.benchmark).tasks if args.benchmark else None
            if args.run_dir is None:
                if not (args.method and args.benchmark):
                    raise ValueError("pass --run-dir, or --method and --benchmark (with --paths)")
                from .paths import PathResolver
                from .paths import repo_root as checkout_root
                from .registry import get_method

                resolver = PathResolver.from_file(args.paths, repo_root=checkout_root())
                args.run_dir = resolver.method_results_dir(get_method(args.method).id) / RESULT_DIRS[args.benchmark]
            result = metrics_from_path(args.run_dir, tasks=tasks)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            print(f"mcitlib: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from .paths import PathResolver, bootstrap_sys_path
    from .paths import repo_root as checkout_root
    from .runner import build_sequence_commands, execute_commands

    try:
        if args.config_root is None:
            args.config_root = checkout_root() / "configs"
        bootstrap_sys_path(Path(args.config_root).resolve().parent)
    except RuntimeError as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 2
    repo_root = Path(args.config_root).resolve().parent
    paths = None
    if args.paths or not args.dry_run:
        paths = PathResolver.from_file(args.paths, repo_root=repo_root)
    try:
        commands = build_sequence_commands(
            args.config_root,
            args.backbone,
            args.method,
            args.benchmark,
            paths=paths,
            task_id=args.task if args.command == "train" else args.after_task,
            only_eval=args.command == "eval",
            sequence=args.sequence,
            model_path=getattr(args, "model_path", None),
            model_base=getattr(args, "model_base", None),
            num_chunks=getattr(args, "num_chunks", None),
            gpus=args.gpus or os.environ.get("MCITLIB_GPUS"),
            master_port=args.master_port or (int(os.environ["MCITLIB_MASTER_PORT"]) if os.environ.get("MCITLIB_MASTER_PORT") else None),
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps([command.as_dict() for command in commands], ensure_ascii=False, indent=2))
        return 0
    run_dir = args.run_dir or paths.run_dir(args.benchmark, args.method)
    try:
        results = execute_commands(commands, run_dir, repo_root, resume=args.resume)
    except RuntimeError as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"run_dir": str(run_dir), "steps": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
