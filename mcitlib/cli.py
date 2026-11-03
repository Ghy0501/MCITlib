# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""The ``mcitlib`` command-line interface (standard-library argparse)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import BACKBONES, BENCHMARKS, METHODS, matrix


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", default="llava")
    parser.add_argument("--method", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--dry-run", action="store_true")


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
    metrics = sub.add_parser("metrics", help="compute continual-learning metrics")
    metrics.add_argument("--run-dir", type=Path, required=True)
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
    if args.command == "metrics":
        print(json.dumps({"run_dir": str(args.run_dir), "status": "metrics implementation lands in L5"}, ensure_ascii=False))
        return 0
    if not args.dry_run:
        print("actual execution is installed in the GPU runner phase; use --dry-run for a command preview")
        return 2
    from .runner import build_sequence_commands
    paths = None
    if args.paths:
        from .paths import PathResolver
        paths = PathResolver.from_file(args.paths, repo_root=Path.cwd())
    try:
        commands = build_sequence_commands(
            args.config_root,
            args.backbone,
            args.method,
            args.benchmark,
            paths=paths,
            task_id=args.task if args.command == "train" else args.after_task,
            only_eval=args.command == "eval",
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 2
    print(json.dumps([command.as_dict() for command in commands], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
