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
    parser.add_argument("--config-root", type=Path, default=None, help="default: <project>/configs")
    parser.add_argument("--sequence", choices=("train", "router"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip commands with completed markers")
    parser.add_argument("--run-dir", type=Path, help="run records directory under the project's logs_root")
    parser.add_argument("--gpus", help="comma-separated GPU ids to use (default: the first gpu_num, or MCITLIB_GPUS)")
    parser.add_argument("--master-port", type=int, help="deepspeed master port (default 9001, or MCITLIB_MASTER_PORT)")
    parser.add_argument("--execution-profile", choices=("four_gpu_validation",),
                        help="opt in to the four-GPU UCIT/MLLM-DCL validation plan")
    parser.add_argument("--validation-fixes", action="store_true", help="apply explicitly documented validation repairs")
    parser.add_argument("--expected-eval-samples", type=int, help="require this many annotations for each repaired DCL router cell")
    parser.add_argument("--expected-eval-hashes", type=json.loads, help=argparse.SUPPRESS)


# Folder names the v3 evaluation scripts use below <results>/ (their result_path values).
RESULT_DIRS = {"ucit": "UCIT", "mllm_dcl": "DCL", "mllm_acl": "ACL"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcitlib", description="MCITlib continual learning toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print the method/backbone/benchmark compatibility matrix")
    train = sub.add_parser("train", help="build or run a continual-learning sequence")
    _add_common(train)
    train.add_argument("--task", type=int)
    train.add_argument("--dgg-sample-profile", action="store_true",
                       help="use the verified 3k DGG sample budget and full replay policy")
    train.add_argument("--dgg-sample-manifest", type=Path,
                       help="override the reviewed 3k sample manifest")
    train.add_argument("--dgg-resume-checkpoint", type=Path,
                       help="resume one DGG task from a retained Trainer checkpoint")
    evaluate = sub.add_parser("eval", help="build or run evaluation")
    _add_common(evaluate)
    evaluate.add_argument("--after-task", type=int)
    evaluate.add_argument("--model-path", help="override the checkpoint for general evaluation")
    evaluate.add_argument("--model-base")
    evaluate.add_argument("--num-chunks", type=int)
    doctor = sub.add_parser("doctor", help="check environment, GPUs, imports, inputs and prerequisites before a run")
    doctor.add_argument("--backbone", default="llava")
    doctor.add_argument("--method", required=True)
    doctor.add_argument("--dgg-sample-profile", action="store_true")
    doctor.add_argument("--dgg-sample-manifest", type=Path)
    doctor.add_argument("--benchmark", required=True)
    doctor.add_argument("--paths", type=Path)
    doctor.add_argument("--sequence", choices=("train", "router"), default="train")
    doctor.add_argument("--gpus", help="comma-separated GPU ids for the plan being checked")
    doctor.add_argument("--master-port", type=int, help="deepspeed master port for the plan being checked")
    doctor.add_argument("--execution-profile", choices=("four_gpu_validation",),
                        help="check the four-GPU UCIT/MLLM-DCL validation plan")
    doctor.add_argument("--skip-gpu", action="store_true", help="do not require CUDA (CPU-only machines)")
    doctor.add_argument("--skip-entrypoints", action="store_true", help="do not start each entry point (faster)")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--validation-fixes", action="store_true", help=argparse.SUPPRESS)
    doctor.add_argument("--expected-eval-samples", type=int, help="require this many annotations for each repaired DCL router cell")
    doctor.add_argument("--expected-eval-hashes", type=json.loads, help=argparse.SUPPRESS)
    smoke = sub.add_parser("smoke", help="preview or run a strictly bounded LoRA-FT smoke test")
    smoke.add_argument("--backbone", choices=("llava", "internvl"), default="llava")
    smoke.add_argument("--benchmark", choices=("ucit", "mllm_dcl"), default="ucit")
    smoke.add_argument("--paths", type=Path)
    smoke.add_argument("--run-dir", type=Path, required=True, help="new directory under logs_root; parent must exist")
    smoke.add_argument("--adapter-dir", type=Path, help="existing parent for new Task1_llava_lora weights (may be external)")
    smoke.add_argument("--samples", type=int, default=4, help="first training records, 1..8 (default 4)")
    smoke.add_argument("--max-steps", type=int, default=1, help="training steps, 1..3 (default 1)")
    smoke.add_argument("--gpus", help="comma-separated GPU ids for the shared two-GPU task")
    smoke.add_argument("--master-port", type=int)
    smoke.add_argument("--execute", action="store_true", help="run full doctor, then execute only on success")
    metrics = sub.add_parser("metrics", help="compute continual-learning metrics")
    metrics.add_argument("--run-dir", type=Path, help="matrix JSON or CoIN results directory")
    metrics.add_argument("--backbone", default="llava")
    metrics.add_argument("--method", help="with --benchmark and --paths: use <results_root>/<method>/<benchmark>")
    metrics.add_argument("--paths", type=Path)
    metrics.add_argument("--benchmark", choices=tuple(item.id for item in BENCHMARKS))
    metrics.add_argument("--output", type=Path, help="also write the complete result as JSON")
    report = sub.add_parser("report", help="export a read-only validation queue snapshot")
    report.add_argument("--run-dir", type=Path, required=True, help="validation queue directory containing state.json")
    report.add_argument("--output-dir", type=Path, required=True, help="separate directory for report artifacts")
    from .config.cli import add_parser as add_config_parser

    add_config_parser(sub)
    return parser


def print_list() -> None:
    print("method\tbackbone\tbenchmark\tsupported\tenvironment\tnote")
    for row in matrix():
        print("\t".join(str(row[key]) for key in ("method", "backbone", "benchmark", "supported", "environment", "note")))
    print(f"\n{len(METHODS)} methods; {len(BACKBONES)} backbones; {len(BENCHMARKS)} benchmarks")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        from .config.cli import run as run_config

        return run_config(args)
    if args.command == "report":
        from .reporting import generate_report

        try:
            outputs = generate_report(args.run_dir, args.output_dir)
        except (ValueError, OSError) as error:
            print(f"mcitlib: {error}", file=sys.stderr)
            return 2
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return 0
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
    if args.command == "smoke":
        from .smoke import build_plan, execute_plan

        try:
            plan = build_plan(args.paths, args.run_dir, args.samples, args.max_steps,
                              args.gpus or os.environ.get("MCITLIB_GPUS"),
                              args.master_port or (int(os.environ["MCITLIB_MASTER_PORT"])
                                                   if os.environ.get("MCITLIB_MASTER_PORT") else None),
                              benchmark=args.benchmark, adapter_dir=args.adapter_dir, backbone=args.backbone)
            if args.execute:
                plan = execute_plan(plan, args.paths)
        except (KeyError, ValueError, FileNotFoundError, OSError, RuntimeError) as error:
            print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0 if plan["status"] in {"preview", "completed"} else 1
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
                args.run_dir = resolver.method_results_dir(get_method(args.method).id, args.backbone) / RESULT_DIRS[args.benchmark]
            result = metrics_from_path(args.run_dir, tasks=tasks)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            print(f"mcitlib: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from .paths import PathResolver, bootstrap_sys_path, managed_path
    from .paths import repo_root as checkout_root
    from .runner import build_sequence_commands, execute_commands

    try:
        repo_root = checkout_root()
        args.config_root = managed_path(args.config_root or "configs", repo_root, label="config_root")
        if args.config_root.parent != repo_root:
            raise ValueError("config_root must be a direct child of the project root")
        bootstrap_sys_path(repo_root)
        paths = PathResolver.from_file(args.paths, repo_root=repo_root) if (
            args.paths or not args.dry_run or args.run_dir
        ) else None
        run_dir = (paths.managed_run_dir(args.run_dir) if args.run_dir else
                   paths.run_dir(args.benchmark, args.method, backbone=args.backbone)) if paths else None
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 2
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
            execution_profile=args.execution_profile,
            validation_fixes=args.validation_fixes,
            expected_eval_samples=args.expected_eval_samples,
            expected_eval_hashes=args.expected_eval_hashes,
        )
        sample = getattr(args, "dgg_sample_profile", False)
        sample_manifest = getattr(args, "dgg_sample_manifest", None)
        if sample_manifest is not None and not sample:
            raise ValueError("--dgg-sample-manifest requires --dgg-sample-profile")
        if sample:
            from .registry import get_method
            if get_method(args.method).id != "dgg":
                raise ValueError("--dgg-sample-profile requires DGG")
            from .methods._dgg.sample import apply_dgg_sample_profile
            apply_dgg_sample_profile(commands, args.benchmark, repo_root, sample_manifest)
        checkpoint = getattr(args, "dgg_resume_checkpoint", None)
        if checkpoint is not None:
            from .registry import get_method
            training = [item for item in commands if item.kind == "train"]
            if get_method(args.method).id != "dgg" or args.task is None or len(training) != 1:
                raise ValueError("--dgg-resume-checkpoint requires one selected DGG task")
            checkpoint = checkpoint.expanduser().resolve()
            output = Path(training[0].argv[training[0].argv.index("--output_dir") + 1]).resolve()
            if not checkpoint.is_dir() or checkpoint.parent != output or not checkpoint.name.startswith("checkpoint-"):
                raise ValueError("DGG resume checkpoint must be a checkpoint directory under that task output")
            training[0].argv.extend(["--dgg_resume_from_checkpoint", str(checkpoint)])
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps([command.as_dict() for command in commands], ensure_ascii=False, indent=2))
        return 0
    try:
        results = execute_commands(commands, run_dir, repo_root, resume=args.resume, logs_root=paths.logs_root)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"run_dir": str(run_dir), "steps": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
