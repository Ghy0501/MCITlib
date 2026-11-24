# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Plan or run a separate UCIT evaluation using verified existing checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...config.schema import Command, PathSet
from ...dataflow import find_gaps
from ...paths import PathResolver, managed_path, repo_root
from ...runner import build_sequence_commands, execute_commands
from ...validation_batch import verified_ucit_sample_info
from .router_results.ucit_validation import EXPERTS, _identifier, _index, _read_jsonl


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _option(command: dict[str, Any] | Command, flag: str) -> str | None:
    argv = command.argv if isinstance(command, Command) else command.get("argv", [])
    return argv[argv.index(flag) + 1] if flag in argv else None


def _checkpoint(path: Path) -> dict[str, Any]:
    files = {}
    for name in ("config.json", "adapter_config.json", "non_lora_trainables.bin", "adapter_model.bin", "adapter_model.safetensors"):
        item = path / name
        if item.is_file():
            stat = item.stat()
            files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "sha256": _hash(item) if name.endswith(".json") else None}
    if (not files.get("config.json") or not files.get("adapter_config.json") or
            not files.get("non_lora_trainables.bin") or
            not (files.get("adapter_model.bin") or files.get("adapter_model.safetensors")) or
            any(item["size"] <= 0 for item in files.values())):
        raise ValueError(f"checkpoint is incomplete: {path}")
    return {"path": str(path), "files": files, "note": "Weights are statted, not hashed or loaded."}


def _terminal_events(path: Path) -> dict[int, dict[str, Any]]:
    return {item["index"]: item for item in _read_jsonl(path) if item.get("status") in {"completed", "failed"}}


def _successful(index: int, command: dict[str, Any], events: dict[int, dict[str, Any]]) -> bool:
    event = events.get(index, {})
    return event.get("status") == "completed" and event.get("return_code") == 0 and event.get("command") == command


def _validate_reused_cell(folder: Path, domain: str, stage: int, annotation: Path, count: int,
                          commands: list[dict[str, Any]], events: dict[int, dict[str, Any]],
                          checkpoint: Path, model_base: Path) -> dict[str, Any]:
    merge = folder / "merge.jsonl"
    score = folder / "Result.text"
    if not merge.is_file() or not score.is_file():
        raise ValueError(f"incomplete existing expert output: {folder}")
    annotations = json.loads(annotation.read_text(encoding="utf-8"))
    predictions = _read_jsonl(merge)
    _index(annotations, annotation)
    _index(predictions, merge)
    annotation_ids = [_identifier(row, annotation) for row in annotations]
    prediction_ids = [_identifier(row, merge) for row in predictions]
    if (len(annotations) != count or len(predictions) != count or
            any(type(a) is not type(b) or a != b for a, b in zip(annotation_ids, prediction_ids))):
        raise ValueError(f"expert output IDs or order differ from verified annotations: {folder}")
    if any(row.get("model_id") != checkpoint.name for row in predictions):
        raise ValueError(f"expert model identifiers differ from checkpoint: {folder}")
    indices = [i for i, cmd in enumerate(commands) if cmd.get("sequence") == "router" and
               cmd.get("task_id") == stage and _option(cmd, "--answers-file") and
               Path(_option(cmd, "--answers-file")).parent == folder]
    if len(indices) != 4 or not all(_successful(i, commands[i], events) for i in indices):
        raise ValueError(f"missing successful recorded inference for {folder}")
    for i in indices:
        cmd = commands[i]
        if (Path(_option(cmd, "--model-path") or "").resolve() != checkpoint or
                Path(_option(cmd, "--model-base") or "").resolve() != model_base or
                Path(_option(cmd, "--question-file") or "").resolve() != annotation.resolve() or
                _option(cmd, "--temperature") != "0" or _option(cmd, "--conv-mode") != "vicuna_v1" or
                _option(cmd, "--num-chunks") != "4"):
            raise ValueError(f"recorded inference inputs differ from expected expert evaluation: {folder}")
    chunks = [folder / f"4_{index}.jsonl" for index in range(4)]
    if not all(path.is_file() for path in chunks):
        raise ValueError(f"missing four source chunks: {folder}")
    if hashlib.sha256(b"".join(path.read_bytes() for path in chunks)).hexdigest() != _hash(merge):
        raise ValueError(f"merged predictions differ from source chunks: {folder}")
    match = re.search(r"(?m)^Samples:\s*(\d+)\s*$", score.read_text(encoding="utf-8"))
    if not match or int(match.group(1)) != count:
        raise ValueError(f"expert score denominator differs from verified sample: {folder}")
    return {"domain": domain, "expert": stage, "samples": count, "merge": str(merge),
            "merge_sha256": _hash(merge), "score": str(score), "score_sha256": _hash(score),
            "completed_inference_indices": indices, "annotation": str(annotation), "annotation_sha256": _hash(annotation)}


def build_plan(old_run_dir: str | Path, results_root: str | Path, logs_dir: str | Path,
               *, config_root: str | Path | None = None) -> dict[str, Any]:
    """Make a read-only, reviewable plan. No output directory is created."""
    old_run = Path(old_run_dir).expanduser().resolve()
    root = repo_root()
    config_root = Path(config_root).resolve() if config_root else root / "configs"
    results = Path(results_root).expanduser()
    logs = Path(logs_dir).expanduser()
    if not results.is_absolute() or not logs.is_absolute():
        raise ValueError("results_root and logs_dir must be absolute")
    results, logs = results.resolve(), logs.resolve()
    managed_path(results, root, within=root / "reports", label="evaluation results")
    managed_path(logs, root, within=root / "logs", label="evaluation logs")
    state_path = old_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = state["jobs"]["mrlora/ucit"]
    if job.get("status") != "completed":
        raise ValueError("source MR-LoRA UCIT job is not completed")
    python = Path(job["python"])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"source job Python interpreter is unavailable: {python}")
    project = Path(state["project"]).resolve()
    job_dir = old_run / "jobs/mrlora/ucit"
    old_paths = PathResolver.from_file(job_dir / "paths.effective.yaml", repo_root=project)
    old_results = old_paths.results_root.resolve()
    old_ckpts = old_paths.paths.ckpt_root.resolve()
    for protected in (old_run, old_results, old_ckpts):
        if results == protected or protected in results.parents or results in protected.parents:
            raise ValueError(f"new results must not overlap source artifacts: {protected}")
    if results.exists() or logs.exists():
        raise ValueError("results_root and logs_dir must be new paths")
    sample = verified_ucit_sample_info(project, old_paths)
    if sample is None:
        raise ValueError("source UCIT data is not bound to a verified sample manifest")
    commands_path = job_dir / "commands.json"
    events_path = job_dir / "events.jsonl"
    commands = json.loads(commands_path.read_text(encoding="utf-8"))["commands"]
    events = _terminal_events(events_path)
    checkpoints = {}
    bases = set()
    for sequence in ("train", "router"):
        for stage in range(1, 7):
            entries = [(i, cmd) for i, cmd in enumerate(commands) if cmd.get("sequence") == sequence and
                       cmd.get("kind") == "train" and cmd.get("task_id") == stage]
            if len(entries) != 1 or not _successful(entries[0][0], entries[0][1], events):
                raise ValueError(f"missing successful {sequence} checkpoint training record for stage {stage}")
            target = Path(_option(entries[0][1], "--output_dir") or "").resolve()
            expected_name = (f"Task{stage}_llava_lora" if sequence == "train" else f"Router_task{stage}_llava_lora")
            if target.name != expected_name or old_ckpts not in target.parents:
                raise ValueError(f"unexpected checkpoint path: {target}")
            model_base = _option(entries[0][1], "--model_name_or_path")
            if not model_base or not Path(model_base).is_absolute():
                raise ValueError(f"missing recorded base model for {sequence} stage {stage}")
            bases.add(str(Path(model_base).resolve()))
            checkpoints[f"{sequence}/{stage}"] = _checkpoint(target)
    if len(bases) != 1:
        raise ValueError("expert and router checkpoints do not share one recorded base model")
    model_base = Path(bases.pop())
    code_files = ("mcitlib/backbones/llava/eval_runners/coin/model_others.py",
                  "mcitlib/backbones/llava/eval_runners/coin/coin_utils.py",
                  "mcitlib/backbones/llava/eval_runners/entry.py", "mcitlib/backbones/llava/overrides.py",
                  "mcitlib/tuners/_peft_registry.py", "mcitlib/tuners/mcit_expert_lora.py",
                  "third_party/LLaVA/llava/model/builder.py", "third_party/LLaVA/llava/mm_utils.py",
                  "third_party/LLaVA/llava/conversation.py",
                  "third_party/LLaVA/llava/model/language_model/llava_llama.py",
                  "third_party/peft/peft/peft_model.py",
                  "mcitlib/methods/_mrlora/eval_runners/eval_deepseek_r1.py",
                  "mcitlib/methods/_mrlora/eval_runners/eval_caption.py")
    for name in code_files:
        if _hash(project / name) != _hash(root / name):
            raise ValueError(f"source inference/scorer code differs from the reviewed checkout: {name}")
    datasets = {}
    for domain in EXPERTS:
        item = project / "data/validation_3k_1k/ucit" / domain / "test.json"
        if _hash(item) != sample["hashes"][domain]:
            raise ValueError(f"source annotations differ from sample manifest: {domain}")
        datasets[domain] = item.resolve()
    old_root = old_results / "mrlora/UCIT/each_dataset"
    reuse = []
    for stage in range(1, 7):
        for domain in EXPERTS:
            folder = old_root / domain / f"MR-LoRA-task{stage}"
            if not folder.exists():
                continue
            reuse.append(_validate_reused_cell(folder, domain, stage, datasets[domain], sample["count"],
                                               commands, events, Path(checkpoints[f"train/{stage}"]["path"]),
                                               model_base))
    new_paths = PathResolver(PathSet(old_paths.paths.model_root, old_paths.paths.data_root, old_ckpts,
                                    project, results, old_paths.paths.logs_root), old_paths.mappings)
    common = {"paths": new_paths, "validation_fixes": True, "execution_profile": "four_gpu_validation",
              "gpus": "0,1,2,3", "master_port": state.get("master_port"),
              "expected_eval_samples": sample["count"], "expected_eval_hashes": sample["hashes"]}
    ordinary = build_sequence_commands(config_root, "llava", "mrlora", "ucit", only_eval=True, **common)
    router = build_sequence_commands(config_root, "llava", "mrlora", "ucit", only_eval=True, sequence="router", **common)
    reused = {(item["expert"], item["domain"]) for item in reuse}
    filtered = []
    for command in ordinary:
        location = _option(command, "--answers-file") or (_option(command, "--output") if command.metadata.get("merge_chunks") else None) or _option(command, "--output-dir")
        if location:
            directory = Path(location).parent if Path(location).name.endswith(".jsonl") else Path(location)
            domain = directory.parent.name
            if (command.task_id, domain) in reused:
                continue
        filtered.append(command)
    for command in [*filtered, *router]:
        base = _option(command, "--model-base")
        if base and Path(base).resolve() != model_base:
            raise ValueError(f"planned inference base model differs from recorded training: {base}")
        if command.argv[0] == "python":
            command.argv[0] = str(python)
    copies = []
    for item in reuse:
        target = results / "mrlora/UCIT/each_dataset" / item["domain"] / f"MR-LoRA-task{item['expert']}"
        for source_key, name, hash_key in (("merge", "merge.jsonl", "merge_sha256"), ("score", "Result.text", "score_sha256")):
            copies.append(Command(argv=[str(python), "-m", "mcitlib.methods._mrlora.ucit_eval", "_copy",
                                        "--source", item[source_key], "--output", str(target / name),
                                        "--sha256", item[hash_key]], kind="extra", task_id=item["expert"], sequence="train",
                                  metadata={"validation_fix": "mrlora_ucit_reuse_verified_expert"}))
    combined = [*copies, *filtered, *router]
    available = tuple(item["path"] for item in checkpoints.values())
    roots = new_paths.external_roots("ucit")
    gaps = find_gaps(combined, roots, available, (str(old_ckpts), str(results)))
    if gaps:
        raise ValueError(f"evaluation plan has unproduced inputs: {gaps[0]}")
    combined.append(Command(argv=[str(python), "-m", "mcitlib.methods._mrlora.ucit_eval", "_finish",
                                  "--results-root", str(results / "mrlora/UCIT_router"),
                                  "--output", str(logs / "metrics.json"), "--expected-count", str(sample["count"]),
                                  "--hashes-json", json.dumps(sample["hashes"], sort_keys=True)],
                            kind="extra", sequence="router", metadata={"validation_fix": "mrlora_ucit_verified_metrics"}))
    planned_files = ("mcitlib/runner.py", "mcitlib/dataflow.py", "mcitlib/validation_batch.py",
                     "mcitlib/doctor.py", "mcitlib/selfcheck.py",
                     "mcitlib/eval/metrics.py", "mcitlib/methods/_mrlora/prompts.py",
                     "mcitlib/methods/_mrlora/router_results/ucit_validation.py",
                     "mcitlib/methods/_mrlora/eval_runners/model_ucit_router.py")
    sources = {"state_sha256": _hash(state_path), "commands_sha256": _hash(commands_path),
               "events_sha256": _hash(events_path), "sample_manifest_sha256": sample["manifest_sha256"],
               "inference_and_scorer_sha256": {name: _hash(root / name) for name in code_files},
               "planned_code_sha256": {name: _hash(root / name) for name in planned_files},
               "reporter_source_sha256": _hash(Path(__file__).resolve())}
    return {"schema_version": 1, "old_run_dir": str(old_run), "results_root": str(results), "logs_dir": str(logs),
            "config_root": str(config_root), "python": str(python), "base_model": str(model_base),
            "source": sources, "sample": sample, "checkpoints": checkpoints,
            "reused": reuse, "missing_expert_cells": [{"expert": stage, "domain": domain} for stage in range(1, 7)
                                                        for domain in EXPERTS if (stage, domain) not in reused],
            "commands": [command.as_dict() for command in combined],
            "note": "Plan only; run the environment and GPU doctor before execution. All six experts are used retrospectively. "
                    "No inference has been run by planning; weight bytes were not hashed. Selected source hashes do not freeze "
                    "all dependencies or prove launch-time environment identity."}


def _copy_verified(source: Path, output: Path, expected: str) -> None:
    if not source.is_file():
        raise ValueError(f"source changed after review: {source}")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"source changed after review: {source}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as out:
        out.write(data)


def _finish(results: Path, output: Path, count: int, hashes: dict[str, str]) -> dict[str, Any]:
    """Require the full routed matrix and matching scorer denominators before metrics."""
    from ...eval.metrics import metrics_from_path

    if count < 1 or set(hashes) != set(EXPERTS) | {"VizWiz:caption", "Flickr30k:caption"}:
        raise ValueError("invalid sample evidence for UCIT metrics")
    expected_files = set()
    for stage in range(1, len(EXPERTS) + 1):
        for domain in EXPERTS[:stage]:
            folder = results / "each_dataset" / domain / f"MR-LoRA-router-task{stage}"
            score = folder / "Result.text"
            expected_files.add(score)
            match = re.search(r"(?m)^Samples:\s*(\d+)\s*$", score.read_text(encoding="utf-8"))
            if not match or int(match.group(1)) != count:
                raise ValueError(f"scorer denominator differs from verified sample: {score}")
            stats = json.loads((folder / "routing_stats.json").read_text(encoding="utf-8"))
            if (stats.get("samples") != count or stats.get("expected_samples") != count or
                    stats.get("annotation_sha256") != hashes[domain] or
                    stats.get("expected_annotation_sha256") != hashes[domain] or
                    stats.get("expert_pool") != list(EXPERTS) or
                    stats.get("retrospective_six_expert_pool") is not True or
                    (domain in {"VizWiz", "Flickr30k"} and not stats.get("caption_alignment_verified"))):
                raise ValueError(f"routing evidence does not match verified samples: {folder}")
            routed = _read_jsonl(folder / "routed_predictions.jsonl")
            scored = _read_jsonl(folder / "Result.json")
            _index(routed, folder / "routed_predictions.jsonl")
            _index(scored, folder / "Result.json")
            if (len(routed) != count or len(scored) != count or
                    any(type(a) is not type(b) or a != b for a, b in
                        zip((_identifier(row, folder) for row in routed),
                            (_identifier(row, folder) for row in scored)))):
                raise ValueError(f"scorer question IDs or count differ from routed predictions: {folder}")
    if set(results.rglob("Result.text")) != expected_files:
        raise ValueError("routed result directory contains missing or unexpected score files")
    result = metrics_from_path(results, tasks=EXPERTS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def _reviewed_plan(path: Path) -> dict[str, Any]:
    reviewed = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = build_plan(reviewed["old_run_dir"], reviewed["results_root"], reviewed["logs_dir"],
                         config_root=reviewed["config_root"])
    if reviewed != rebuilt:
        raise ValueError("source evidence or planned commands changed since review")
    if Path(sys.executable).resolve() != Path(reviewed["python"]).resolve():
        raise ValueError(f"this action must use the recorded job interpreter: {reviewed['python']}")
    return reviewed


def _doctor_plan(reviewed: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    """Run the standard environment, GPU, entry-point and input checks for this plan."""
    from ...doctor import (
        OK,
        Report,
        check_entry_points,
        check_environment,
        check_gpus,
        check_inputs,
        check_vendored_imports,
    )
    from ...selfcheck import distinct_entry_points

    root = repo_root()
    report = Report()
    commands = [Command(**item) for item in reviewed["commands"]]
    # Selfcheck classifies the interpreter used by this process, including symlink aliases.
    for command in commands:
        if Path(command.argv[0]).resolve() == Path(reviewed["python"]).resolve():
            command.argv[0] = sys.executable
    state = json.loads((Path(reviewed["old_run_dir"]) / "state.json").read_text(encoding="utf-8"))
    project = Path(state["project"]).resolve()
    paths = PathResolver.from_file(Path(reviewed["old_run_dir"]) / "jobs/mrlora/ucit/paths.effective.yaml",
                                   repo_root=project)
    check_environment(report, "llava", root)
    check_gpus(report, 4, "mrlora", ["0", "1", "2", "3"])
    check_vendored_imports(report, root)
    entries = distinct_entry_points(commands)
    if not entries:
        raise ValueError("doctor could not classify any planned entry point")
    check_entry_points(report, commands, root)
    check_inputs(report, commands, Path(reviewed["base_model"]).parent, project / "data",
                 (Path(reviewed["results_root"]), paths.paths.ckpt_root))
    report.add(OK, "verified source and dataflow", "read-only plan rebuilt from recorded inputs with no gaps")
    return {**report.as_dict(), "plan": str(plan_path.resolve()), "plan_sha256": _hash(plan_path),
            "checked_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
            "note": "Doctor checks environment, visible GPUs, imports and sample inputs; it does not execute inference or load weights."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="write a reviewable plan without creating evaluation outputs")
    plan.add_argument("--old-run-dir", type=Path, required=True)
    plan.add_argument("--results-root", type=Path, required=True)
    plan.add_argument("--logs-dir", type=Path, required=True)
    plan.add_argument("--plan-output", type=Path, required=True)
    doctor = sub.add_parser("doctor", help="check a reviewed evaluation plan without loading weights")
    doctor.add_argument("--plan-file", type=Path, required=True)
    doctor.add_argument("--report-output", type=Path, required=True)
    execute = sub.add_parser("execute", help="revalidate and execute an approved plan")
    execute.add_argument("--plan-file", type=Path, required=True)
    execute.add_argument("--doctor-report", type=Path, required=True)
    copy = sub.add_parser("_copy", help=argparse.SUPPRESS)
    copy.add_argument("--source", type=Path, required=True)
    copy.add_argument("--output", type=Path, required=True)
    copy.add_argument("--sha256", required=True)
    finish = sub.add_parser("_finish", help=argparse.SUPPRESS)
    finish.add_argument("--results-root", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    finish.add_argument("--expected-count", type=int, required=True)
    finish.add_argument("--hashes-json", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "_copy":
            _copy_verified(args.source, args.output, args.sha256)
            return 0
        if args.action == "_finish":
            result = _finish(args.results_root, args.output, args.expected_count, json.loads(args.hashes_json))
            print(json.dumps(result, ensure_ascii=False))
            return 0
        if args.action == "plan":
            result = build_plan(args.old_run_dir, args.results_root, args.logs_dir)
            root = repo_root()
            plan_output = managed_path(args.plan_output, root, within=root / "reports", label="plan output")
            plan_output.parent.mkdir(parents=True, exist_ok=True)
            plan_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"plan": str(plan_output), "reused_cells": len(result["reused"]),
                              "missing_cells": len(result["missing_expert_cells"]), "commands": len(result["commands"])}))
            return 0
        reviewed = _reviewed_plan(args.plan_file)
        if args.action == "doctor":
            root = repo_root()
            report_output = managed_path(args.report_output, root, within=root / "reports", label="doctor report")
            if report_output.exists():
                raise ValueError(f"refusing to overwrite doctor report: {report_output}")
            result = _doctor_plan(reviewed, args.plan_file)
            report_output.parent.mkdir(parents=True, exist_ok=True)
            with report_output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(json.dumps(result, ensure_ascii=False))
            return 1 if result["failed"] else 0
        doctor_result = json.loads(args.doctor_report.read_text(encoding="utf-8"))
        if doctor_result.get("failed") is not False or doctor_result.get("plan_sha256") != _hash(args.plan_file):
            raise ValueError("a passing doctor report for this exact reviewed plan is required")
        commands = [Command(**item) for item in reviewed["commands"]]
        execute_commands(commands, reviewed["logs_dir"], repo_root(), logs_root=repo_root() / "logs")
        print(json.dumps({"status": "completed", "results_root": reviewed["results_root"], "logs_dir": reviewed["logs_dir"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"mrlora UCIT evaluation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
