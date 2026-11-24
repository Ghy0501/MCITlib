# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Prepare and, only when explicitly invoked, run bounded four-GPU validation jobs.

Preparation reads the previous sample profile but writes only to a new directory
under this checkout's logs. Each job is independently checked and executed with
its original method environment. This is an integration smoke, not a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

from mcitlib.config.schema import Command, PathSet
from mcitlib.paths import PathResolver, managed_path, process_env, repo_root
from mcitlib.registry import get_method
from mcitlib.runner import build_sequence_commands
from mcitlib.smoke import _first_records, _validate_records

METHODS = ("moelora", "olora", "disco", "hide", "smolora", "clmoe", "modalprompt", "sefe")
BENCHMARKS = ("ucit", "mllm_dcl")
DEFAULT_STATE = Path(__file__).resolve().parents[1] / "logs/validation_3k_1k/llava-sample-20261106-4gpu-01/state.json"
LIMIT_TOKENS = 16


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _value(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"expected exactly one {flag}: {argv[:6]}")
    return argv[argv.index(flag) + 1]


def _set(argv: list[str], flag: str, value: str, *, append: bool = False) -> None:
    if append and flag not in argv:
        argv.extend((flag, value))
        return
    _value(argv, flag)
    argv[argv.index(flag) + 1] = value


def _inside(path: str | Path, root: Path) -> bool:
    candidate = Path(path).resolve()
    return candidate == root or root in candidate.parents


def _command(value: dict[str, Any]) -> Command:
    return Command(**value)


def _evaluation_group(commands: list[Command]) -> tuple[list[Command], Command]:
    groups: dict[tuple[str, str], list[Command]] = {}
    for command in commands:
        if command.kind != "eval" or "--answers-file" not in command.argv:
            continue
        question = _value(command.argv, "--question-file")
        directory = str(Path(_value(command.argv, "--answers-file")).parent)
        groups.setdefault((question, directory), []).append(command)
    if not groups:
        raise ValueError("runner produced no evaluation inference group")
    (_, directory), inference = list(groups.items())[-1]
    if len(inference) != 4 or {_value(item.argv, "--chunk-idx") for item in inference} != set("0123"):
        raise ValueError(f"expected four inference chunks in {directory}")
    answers = {_value(item.argv, "--answers-file") for item in inference}
    merges = [item for item in commands if item.metadata.get("merge_chunks") and
              "--inputs" in item.argv and set(item.argv[item.argv.index("--inputs") + 1:]) == answers]
    if len(merges) != 1:
        raise ValueError(f"expected a unique four-chunk merge for {directory}")
    return inference, merges[0]


def _source(path: Path, destination: Path, count: int, folder: Path, *, training: bool) -> dict[str, Any]:
    records = _first_records(path, count)
    if training:
        _validate_records(records, folder)
    else:
        identifiers = []
        for item in records:
            image = item.get("image")
            target = (folder / image).resolve() if isinstance(image, str) else folder
            if (item.get("question_id") is None or not isinstance(item.get("text"), str) or
                    not isinstance(image, str) or folder.resolve() not in target.parents or not target.is_file()):
                raise ValueError(f"invalid evaluation record or image in {path}")
            identifiers.append(str(item["question_id"]))
        if len(set(identifiers)) != count:
            raise ValueError(f"duplicate question IDs in {path}")
    _write_json(destination, records)
    return {"source": str(path), "source_sha256": _sha(path), "subset": str(destination),
            "subset_sha256": _sha(destination), "count": count,
            **({"question_ids": identifiers} if not training else {})}


def _job_environment(root: Path, method: str, old: dict, python_modalprompt: str | None) -> tuple[str, str]:
    if method != "modalprompt":
        return old["python"], old["environment"]
    if not python_modalprompt:
        raise ValueError("ModalPrompt requires --python-modalprompt for its isolated DeepSpeed 0.12.6 environment")
    # Keep the virtualenv path itself: resolving a symlink can select its base interpreter.
    python = os.path.abspath(os.path.join(root, os.path.expanduser(python_modalprompt)))
    return python, get_method(method).env


def _environment_error(method: str, job: dict[str, Any]) -> str | None:
    if method == "modalprompt" and job.get("environment") != get_method(method).env:
        return "ModalPrompt plan requires the isolated llava-modalprompt-ds126 environment"
    return None


def _job_plan(root: Path, run_dir: Path, state: dict, method: str, benchmark: str,
              samples: int, max_steps: int, port: int, python_modalprompt: str | None = None) -> dict[str, Any]:
    old = state["jobs"][f"{method}/{benchmark}"]
    python, environment = _job_environment(root, method, old, python_modalprompt)
    runtime = Path(state["project"]).resolve()
    original = PathResolver.from_file(Path(old["effective_paths"]), repo_root=runtime)
    job_dir = run_dir / "jobs" / method / benchmark
    ckpts, results = run_dir / "ckpts" / method / benchmark, run_dir / "results" / method / benchmark
    paths = PathSet(original.paths.model_root, original.paths.data_root, ckpts, root, results, job_dir)
    resolver = PathResolver(paths, original.mappings)
    cache = {"TRITON_CACHE_DIR": str(run_dir / "cache/triton"),
             "TORCH_EXTENSIONS_DIR": str(run_dir / "cache/torch"),
             "HF_HOME": str(run_dir / "cache/huggingface"),
             "XDG_CACHE_HOME": str(run_dir / "cache/xdg"),
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    stages, carry = [], []
    for task in (1, 2):
        commands = build_sequence_commands(root / "configs", "llava", method, benchmark, paths=resolver,
                                           task_id=task, validation_fixes=True,
                                           execution_profile="four_gpu_validation", gpus="0,1,2,3", master_port=port)
        if method == "modalprompt":
            commands += build_sequence_commands(root / "configs", "llava", method, benchmark, paths=resolver,
                                                task_id=task, only_eval=True, validation_fixes=True,
                                                execution_profile="four_gpu_validation", gpus="0,1,2,3", master_port=port)
        trains = [item for item in commands if item.kind == "train"]
        if len(trains) != 1:
            raise ValueError(f"{method}/{benchmark} task {task}: expected one training command")
        train = trains[0]
        if _value(train.argv, "--include") != "localhost:0,1,2,3":
            raise ValueError("four GPU training assignment is required")
        train_source = Path(_value(train.argv, "--data_path"))
        train_folder = Path(_value(train.argv, "--image_folder"))
        train_subset = job_dir / f"task{task}-train.json"
        train_sample = _source(train_source, train_subset, samples, train_folder, training=True)
        for flag, value in (("--data_path", str(train_subset)), ("--per_device_train_batch_size", "1"),
                            ("--per_device_eval_batch_size", "1"), ("--gradient_accumulation_steps", "1"),
                            ("--dataloader_num_workers", "0"), ("--save_strategy", "no"), ("--report_to", "none"),
                            ("--logging_steps", "1")):
            _set(train.argv, flag, value)
        _set(train.argv, "--max_steps", str(max_steps), append=True)
        # Never let Transformers replace NaN/Inf loss entries with a logged average.
        _set(train.argv, "--logging_nan_inf_filter", "False", append=True)
        if method == "modalprompt" and task == 2:
            if _value(train.argv, "--previous_task_model_path") != stages[0]["checkpoint"]:
                raise ValueError("ModalPrompt task 2 must load this smoke task 1 checkpoint")
        train.env.update(cache)
        if method == "modalprompt":
            # Bounded evidence is separate for each job and stays under its log root.
            train.env["MCITLIB_MODAL_OBSERVE_DIR"] = str(job_dir / "observations")
        inference, merge = _evaluation_group(commands)
        eval_source = Path(_value(inference[0].argv, "--question-file"))
        eval_folder = Path(_value(inference[0].argv, "--image-folder"))
        eval_subset = job_dir / f"task{task}-questions.json"
        eval_sample = _source(eval_source, eval_subset, 4, eval_folder, training=False)
        for command in inference:
            _set(command.argv, "--question-file", str(eval_subset))
            if _value(command.argv, "--num-chunks") != "4":
                raise ValueError("four inference chunks are required")
            index = _value(command.argv, "--chunk-idx")
            if command.env.get("CUDA_VISIBLE_DEVICES") != index:
                raise ValueError("each inference chunk must pin its corresponding GPU")
            if command.argv[1:3] != ["-m", "mcitlib.backbones.llava.eval_runners.entry"]:
                raise ValueError("unexpected inference entry")
            target = command.argv[3]
            command.argv[3] = "tools.bounded_eval_entry"
            command.env.update(cache)
            command.env.update(MCITLIB_SMOKE_EVAL_TARGET=target, MCITLIB_SMOKE_MAX_NEW_TOKENS=str(LIMIT_TOKENS))
        merge.env.update(cache)
        extras = [item for item in commands if item.kind == "extra"]
        if method != "sefe" and extras:
            raise ValueError(f"unrecognized extra stages for {method}/{benchmark}")
        merge_model = [item for item in extras if "mcitlib/methods/_sefe/merge.py" in item.argv]
        key_next = [item for item in extras if "mcitlib/methods/_sefe/key_elements.py" in item.argv]
        if method == "sefe" and (len(merge_model) != 1 or len(key_next) != 1):
            raise ValueError("SEFE requires its recorded model merge and key-element steps")
        for item in merge_model + key_next:
            item.env.update(cache)
        steps = [*carry, train, *merge_model, *inference, merge]
        carry = key_next
        stages.append({"task": task, "training_sample": train_sample, "evaluation_sample": eval_sample,
                       "checkpoint": _value(train.argv, "--output_dir"),
                       "merged_answers": _value(merge.argv, "--output"),
                       "commands": [item.as_dict() for item in steps]})
    return {"method": method, "benchmark": benchmark, "python": python, "environment": environment,
            "job_dir": str(job_dir), "ckpt_root": str(ckpts), "results_root": str(results),
            "model_root": str(paths.model_root), "data_root": str(paths.data_root),
            "external_roots": list(resolver.external_roots(benchmark)), "stages": stages}


def _selected(values: list[str] | None) -> list[str]:
    all_jobs = [f"{method}/{benchmark}" for method in METHODS for benchmark in BENCHMARKS]
    if not values:
        return all_jobs
    if len(set(values)) != len(values) or any(value not in all_jobs for value in values):
        raise ValueError("--job must be a unique method/benchmark from the supported set")
    return values


def prepare(args: argparse.Namespace) -> int:
    root = repo_root().resolve()
    run_dir = managed_path(args.run_dir, root, within=root / "logs", label="bounded smoke run")
    if run_dir.exists() or run_dir.is_symlink():
        raise ValueError(f"bounded smoke run already exists: {run_dir}")
    if not 4 <= args.samples <= 8 or not 2 <= args.max_steps <= 3 or not 1 <= args.master_port <= 65535:
        raise ValueError("samples must be 4..8, max-steps 2..3, and master-port 1..65535")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=False)
    plan: dict[str, Any] = {"version": 1, "project": str(root), "run_dir": str(run_dir),
                            "source_state": str(args.state.resolve()), "source_state_sha256": _sha(args.state),
                            "limits": {"samples": args.samples, "evaluation_samples": 4, "max_steps": args.max_steps,
                                       "train_batch_per_device": 1, "gradient_accumulation_steps": 1,
                                       "dataloader_workers": 0, "max_new_tokens": LIMIT_TOKENS,
                                       "gpus": ["0", "1", "2", "3"], "execution_profile": "four_gpu_validation"},
                            "jobs": {}}
    for key in _selected(args.job):
        method, benchmark = key.split("/")
        try:
            plan["jobs"][key] = _job_plan(root, run_dir, state, method, benchmark,
                                           args.samples, args.max_steps, args.master_port,
                                           getattr(args, "python_modalprompt", None))
        except (OSError, ValueError, KeyError, TypeError) as error:
            plan["jobs"][key] = {"method": method, "benchmark": benchmark, "preparation_error": str(error)}
    path = run_dir / "plan.json"
    _write_json(path, plan)
    (run_dir / "plan.sha256").write_text(_sha(path) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(path), "sha256": _sha(path), "jobs": len(plan["jobs"]),
                      "preparation_errors": {key: value["preparation_error"] for key, value in plan["jobs"].items()
                                             if "preparation_error" in value}}, ensure_ascii=False, indent=2))
    return int(any("preparation_error" in job for job in plan["jobs"].values()))


def _read_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    root = repo_root().resolve()
    if not _inside(path, root / "logs") or path.name != "plan.json":
        raise ValueError("plan must be a bounded smoke plan under this checkout's logs")
    if (path.parent / "plan.sha256").read_text(encoding="utf-8").strip() != _sha(path):
        raise ValueError("bounded smoke plan hash does not match")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("project") != str(root) or plan.get("run_dir") != str(path.parent):
        raise ValueError("bounded smoke plan is for a different checkout or run")
    return plan


def _commands(plan: dict[str, Any], job: dict[str, Any], tasks: int) -> list[Command]:
    if tasks not in (1, 2):
        raise ValueError("--tasks must be 1 or 2")
    run_dir = Path(plan["run_dir"])
    result = []
    for stage in job["stages"][:tasks]:
        for sample in (stage["training_sample"], stage["evaluation_sample"]):
            if _sha(Path(sample["source"])) != sample["source_sha256"] or _sha(Path(sample["subset"])) != sample["subset_sha256"]:
                raise ValueError(f"source or bounded sample changed: {sample['subset']}")
        for value in stage["commands"]:
            command = _command(value)
            if Path(command.cwd).resolve() != Path(plan["project"]):
                raise ValueError("smoke command cwd must be the isolated checkout")
            for flag in ("--output_dir", "--answers-file", "--output", "--output-dir", "--save-model-path"):
                if flag in command.argv and not _inside(_value(command.argv, flag), run_dir):
                    raise ValueError(f"{flag} escapes the isolated smoke directory")
            if "mcitlib/methods/_sefe/key_elements.py" in command.argv:
                index = command.argv.index("mcitlib/methods/_sefe/key_elements.py")
                if not _inside(command.argv[index + 2], run_dir):
                    raise ValueError("SEFE key-element output directory escapes the smoke directory")
            if command.kind == "train":
                if (_value(command.argv, "--include") != "localhost:0,1,2,3" or
                    _value(command.argv, "--max_steps") != str(plan["limits"]["max_steps"]) or
                    _value(command.argv, "--per_device_train_batch_size") != "1" or
                    _value(command.argv, "--gradient_accumulation_steps") != "1" or
                    _value(command.argv, "--save_strategy") != "no" or
                    _value(command.argv, "--logging_nan_inf_filter").lower() != "false" or
                    _value(command.argv, "--data_path") != stage["training_sample"]["subset"]):
                    raise ValueError("training command violates bounded smoke limits")
            if "--answers-file" in command.argv:
                index = _value(command.argv, "--chunk-idx")
                if (index not in "0123" or command.env.get("CUDA_VISIBLE_DEVICES") != index or
                    _value(command.argv, "--num-chunks") != "4" or
                    _value(command.argv, "--question-file") != stage["evaluation_sample"]["subset"] or
                    command.argv[3] != "tools.bounded_eval_entry" or
                    command.env.get("MCITLIB_SMOKE_MAX_NEW_TOKENS") != str(LIMIT_TOKENS)):
                    raise ValueError("inference command violates bounded smoke limits")
            result.append(command)
    return result


def _doctor(plan: dict[str, Any], job: dict[str, Any], commands: list[Command], skip_gpu: bool) -> dict[str, Any]:
    from dataclasses import replace

    from mcitlib import doctor

    root = Path(plan["project"])
    report = doctor.Report()
    report.add(doctor.OK, "bounded sequence", f"{job['method']}/{job['benchmark']}: {len(commands)} exact steps")
    doctor.check_environment(report, job["environment"], root)
    if not skip_gpu:
        doctor.check_gpus(report, doctor.gpus_needed(commands), job["method"], ["0", "1", "2", "3"])
    doctor.check_vendored_imports(report, root)
    outputs = (Path(job["ckpt_root"]), Path(job["results_root"]))
    doctor.check_inputs(report, commands, Path(job["model_root"]), Path(job["data_root"]), outputs)
    doctor.check_dataflow(report, commands, tuple(job["external_roots"]), (), tuple(map(str, outputs)))
    doctor.check_prerequisites(report, job["method"], job["benchmark"], commands, root, outputs[0])
    entries = ([replace(item, env={**item.env, "CUDA_VISIBLE_DEVICES": ""}) for item in commands]
               if skip_gpu else commands)
    doctor.check_entry_points(report, entries, root)
    # In particular, SEFE task 2 loads a merged model using --model_name_or_path.
    from mcitlib.dataflow import produced_paths

    produced: set[str] = set()
    missing = []
    for command in commands:
        if "--model_name_or_path" in command.argv:
            path = _value(command.argv, "--model_name_or_path")
            if _inside(path, outputs[0]) and path not in produced:
                missing.append(path)
        produced.update(produced_paths([command]))
    report.add(doctor.FAIL if missing else doctor.OK, "generated training models",
               ", ".join(missing) if missing else "all generated model paths have earlier producers")
    return report.as_dict()


def _check_checkpoint(stage: dict[str, Any], max_steps: int) -> dict[str, Any]:
    from mcitlib.validation_batch import checkpoint_error

    train = next(item for item in stage["commands"] if item["kind"] == "train")
    if error := checkpoint_error(train["argv"]):
        raise RuntimeError(error)
    output = Path(stage["checkpoint"])
    if "--lora_enable" in train["argv"] and _value(train["argv"], "--lora_enable").lower() == "true":
        adapter_config = output / "adapter_config.json"
        if not adapter_config.is_file() or not adapter_config.stat().st_size:
            raise RuntimeError(f"missing or empty adapter_config.json in {output}")
    state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != max_steps:
        raise RuntimeError(f"expected {max_steps} optimizer steps in {output}")
    logs = state.get("log_history", [])
    losses = [row["loss"] for row in logs if "loss" in row]
    lrs = [row["learning_rate"] for row in logs if "learning_rate" in row]
    if (not losses or not lrs or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses + lrs)
            or not any(x > 0 for x in lrs)):
        raise RuntimeError(f"missing/non-finite loss or no positive logged learning rate in {output}")
    files = {path.name: path.stat().st_size for path in output.iterdir() if path.is_file()}
    update_evidence = "optimizer steps and positive learning rate logged"
    if "--lora_enable" in train["argv"] and _value(train["argv"], "--lora_enable").lower() == "true":
        import torch

        adapter_file = output / "adapter_model.bin"
        if adapter_file.is_file():
            adapter = torch.load(adapter_file, map_location="cpu")
        else:
            from safetensors.torch import load_file
            adapter = load_file(str(output / "adapter_model.safetensors"), device="cpu")
        weights = [value for name, value in adapter.items() if "lora_B" in name and torch.is_tensor(value)]
        if not weights or not all(torch.isfinite(value).all().item() for value in weights) or not any(
            torch.count_nonzero(value).item() for value in weights
        ):
            raise RuntimeError(f"LoRA B weights are missing, non-finite or unchanged from zero initialization in {output}")
        update_evidence = "finite nonzero LoRA B weights and positive learning rate"
    return {"checkpoint": str(output), "files_bytes": files, "global_step": state["global_step"],
            "loss": losses, "learning_rate": lrs, "update_evidence": update_evidence}


def _check_prompt_resume(previous: dict[str, Any], current: dict[str, Any]) -> str:
    import torch

    first = torch.load(Path(previous["checkpoint"]) / "non_lora_trainables.bin", map_location="cpu")
    second = torch.load(Path(current["checkpoint"]) / "non_lora_trainables.bin", map_location="cpu")
    matching = [name for name in first if name.endswith("embed_tokens.weight") and name in second]
    if len(matching) != 1:
        raise RuntimeError("ModalPrompt checkpoints need one matching token embedding matrix")
    before, after = first[matching[0]], second[matching[0]]
    train = next(item for item in current["commands"] if item["kind"] == "train")
    prompt_tokens = int(_value(train["argv"], "--prefix_len")) * int(_value(train["argv"], "--num_tasks"))
    if (before.shape != after.shape or before.shape[0] < prompt_tokens or
            not torch.isfinite(after[-prompt_tokens:]).all().item() or
            torch.equal(before[-prompt_tokens:], after[-prompt_tokens:])):
        raise RuntimeError("ModalPrompt task 2 prompt embeddings did not change after task 1 reload")
    return f"prompt embeddings changed from the loaded task 1 checkpoint ({prompt_tokens} prompt tokens)"


def _check_answers(stage: dict[str, Any]) -> dict[str, Any]:
    path = Path(stage["merged_answers"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("question_id")) for row in rows]
    if (len(rows) != 4 or set(ids) != set(stage["evaluation_sample"]["question_ids"]) or
            len(set(ids)) != 4 or any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows)):
        raise RuntimeError(f"generated answers are incomplete, duplicated, or empty: {path}")
    return {"path": str(path), "count": len(rows), "sha256": _sha(path), "question_ids": ids}


def _worker(plan_path: Path, job_key: str, tasks: int, action: str, skip_gpu: bool, min_free_gib: float) -> int:
    from mcitlib.runner import execute_commands

    plan = _read_plan(plan_path)
    job = plan["jobs"][job_key]
    if "preparation_error" in job:
        raise ValueError(job["preparation_error"])
    if error := _environment_error(job_key.split("/")[0], job):
        raise ValueError(error)
    commands = _commands(plan, job, tasks)
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    doctor_result = _doctor(plan, job, commands, skip_gpu)
    _write_json(job_dir / f"doctor-{tasks}-{'cpu' if skip_gpu else 'gpu'}.json", doctor_result)
    if doctor_result["failed"]:
        return 1
    if action == "doctor":
        return 0
    if skip_gpu:
        raise ValueError("GPU execution cannot use the CPU-only doctor")
    if any(Path(stage["checkpoint"]).exists() for stage in job["stages"][:tasks]):
        raise ValueError("checkpoint output already exists; prepare a fresh run")
    state: dict[str, Any] = {"job": job_key, "status": "running", "tasks": tasks,
                             "plan_sha256": _sha(plan_path), "checkpoints": {}, "answers": {}}
    _write_json(job_dir / "result.json", state)

    def event(item: dict[str, Any]) -> None:
        with (job_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if item["status"] != "completed":
            return
        stage = job["stages"][item["task"] - 1]
        if item["kind"] == "train":
            state["checkpoints"][str(item["task"])] = _check_checkpoint(stage, plan["limits"]["max_steps"])
            if job["method"] == "modalprompt" and item["task"] == 2:
                state["checkpoints"]["2"]["update_evidence"] = _check_prompt_resume(job["stages"][0], stage)
        elif item["command"]["metadata"].get("merge_chunks"):
            state["answers"][str(item["task"])] = _check_answers(stage)
        _write_json(job_dir / "result.json", state)

    def before_step(index: int, command: Command) -> None:
        free = shutil.disk_usage(job_dir).free / 2**30
        if free < min_free_gib:
            raise RuntimeError(f"only {free:.1f} GiB free, need {min_free_gib:.1f} GiB")

    try:
        execute_commands(commands, job_dir, plan["project"], logs_root=job_dir,
                         on_event=event, before_step=before_step)
        state["status"] = "completed"
    except (RuntimeError, ValueError, OSError) as error:
        state.update(status="failed", error=str(error))
    _write_json(job_dir / "result.json", state)
    return int(state["status"] != "completed")


def _run_isolated(argv: list[str], env: dict[str, str], cwd: str, log, timeout: float) -> int:
    """Own a process group so only this smoke job and its ranks are stopped on timeout."""
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # The unreaped leader's PID cannot be reused; DeepSpeed's local launcher
        # and ranks inherit this group. Kill just this job before moving on.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        return 124


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--state", type=Path, default=DEFAULT_STATE)
    create.add_argument("--samples", type=int, default=4)
    create.add_argument("--max-steps", type=int, default=2)
    create.add_argument("--master-port", type=int, default=29913)
    create.add_argument("--job", action="append")
    create.add_argument("--python-modalprompt", help="Python from the isolated llava-modalprompt-ds126 environment")
    for action in ("doctor", "execute"):
        command = sub.add_parser(action)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--job", action="append", required=True)
        command.add_argument("--tasks", type=int, choices=(1, 2), default=1)
        command.add_argument("--skip-gpu", action="store_true")
        command.add_argument("--min-free-gib", type=float, default=20.0)
        command.add_argument("--timeout", type=int, default=1800, help="maximum seconds per job (default 1800)")
        command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args)
    if args.action == "execute" and args.skip_gpu:
        raise ValueError("execute requires the complete GPU doctor")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 10:
        raise ValueError("min-free-gib must be finite and at least 10")
    if not 60 <= args.timeout <= 3600:
        raise ValueError("job timeout must be 60..3600 seconds")
    plan = _read_plan(args.plan)
    jobs = _selected(args.job)
    if any(key not in plan["jobs"] for key in jobs):
        raise ValueError("job was not prepared in this plan")
    if args.worker:
        if len(jobs) != 1:
            raise ValueError("internal worker needs exactly one job")
        return _worker(args.plan, jobs[0], args.tasks, args.action, args.skip_gpu, args.min_free_gib)
    failures = {}
    for key in jobs:
        job = plan["jobs"][key]
        if "preparation_error" in job:
            failures[key] = job["preparation_error"]
            continue
        if error := _environment_error(key.split("/")[0], job):
            failures[key] = error
            continue
        python = job.get("python")
        if not python or not os.path.isfile(python) or not os.access(python, os.X_OK):
            failures[key] = f"Python environment is unavailable: {python or job.get('environment')}"
            continue
        env = process_env({}, plan["project"])
        env.update(CUDA_VISIBLE_DEVICES="" if args.skip_gpu else "0,1,2,3", PYTHONDONTWRITEBYTECODE="1",
                   TRITON_CACHE_DIR=str(Path(plan["run_dir"]) / "cache/triton"),
                   TORCH_EXTENSIONS_DIR=str(Path(plan["run_dir"]) / "cache/torch"),
                   HF_HOME=str(Path(plan["run_dir"]) / "cache/huggingface"),
                   XDG_CACHE_HOME=str(Path(plan["run_dir"]) / "cache/xdg"),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        argv = [job["python"], str(Path(__file__).resolve()), args.action, "--plan", str(args.plan),
                "--job", key, "--tasks", str(args.tasks), "--min-free-gib", str(args.min_free_gib), "--worker"]
        if args.skip_gpu:
            argv.append("--skip-gpu")
        job_dir = Path(job["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (job_dir / f"{args.action}-{args.tasks}.log").open("w", encoding="utf-8") as log:
                code = _run_isolated(argv, env, plan["project"], log, args.timeout)
        except OSError as error:
            failures[key] = f"{type(error).__name__}: {error}"
            continue
        if code:
            failures[key] = f"{'timed out' if code == 124 else f'exit {code}'}; see {job_dir}/{args.action}-{args.tasks}.log"
    print(json.dumps({"action": args.action, "tasks": args.tasks, "jobs": jobs, "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
