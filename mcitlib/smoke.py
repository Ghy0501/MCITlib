# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""A deliberately bounded, isolated LoRA-FT integration smoke run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config.schema import Command
from .paths import PathResolver, managed_path, process_env, repo_root
from .runner import build_sequence_commands

MAX_SAMPLES = 8
MAX_STEPS = 3
MAX_NEW_TOKENS = 8
SUPPORTED_BENCHMARKS = ("ucit", "mllm_dcl")


def _value(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"expected exactly one {flag} in the shared training command")
    return argv[argv.index(flag) + 1]


def _set_value(argv: list[str], flag: str, value: str, *, required: bool = True) -> None:
    if flag not in argv and not required:
        argv.extend([flag, value])
        return
    _value(argv, flag)
    argv[argv.index(flag) + 1] = value


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _safe_run_dir(run_dir: Path, resolver: PathResolver, command: Command, root: Path) -> Path:
    run_dir = run_dir.expanduser()
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    if run_dir.exists() or run_dir.is_symlink():
        raise ValueError(f"smoke run directory already exists: {run_dir}")
    run_dir = resolver.managed_run_dir(run_dir)
    output_roots = (resolver.paths.ckpt_root, resolver.results_root)
    input_flags = ("--model_name_or_path", "--vision_tower", "--pretrain_mm_mlp_adapter", "--data_path", "--image_folder")
    inputs = [Path(_value(command.argv, flag)) for flag in input_flags]
    # Protect input targets for every scope, even when the selected run uses a different benchmark.
    inputs += [Path(target) for mappings in resolver.mappings.values() for _, target in mappings]
    for path in (*output_roots, *inputs):
        path = (path if path.is_absolute() else root / path).resolve()
        if _overlap(run_dir, path):
            raise ValueError(f"smoke run directory overlaps an experiment output or input: {path}")
    if run_dir.exists() or run_dir.is_symlink():
        raise ValueError(f"smoke run directory already exists: {run_dir}")
    if not run_dir.parent.is_dir():
        raise ValueError(f"smoke run directory parent must already exist: {run_dir.parent}")
    return run_dir


def _safe_adapter_dir(parent: Path | None, run_dir: Path, resolver: PathResolver,
                      command: Command, root: Path, backbone: str = "llava") -> Path:
    name = "Task1_internvl_lora" if backbone == "internvl" else "Task1_llava_lora"
    if parent is None:
        return run_dir / name
    parent = parent.expanduser()
    parent = (parent if parent.is_absolute() else root / parent).resolve()
    if not parent.is_dir():
        raise ValueError(f"adapter parent must already exist: {parent}")
    adapter = parent / name
    if adapter.exists() or adapter.is_symlink():
        raise ValueError(f"smoke adapter directory already exists: {adapter}")
    inputs = [Path(_value(command.argv, flag)).resolve() for flag in (
        "--model_name_or_path", "--vision_tower", "--pretrain_mm_mlp_adapter", "--data_path", "--image_folder"
    )]
    inputs += [Path(target).resolve() for mappings in resolver.mappings.values() for _, target in mappings]
    protected = [run_dir, resolver.logs_root, resolver.results_root, *inputs]
    if any(_overlap(adapter, path.resolve()) for path in protected):
        raise ValueError("smoke adapter directory overlaps a run, results, or input path")
    if adapter == resolver.paths.ckpt_root or adapter in resolver.paths.ckpt_root.parents:
        raise ValueError("smoke adapter directory cannot contain the checkpoint root")
    return adapter


def build_plan(paths_file: Path | None, run_dir: Path, samples: int, max_steps: int,
               gpus: str | None = None, master_port: int | None = None,
               benchmark: str = "ucit", adapter_dir: Path | None = None,
               backbone: str = "llava") -> dict[str, Any]:
    """Construct the sole allowed train and inference commands without writing anything."""
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"smoke benchmark must be one of: {', '.join(SUPPORTED_BENCHMARKS)}")
    if backbone not in {"llava", "internvl"}:
        raise ValueError("smoke backbone must be llava or internvl")
    if not 1 <= samples <= MAX_SAMPLES or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"samples must be 1..{MAX_SAMPLES} and max-steps must be 1..{MAX_STEPS}")
    if master_port is not None and not 1 <= master_port <= 65535:
        raise ValueError("master-port must be 1..65535")
    root = repo_root()
    resolver = PathResolver.from_file(paths_file, repo_root=root)
    if backbone == "internvl":
        gpus = gpus or "0,1,2,3"
    profile = "four_gpu_validation" if backbone == "internvl" else None
    commands = build_sequence_commands(root / "configs", backbone, "lora_ft", benchmark, paths=resolver,
                                       task_id=1, gpus=gpus, master_port=master_port, execution_profile=profile)
    trains = [command for command in commands if command.kind == "train"]
    if len(trains) != 1:
        raise ValueError("expected exactly one task 1 training command")
    train = trains[0]
    if (train.task_id != 1 or train.sequence != "train" or train.argv[0] != "deepspeed"
            or _value(train.argv, "--module") != "mcitlib.backbones.llava.train_entry"
            or train.env.get("MCITLIB_METHOD") != "lora_ft"
            or train.env.get("MCITLIB_TRAIN_ENTRY") != "standard"
            or train.env.get("MCITLIB_MODEL_OVERRIDE")
            or _value(train.argv, "--lora_enable").lower() != "true"
            or any(flag in train.argv for flag in ("--previous_task_model_path", "--memory_data_path",
                                                   "--init_lora_from_gradients", "--resume_from_checkpoint",
                                                   "--lora_weight_path", "--pt_enable"))):
        raise ValueError("shared training command is not the supported initial LoRA-FT task")
    run_dir = _safe_run_dir(run_dir, resolver, train, root)
    source = _value(train.argv, "--data_path")
    images = _value(train.argv, "--image_folder")
    base = _value(train.argv, "--model_name_or_path")
    subset = run_dir / "train-subset.json"
    adapter = _safe_adapter_dir(adapter_dir, run_dir, resolver, train, root, backbone)
    cache_env = {"TRITON_CACHE_DIR": str(run_dir / "cache/triton"),
                 "TORCH_EXTENSIONS_DIR": str(run_dir / "cache/torch_extensions")}
    train.env.update(cache_env)
    for flag, value in (
        ("--data_path", str(subset)), ("--output_dir", str(adapter)),
        ("--per_device_train_batch_size", "1"), ("--per_device_eval_batch_size", "1"),
        ("--gradient_accumulation_steps", "1"), ("--dataloader_num_workers", "0"),
        ("--save_strategy", "no"), ("--report_to", "none"),
    ):
        _set_value(train.argv, flag, value)
    _set_value(train.argv, "--max_steps", str(max_steps), required=False)
    _set_value(train.argv, "--logging_dir", str(run_dir / "logs"), required=False)
    include = _value(train.argv, "--include")
    host, gpu_list = include.split(":", 1)
    gpu_ids = gpu_list.split(",")
    if host != "localhost" or not gpu_ids or any(not gpu.isdigit() for gpu in gpu_ids):
        raise ValueError(f"unsupported training GPU selection: {include}")
    if backbone == "internvl":
        sample_profile = train.metadata.pop("training_profile", {})
        train.metadata["smoke_budget"] = {"source_profile": sample_profile, "role": "bounded_smoke",
            "global_batch_size": len(gpu_ids), "microbatch": 1, "gradient_accumulation": 1,
            "max_steps": max_steps}
    infer_env = {key: value for key, value in train.env.items() if key != "MCITLIB_TRAIN_ENTRY"}
    infer_env.update({"MCITLIB_METHOD": "lora_ft", "CUDA_VISIBLE_DEVICES": gpu_ids[0], **cache_env})
    infer = Command(
        argv=["python", "-m", "mcitlib.backbones.llava.eval_runners.entry",
              "mcitlib.backbones.llava.smoke_infer", "--benchmark", benchmark,
              "--adapter", str(adapter), "--model-base", base,
              "--subset", str(subset), "--image-folder", images, "--output", str(run_dir / "inference.json")],
        cwd=str(root), env=infer_env,
        kind="smoke-inference", task_id=1,
    )
    return {
        "status": "preview", "backbone": backbone, "method": "lora_ft", "benchmark": benchmark, "task": 1,
        "run_dir": str(run_dir), "adapter_dir": str(adapter.parent) if adapter_dir is not None else None,
        "adapter_path": str(adapter), "source_data": source, "image_folder": images,
        "limits": {"samples": samples, "max_steps": max_steps, "batch_per_device": 1,
                   "gradient_accumulation_steps": 1, "workers": 0, "inference_samples": 1,
                   "max_new_tokens": MAX_NEW_TOKENS},
        "training_command": train.as_dict(), "inference_command": infer.as_dict(),
    }


def _first_records(path: Path, count: int) -> list[dict[str, Any]]:
    """Read exactly the first count array records, without loading the entire dataset."""
    decoder = json.JSONDecoder()
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        buffer, position, eof, first = "", 0, False, True
        while len(result) < count:
            if not eof and len(buffer) - position < 65536:
                buffer = buffer[position:]
                position = 0
                chunk = handle.read(65536)
                buffer += chunk
                eof = not chunk
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"{path}: fewer than {count} records")
                continue
            if first:
                if buffer[position] != "[":
                    raise ValueError(f"{path}: expected a JSON array")
                position += 1
                first = False
                continue
            if result:
                if buffer[position] == "]":
                    raise ValueError(f"{path}: fewer than {count} records")
                if buffer[position] != ",":
                    raise ValueError(f"{path}: expected a comma between records")
                position += 1
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"{path}: truncated JSON array")
                # Keep the comma in the buffer so it can be parsed after the next read.
                position = buffer.rfind(",", 0, position) if result else position
                continue
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(65536)
                if not chunk:
                    eof = True
                else:
                    buffer += chunk
                # Revisit the separator along with this incomplete record.
                if result:
                    position = buffer.rfind(",", 0, position)
                continue
            if not isinstance(record, dict):
                raise ValueError(f"{path}: record {len(result) + 1} is not an object")
            result.append(record)
            position = end
    return result


def _validate_records(records: list[dict[str, Any]], folder: Path) -> None:
    folder = folder.resolve()
    for index, record in enumerate(records):
        image = record.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"training record {index + 1} needs one image")
        image_path = (folder / image).resolve()
        if folder not in image_path.parents or not image_path.is_file():
            raise ValueError(f"training record {index + 1} has no image inside {folder}: {image}")
    messages = records[0].get("conversations")
    if not isinstance(messages, list) or not any(isinstance(item, dict) and item.get("from") == "human"
                                                  and isinstance(item.get("value"), str) and item["value"].strip()
                                                  for item in messages):
        raise ValueError("first training record needs a human conversation for inference")


def _run(command: dict[str, Any], root: Path, log: Path) -> None:
    argv = list(command["argv"])
    if argv[0] == "python":
        argv[0] = sys.executable
    env = process_env(command["env"], root)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    log = managed_path(log, root, label="smoke log")
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(argv, cwd=command["cwd"], env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{command['kind']} failed with exit {result.returncode}; log: {log}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def execute_plan(plan: dict[str, Any], paths_file: Path | None) -> dict[str, Any]:
    """Run the full doctor before creating a directory or executing either command."""
    from .doctor import run_doctor

    root = repo_root()
    benchmark = plan["benchmark"]
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"smoke benchmark must be one of: {', '.join(SUPPORTED_BENCHMARKS)}")
    backbone = plan.get("backbone", "llava")
    doctor_options = {"skip_gpu": False, "skip_entrypoints": False, "sequence": "train"}
    if backbone == "internvl":
        doctor_options.update(backbone_id=backbone, execution_profile="four_gpu_validation", gpus="0,1,2,3")
    report = run_doctor("lora_ft", benchmark, paths_file, root, **doctor_options)
    plan["doctor"] = report.as_dict()
    if report.failed:
        plan.update(status="doctor_failed", error="full doctor reported at least one failure")
        return plan
    # Rebuild after doctor so the safety checks and config used for execution are current.
    train = plan["training_command"]
    include = _value(train["argv"], "--include").split(":", 1)[1]
    port = int(_value(train["argv"], "--master_port"))
    fresh = build_plan(paths_file, Path(plan["run_dir"]), plan["limits"]["samples"],
                       plan["limits"]["max_steps"], include, port, benchmark=benchmark,
                       adapter_dir=Path(plan["adapter_dir"]) if plan.get("adapter_dir") else None,
                       backbone=backbone)
    fresh["doctor"] = plan["doctor"]
    plan = fresh
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(exist_ok=False)
    plan["status"] = "running"
    try:
        records = _first_records(Path(plan["source_data"]), plan["limits"]["samples"])
        _validate_records(records, Path(plan["image_folder"]))
        _write_json(run_dir / "train-subset.json", records)
        _run(plan["training_command"], root, run_dir / "train.log")
        adapter = Path(plan["adapter_path"])
        state = json.loads((adapter / "trainer_state.json").read_text(encoding="utf-8"))
        actual_steps = state.get("global_step")
        plan["actual_global_steps"] = actual_steps
        if type(actual_steps) is not int or actual_steps != plan["limits"]["max_steps"]:
            raise RuntimeError(f"unexpected trainer global_step: {actual_steps}")
        files = [adapter / name for name in ("config.json", "adapter_config.json", "adapter_model.bin",
                                               "non_lora_trainables.bin", "trainer_state.json")]
        missing = [str(path) for path in files if not path.is_file() or not path.stat().st_size]
        if missing:
            raise RuntimeError(f"missing or empty adapter files: {', '.join(missing)}")
        plan["adapter"] = {"path": str(adapter), "files": [str(path) for path in files]}
        _run(plan["inference_command"], root, run_dir / "inference.log")
        inference = json.loads((run_dir / "inference.json").read_text(encoding="utf-8"))
        if (inference.get("benchmark") != benchmark or inference.get("adapter_path") != str(adapter)
                or inference.get("loaded") is not True
                or type(inference.get("new_tokens")) is not int
                or not 0 <= inference["new_tokens"] <= MAX_NEW_TOKENS
                or not isinstance(inference.get("text"), str)):
            raise RuntimeError("inference result failed adapter, sample or token limit checks")
        plan["inference"] = inference
        plan["status"] = "completed"
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        plan.update(status="failed", error=str(error))
    _write_json(run_dir / "smoke-result.json", plan)
    return plan
