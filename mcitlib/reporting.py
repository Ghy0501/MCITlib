# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Read-only snapshots of validation queues; no model or GPU imports are needed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .eval.metrics import _parse_result_text, continual_metrics
from .registry import get_benchmark

_RESULT_DIRS = {"ucit": "UCIT", "mllm_dcl": "DCL", "mllm_acl": "ACL"}
_NO_RESULTS = {"pending", "not_run", "blocked", "blocked_disk", "blocked_hardware"}
_METRICS = ("mft", "mfn", "maa", "bwt")
_MAX_METADATA = 8 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve(value: Any, base: Path | None = None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base:
        path = base / path
    return path.resolve()


def _file_info(path: Path | None) -> dict[str, Any] | None:
    """Fingerprint small metadata only; never traverse or hash model weights."""
    if path is None:
        return None
    result: dict[str, Any] = {"path": str(path), "exists": False, "sha256": None}
    try:
        if not path.is_file():
            return result
        stat = path.stat()
        result.update(exists=True, size_bytes=stat.st_size,
                      modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
        if stat.st_size <= _MAX_METADATA:
            data = path.read_bytes()
            if len(data) <= _MAX_METADATA:
                result["sha256"] = hashlib.sha256(data).hexdigest()
        else:
            result["hash_note"] = "metadata exceeds size limit; not hashed"
    except OSError as exc:
        result["error"] = str(exc)
    return result


def _json_file(path: Path | None, warnings: list[str], *, optional: bool = True) -> Any:
    if path is None:
        return None
    try:
        if optional and not path.is_file():
            return None
        if path.stat().st_size > _MAX_METADATA:
            warnings.append(f"metadata exceeds {_MAX_METADATA} bytes: {path}")
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        warnings.append(f"cannot read {path}: {exc}")
        return None


def _elapsed(start: Any, end: Any) -> float | None:
    try:
        first = datetime.fromisoformat(start.replace("Z", "+00:00"))
        last = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if first.tzinfo is None or last.tzinfo is None:
            return None
        seconds = (last - first).total_seconds()
        return round(seconds, 3) if seconds >= 0 else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _git_info(project: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"project": str(project) if project else None, "git_head_at_snapshot": None,
                              "git_root": None,
                              "note": "Snapshot observations, not proof of code or dirty state when the run started."}
    if project and project.is_dir():
        try:
            env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
            def git(*args: str) -> str:
                return subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True,
                                      text=True, timeout=3, env=env).stdout.strip()
            result["git_root"] = git("rev-parse", "--show-toplevel")
            result["git_head_at_snapshot"] = git("rev-parse", "--verify", "HEAD")
        except (OSError, subprocess.SubprocessError):
            pass
        result["source_files"] = {name: _file_info(project / name) for name in (
            "mcitlib/validation_batch.py", "mcitlib/eval/metrics.py", "mcitlib/registry.py", "mcitlib/cli.py")}
    return result


def _sample_info(state: dict[str, Any], project: Path | None, source: Path | None,
                 warnings: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Find an explicit sample manifest, or one named by the recorded paths profile."""
    manifest_path = _resolve(state.get("sample_manifest"), project)
    recipe_path = _resolve(state.get("sample_recipe"), project)
    association = "state metadata" if manifest_path else None
    if manifest_path is None and project and source and source.is_file():
        try:
            text = source.read_text(encoding="utf-8") if source.stat().st_size <= _MAX_METADATA else ""
            candidates = sorted(set(re.findall(r"(?:\$\{repo_root\}/)?data/([\w.-]+)/", text)))
            matches = [project / "configs" / name for name in candidates
                       if (project / "configs" / name / "manifest.json").is_file()]
            if len(matches) == 1:
                manifest_path, recipe_path = matches[0] / "manifest.json", matches[0] / "recipe.json"
                association = "recorded paths profile references this sample data directory"
            elif len(matches) > 1:
                warnings.append("multiple sample manifests match the paths profile; none selected")
        except (OSError, UnicodeError) as exc:
            warnings.append(f"cannot inspect paths profile {source}: {exc}")
    sample = _json_file(manifest_path, warnings)
    recipe = _json_file(recipe_path, warnings)
    if sample is not None and not isinstance(sample, dict):
        warnings.append(f"sample manifest is not an object: {manifest_path}")
        sample = None
    result: dict[str, Any] = {
        "association": association, "manifest": _file_info(manifest_path), "recipe": _file_info(recipe_path),
        "version": sample.get("version") if sample else None,
        "seed": sample.get("seed") if sample else None,
        "recorded_recipe_sha256": sample.get("recipe_sha256") if sample else None,
        "sampling": sample.get("sampling") if sample else None,
        "overlap_rule": sample.get("overlap_rule") if sample else None,
        "caption_rule": sample.get("caption_rule") if sample else None,
        "recipe_version": recipe.get("version") if isinstance(recipe, dict) else None,
        "qualification": "Sample validation; this is not a claim of formal reproduction.",
    }
    if result["recorded_recipe_sha256"] and result["recipe"] and result["recipe"]["sha256"]:
        result["recipe_hash_matches"] = result["recorded_recipe_sha256"] == result["recipe"]["sha256"]
        if not result["recipe_hash_matches"]:
            warnings.append("sample recipe fingerprint differs from the sample manifest")
    samples: dict[str, dict[str, Any]] = {}
    for task in sample.get("tasks", []) if sample else []:
        if not isinstance(task, dict) or not task.get("benchmark") or not task.get("name"):
            continue
        splits = task.get("splits", {})
        split_info = {}
        for name in ("train", "test"):
            item = splits.get(name, {}) if isinstance(splits, dict) else {}
            if not isinstance(item, dict):
                item = {}
            split_info[name] = {key: item.get(key) for key in
                                ("sample_count", "source_count", "source", "source_sha256", "output", "output_sha256")}
        caption = task.get("caption_reference")
        samples.setdefault(task["benchmark"], {})[task["name"]] = {
            **split_info,
            "caption_reference": ({key: caption.get(key) for key in ("images", "annotations", "source", "source_sha256", "output", "output_sha256")}
                                  if isinstance(caption, dict) else None),
        }
    return result, samples


def _events(path: Path, warnings: list[str]) -> tuple[dict[str, Any] | None, int]:
    last = None
    count = 0
    try:
        if not path.is_file():
            return None, 0
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    warnings.append(f"incomplete final event at {path}:{number} ignored")
                    break
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("event is not an object")
                except (ValueError, TypeError) as exc:
                    warnings.append(f"invalid event at {path}:{number}: {exc}")
                    continue
                count += 1
                last = {key: value for key, value in event.items() if key not in {"command", "executed_argv"}}
    except (OSError, UnicodeError) as exc:
        warnings.append(f"cannot read events {path}: {exc}")
    return last, count


def _tail(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 8192))
            return handle.read().decode("utf-8", errors="replace").strip() or None
    except OSError:
        return None


def _configured_samples(path: Path, project: Path | None, benchmark: str, task_samples: dict[str, Any],
                        warnings: list[str]) -> dict[str, Any]:
    """Associate a split only with its exact, applicable source-to-output mapping."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_METADATA:
            return {task: None for task in task_samples}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        warnings.append(f"cannot inspect effective paths {path}: {exc}")
        return {task: None for task in task_samples}
    mappings = data.get("path_mappings", {}) if isinstance(data, dict) else {}
    replacements = {"${repo_root}": str(project) if project else ""}
    if isinstance(data, dict):
        for key in ("data_root", "model_root", "ckpt_root", "results_root", "logs_root"):
            if isinstance(data.get(key), str):
                replacements["${" + key + "}"] = data[key]
    def normalize(value: str) -> str:
        for token, replacement in replacements.items():
            if replacement:
                value = value.replace(token, replacement)
        return os.path.normpath(value)
    pairs: set[tuple[str, str]] = set()
    if isinstance(mappings, dict):
        for scope in ("common", benchmark):
            entries = mappings.get(scope, [])
            if isinstance(entries, list):
                pairs.update((normalize(entry["from"]), normalize(entry["to"])) for entry in entries
                             if isinstance(entry, dict) and isinstance(entry.get("from"), str)
                             and isinstance(entry.get("to"), str))
    def matches(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        source, target = item.get("source"), item.get("output")
        if not isinstance(source, str) or not isinstance(target, str):
            return False
        source, target = normalize(source), normalize(target)
        return (source, target) in pairs and not any(src == source and dst != target for src, dst in pairs)
    matched: dict[str, Any] = {}
    for task, item in task_samples.items():
        splits = {}
        for name in ("train", "test"):
            split = item.get(name, {})
            splits[name] = split if matches(split) else None
        caption = item.get("caption_reference")
        matched[task] = {**splits, "caption_reference": caption if matches(caption) else None}
        if not all(splits.values()):
            warnings.append(f"sample manifest splits for {task} do not all match effective paths; unmatched counts are N/A")
    return matched


def _matrix_payload(path: Path, tasks: list[str], warnings: list[str],
                    expected_source: Path | None = None) -> list[list[float]] | None:
    payload = _json_file(path, warnings)
    if payload is None:
        return None
    try:
        if expected_source is not None:
            recorded_source = payload.get("source") if isinstance(payload, dict) else None
            if not isinstance(recorded_source, str) or not Path(recorded_source).is_absolute() or Path(recorded_source).resolve() != expected_source:
                raise ValueError("metrics source does not match the explicitly recorded result directory")
        names = payload.get("tasks") if isinstance(payload, dict) else None
        matrix = payload.get("matrix") if isinstance(payload, dict) else payload
        if names and names != tasks:
            raise ValueError("task order differs from benchmark")
        if not isinstance(matrix, list) or len(matrix) != len(tasks):
            raise ValueError("matrix does not contain every benchmark stage")
        rows: list[list[float]] = []
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != i + 1:
                raise ValueError("matrix is not a complete lower triangle")
            values = [float(value) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("matrix contains non-finite values")
            rows.append(values)
        return rows
    except (TypeError, ValueError) as exc:
        warnings.append(f"invalid matrix in {path}: {exc}")
        return None


def _matrix(root: Path | None, metrics_path: Path | None, tasks: list[str], status: str,
            warnings: list[str], expected_metrics_source: Path | None = None
            ) -> tuple[list[list[float | None]], list[list[int | None]], list[list[Any]], bool]:
    matrix: list[list[float | None]] = [[None] * (i + 1) for i in range(len(tasks))]
    samples: list[list[int | None]] = [[None] * (i + 1) for i in range(len(tasks))]
    sources: list[list[Any]] = [[None] * (i + 1) for i in range(len(tasks))]
    conflicting = False
    seen_results = False
    if root and root.is_dir():
        for row in range(len(tasks)):
            for column in range(row + 1):
                files = sorted((root / "each_dataset" / tasks[column]).glob(f"*-task{row + 1}/Result.text"))
                if not files:
                    continue
                seen_results = True
                if status in _NO_RESULTS:
                    continue
                cell_values: list[tuple[float, int | None, str]] = []
                bad = False
                for result in files:
                    try:
                        if result.stat().st_size > _MAX_METADATA:
                            raise ValueError("result exceeds metadata size limit")
                        before = result.read_bytes()
                        value = _parse_result_text(result)
                        after = result.read_bytes()
                        if before != after:
                            raise ValueError("result changed while it was read")
                        if not math.isfinite(value):
                            raise ValueError("non-finite result")
                        text = before.decode("utf-8")
                        match = re.search(r"(?m)^Samples:\s*(\d+)\s*$", text)
                        cell_values.append((value, int(match.group(1)) if match else None, str(result)))
                    except (OSError, ValueError, TypeError, SyntaxError, UnicodeError) as exc:
                        warnings.append(f"invalid result {result}: {exc}")
                        bad = True
                if bad or len({(x[0], x[1]) for x in cell_values}) > 1:
                    conflicting = True
                    warnings.append(f"ambiguous matrix cell stage {row + 1}, task {tasks[column]}")
                elif cell_values:
                    matrix[row][column], samples[row][column] = cell_values[0][:2]
                    sources[row][column] = [x[2] for x in cell_values]
    matrix_files = ([root / "matrix.json", root / "accuracy_matrix.json"] if root else []) + ([metrics_path] if metrics_path else [])
    existing = [path for path in matrix_files if path.is_file()]
    if status in _NO_RESULTS and (seen_results or existing):
        warnings.append(f"{status} job has result files; suppressed because they may be stale")
        return matrix, samples, sources, True
    for path in existing:
        observed = _matrix_payload(path, tasks, warnings, expected_metrics_source if path == metrics_path else None)
        if observed is None:
            conflicting = True
            continue
        for row, values in enumerate(observed):
            for column, value in enumerate(values):
                if matrix[row][column] is not None and matrix[row][column] != value:
                    warnings.append(f"matrix disagrees with result at stage {row + 1}, task {tasks[column]}: {path}")
                    matrix[row][column] = None
                    sources[row][column] = None
                    conflicting = True
                elif matrix[row][column] is None and not conflicting:
                    matrix[row][column] = value
                    sources[row][column] = [str(path)]
    return matrix, samples, sources, conflicting


def _argument(argv: list[Any], flag: str) -> str | None:
    """Read one explicit flag; duplicate or missing values remain unknown."""
    values = []
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 >= len(argv) or str(argv[index + 1]).startswith("--"):
                return None
            values.append(str(argv[index + 1]))
        elif isinstance(item, str) and item.startswith(flag + "="):
            values.append(item[len(flag) + 1:])
    return values[0] if len(values) == 1 else None


def _positive_int(value: Any, *, zero: bool = False) -> int | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d+", value):
        return None
    number = int(value)
    return number if number >= (0 if zero else 1) else None


def _world_size(argv: list[Any]) -> int | None:
    # Explicit DeepSpeed host/slot include is evidence; visible GPUs alone are not.
    include = _argument(argv, "--include")
    if include is None:
        return None
    slots: set[tuple[str, str]] = set()
    for host_slots in include.split("@"):
        host, separator, raw_slots = host_slots.partition(":")
        if not separator or not host or not re.fullmatch(r"\d+(?:,\d+)*", raw_slots):
            return None
        for slot in raw_slots.split(","):
            key = (host, slot)
            if key in slots:
                return None
            slots.add(key)
    return len(slots) or None


def _command_resources(payload: Any, warnings: list[str]) -> dict[str, Any]:
    """Summarize recorded command plans without assuming they were executed."""
    result: dict[str, Any] = {"commands_execution_profile": None, "training": None,
                              "evaluation": None, "gradient_preparation": None,
                              "note": "Values are read or calculated from recorded commands, not proof of execution."}
    if not isinstance(payload, dict) or not isinstance(payload.get("commands"), list):
        return result
    result.update(commands_execution_profile=payload.get("execution_profile"), training=[],
                  evaluation=[], gradient_preparation=[])
    for index, command in enumerate(payload["commands"]):
        if not isinstance(command, dict) or not isinstance(command.get("argv"), list):
            continue
        argv = command["argv"]
        metadata = command.get("metadata") if isinstance(command.get("metadata"), dict) else {}
        env = command.get("env") if isinstance(command.get("env"), dict) else {}
        common = {"command_index": index, "task": command.get("task_id"), "sequence": command.get("sequence"),
                  "execution_profile": metadata.get("execution_profile")}
        gradient = any(isinstance(item, str) and (item.endswith(".extract_gradients") or item.endswith("/extract_gradients.py"))
                       for item in argv)
        if gradient:
            result["gradient_preparation"].append({**common,
                "num_chunks": _positive_int(_argument(argv, "--num-chunks")),
                "chunk_index": _positive_int(_argument(argv, "--chunk-idx"), zero=True),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "merge_only": "--merge-only" in argv,
                "recorded_shards": metadata.get("gradient_preparation_shards")})
        elif command.get("kind") == "train":
            world = _world_size(argv)
            batch = _positive_int(_argument(argv, "--per_device_train_batch_size"))
            accumulation = _positive_int(_argument(argv, "--gradient_accumulation_steps"))
            effective = world * batch * accumulation if all(x is not None for x in (world, batch, accumulation)) else None
            recorded = metadata.get("effective_batch")
            if effective is not None and recorded is not None and recorded != effective:
                warnings.append(f"training command {index} effective batch disagrees with metadata")
                effective = None
            result["training"].append({**common, "world_size": world, "include": _argument(argv, "--include"),
                "per_device_batch_size": batch, "gradient_accumulation_steps": accumulation,
                "effective_batch_size": effective, "recorded_effective_batch": recorded})
        elif command.get("kind") == "eval" and any(isinstance(item, str) and
                (item in {"--num-chunks", "--chunk-idx"} or item.startswith(("--num-chunks=", "--chunk-idx="))) for item in argv):
            result["evaluation"].append({**common, "num_chunks": _positive_int(_argument(argv, "--num-chunks")),
                "chunk_index": _positive_int(_argument(argv, "--chunk-idx"), zero=True),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "parallel_group": metadata.get("parallel_group")})
    return result


def _lineage(state: dict[str, Any], run_dir: Path, warnings: list[str]) -> dict[str, Any]:
    previous = state.get("previous_run")
    receipt = _resolve(state.get("transition_receipt"), run_dir)
    result = {"previous_run": previous, "related_runs": state.get("related_runs"),
              "transition_receipt": _file_info(receipt), "receipt_summary": None,
              "note": "Related runs are linked only; statuses, matrices and metrics are not combined."}
    if receipt:
        payload = _json_file(receipt, warnings)
        if not receipt.is_file():
            warnings.append(f"transition receipt is unavailable: {receipt}")
        elif isinstance(payload, dict):
            keys = ("schema_version", "status", "runtime_project", "previous_run", "old_run_id", "new_run_id",
                    "execution_profile", "gpus", "jobs", "manifest", "manifest_sha256", "applied_at")
            result["receipt_summary"] = {key: payload.get(key) for key in keys}
            if payload.get("new_run_id") and payload["new_run_id"] != state.get("run_id"):
                warnings.append("transition receipt new_run_id differs from state run_id")
            if previous and payload.get("previous_run") and payload["previous_run"] != previous:
                warnings.append("transition receipt previous_run differs from state previous_run")
    return result


def _job(key: str, raw: dict[str, Any], state: dict[str, Any], run_dir: Path,
         snapshot: datetime, samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    pieces = key.split("/", 1)
    method = raw.get("method") or pieces[0]
    benchmark = raw.get("benchmark") or (pieces[1] if len(pieces) > 1 else "")
    if not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", x) and x not in {".", ".."}
               for x in (method, benchmark)):
        raise ValueError(f"invalid method or benchmark in job {key!r}")
    directory = run_dir / "jobs" / method / benchmark
    status = raw.get("status") or "unknown"
    try:
        tasks = list(get_benchmark(benchmark).tasks)
    except KeyError:
        tasks = []
        warnings.append(f"unknown benchmark {benchmark}; task matrix unavailable")
    event, event_count = _events(directory / "events.jsonl", warnings)
    step = raw.get("step") if isinstance(raw.get("step"), dict) else None
    if event and step and any(event.get(name) != step.get(name) for name in ("index", "status", "task")):
        warnings.append("last event differs from the state step; queue may have advanced during the snapshot")
    if event and status in _NO_RESULTS:
        warnings.append(f"{status} job has events; they may be stale")
    if status == "completed" and event and event.get("status") in {"failed", "blocked_disk"}:
        warnings.append("completed job has a failing last event; aggregate metrics suppressed")
    if status == "running" and state.get("current") != key:
        warnings.append("job is running but is not the state's current job")
    failures: list[dict[str, Any]] = []
    python_check = None
    for sequence in ("train", "router"):
        doctor_path = directory / f"doctor-{sequence}.json"
        doctor = _json_file(doctor_path, warnings)
        checks = doctor.get("checks", []) if isinstance(doctor, dict) else []
        if isinstance(checks, list):
            failures.extend({**c, "doctor_report": str(doctor_path)} for c in checks if isinstance(c, dict) and c.get("status") == "FAIL")
            if sequence == "train":
                python_check = next((c.get("detail") for c in checks if isinstance(c, dict) and c.get("name") == "python"), None)
    if failures and status == "completed":
        warnings.append("completed job has failed doctor checks; aggregate metrics suppressed")
    if failures and status in {"pending", "not_run"}:
        warnings.append(f"{status} job has failed doctor checks; they may be stale")
    commands = _json_file(directory / "commands.json", warnings)
    resources = _command_resources(commands, warnings)
    if (state.get("execution_profile") and resources["commands_execution_profile"] and
            state["execution_profile"] != resources["commands_execution_profile"]):
        warnings.append("recorded command execution profile differs from state execution profile")
    command_list = commands.get("commands", []) if isinstance(commands, dict) else []
    if not isinstance(command_list, list):
        command_list = []
    sample_expectation = {
        "state_expected_eval_samples": raw.get("expected_eval_samples"),
        "commands_expected_eval_samples": commands.get("expected_eval_samples") if isinstance(commands, dict) else None,
        "state_expected_eval_hashes": raw.get("expected_eval_hashes"),
        "commands_expected_eval_hashes": commands.get("expected_eval_hashes") if isinstance(commands, dict) else None,
    }
    sample_conflict = False
    for suffix in ("eval_samples", "eval_hashes"):
        state_value = sample_expectation[f"state_expected_{suffix}"]
        commands_value = sample_expectation[f"commands_expected_{suffix}"]
        if state_value is not None and commands_value is not None and state_value != commands_value:
            warnings.append(f"expected {suffix} differs between state and commands; aggregate metrics suppressed")
            sample_conflict = True
    roots = state.get("roots") if isinstance(state.get("roots"), dict) else {}
    results_root = _resolve(roots.get("results_root"))
    backbone_dir = Path("internvl") if state.get("backbone") == "internvl" else Path()
    expert_results = (results_root / method / benchmark / backbone_dir / method / _RESULT_DIRS[benchmark]
                      if results_root and benchmark in _RESULT_DIRS else None)
    result_dir = expert_results
    metrics_file: Path | None = directory / "metrics.json"
    expected_metrics_source: Path | None = None
    result_source = "legacy"
    if method == "mrlora" and benchmark in {"mllm_dcl", "ucit"}:
        state_dir = raw.get("metrics_result_dir")
        commands_dir = commands.get("metrics_result_dir") if isinstance(commands, dict) else None
        declarations = [value for value in (state_dir, commands_dir) if value is not None]
        if declarations:
            folder = "UCIT_router" if benchmark == "ucit" else "DCL_router"
            expected = (results_root / method / benchmark / backbone_dir / method / folder).resolve() if results_root else None
            valid = (expected is not None and all(isinstance(value, str) and Path(value).is_absolute()
                     and Path(value).resolve() == expected for value in declarations))
            if valid:
                result_dir = expected
                expected_metrics_source = expected
                result_source = "router"
            else:
                warnings.append(f"invalid or conflicting MR-LoRA {benchmark} metrics_result_dir; results are unavailable")
                result_dir = None
                metrics_file = None
                result_source = "invalid"
    matrix, observed_samples, cell_sources, conflict = _matrix(
        result_dir, metrics_file, tasks, status, warnings, expected_metrics_source)
    current_task = raw.get("task")
    if status in {"running", "failed", "stopped"} and type(current_task) is int and current_task >= 0:
        for row in range(current_task, len(matrix)):
            if any(value is not None for value in matrix[row]):
                warnings.append(f"stage {row + 1} results are newer than the state task {current_task}; suppressed in this snapshot")
                matrix[row] = [None] * len(matrix[row])
                observed_samples[row] = [None] * len(observed_samples[row])
                cell_sources[row] = [None] * len(cell_sources[row])
    complete = bool(tasks) and all(value is not None for row in matrix for value in row)
    if status == "completed" and not complete:
        warnings.append("completed status but the benchmark matrix is incomplete")
    legacy_ucit = method == "mrlora" and benchmark == "ucit" and result_source == "legacy"
    if legacy_ucit:
        warnings.append("legacy MR-LoRA UCIT scores do not establish routed evaluation; aggregate metrics suppressed")
    can_compute = status == "completed" and complete and not conflict and not sample_conflict and not failures and not legacy_ucit and not (event and event.get("status") in {"failed", "blocked_disk"})
    metrics = continual_metrics(matrix) if can_compute else {name: None for name in _METRICS}
    logs = dict(raw.get("logs")) if isinstance(raw.get("logs"), dict) else {}
    for name, filename in (("events", "events.jsonl"), ("commands", "commands.json"), ("training", "training.log"),
                           ("evaluation", "evaluation.log"), ("worker", "worker.stdout.log"),
                           ("worker_stderr", "worker.stderr.log"), ("doctor_train", "doctor-train.json"),
                           ("doctor_router", "doctor-router.json"), ("metrics_json", "metrics.json")):
        if name not in logs and (directory / filename).is_file():
            logs[name] = str(directory / filename)
    failure_log = raw.get("failure_log")
    error = raw.get("error")
    if not error and status in {"failed", "blocked", "blocked_disk", "blocked_hardware", "stopped"}:
        error = (event or {}).get("error")
        if not error and failures:
            error = "; ".join(f"{c.get('name')}: {c.get('detail')}" for c in failures[:4])
    if not failure_log and status in {"failed", "blocked", "blocked_disk", "blocked_hardware"}:
        failure_log = (event or {}).get("log")
        if not failure_log and failures:
            failure_log = failures[0]["doctor_report"]
    end = raw.get("finished_at") or (snapshot.isoformat() if status == "running" else None)
    effective = _resolve(raw.get("effective_paths"), run_dir) or directory / "paths.effective.yaml"
    task_samples = samples.get(benchmark, {})
    planned_samples = _configured_samples(effective, _resolve(state.get("project")), benchmark, task_samples, warnings)
    return {
        "key": key, "method": method, "benchmark": benchmark, "status": status,
        "phase": raw.get("phase"), "task": raw.get("task"), "step": step,
        "latest_event": event, "event_count": event_count,
        "started_at": raw.get("started_at"), "finished_at": raw.get("finished_at"),
        "elapsed_seconds": _elapsed(raw.get("started_at"), end),
        "elapsed_is_open": status == "running" and bool(raw.get("started_at")),
        "error": error, "failure_log": failure_log,
        "error_tail": raw.get("error_tail") or (_tail(_resolve(failure_log, directory)) if failure_log else None),
        "logs": logs, "doctor_failures": failures,
        "tasks": tasks, "matrix": matrix, "matrix_complete": complete, "matrix_sources": cell_sources,
        "observed_samples": observed_samples, "metrics": metrics,
        "planned_samples": {task: planned_samples.get(task) for task in tasks},
        "results_dir": str(result_dir) if result_dir else None, "result_source": result_source,
        "expert_results_dir": str(expert_results) if expert_results and result_source == "router" else None,
        "environment": {"name": raw.get("environment"), "python": raw.get("python"), "python_version_from_doctor": python_check},
        "resources": resources,
        "provenance": {"effective_paths": _file_info(effective),
                       "commands": _file_info(directory / "commands.json"), "command_count": len(command_list) if commands is not None else None,
                       "doctor_train": _file_info(directory / "doctor-train.json"),
                       "doctor_router": _file_info(directory / "doctor-router.json"),
                       "metrics_json": _file_info(directory / "metrics.json"),
                       "sample_expectation": sample_expectation,
                       "result_selection": {"kind": result_source, "results_dir": str(result_dir) if result_dir else None,
                                            "state_metrics_result_dir": raw.get("metrics_result_dir"),
                                            "commands_metrics_result_dir": commands.get("metrics_result_dir") if isinstance(commands, dict) else None}},
        "warnings": warnings,
    }


def build_report(run_dir: str | Path) -> dict[str, Any]:
    """Read the queue into a JSON-serializable, point-in-time observation."""
    run_dir = Path(run_dir).expanduser().resolve()
    state_path = run_dir / "state.json"
    snapshot = _now()
    try:
        state_bytes = state_path.read_bytes()
        state = json.loads(state_bytes)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"cannot read state {state_path}: {exc}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("jobs"), dict):
        raise ValueError(f"state must be an object with a jobs mapping: {state_path}")
    warnings: list[str] = []
    project = _resolve(state.get("project"))
    source = _resolve(state.get("source_paths"), project)
    auxiliary = _resolve(state.get("auxiliary_paths"), project)
    sample_info, sample_counts = _sample_info(state, project, source, warnings)
    aux_manifest = auxiliary.parent / "manifest.json" if auxiliary else None
    aux = _json_file(aux_manifest, warnings)
    if isinstance(aux, dict):
        sample_info["auxiliary_manifest"] = _file_info(aux_manifest)
        sample_info["auxiliary_recorded_sample_manifest_sha256"] = aux.get("sample_manifest_sha256")
        observed_hash = (sample_info.get("manifest") or {}).get("sha256")
        if observed_hash and aux.get("sample_manifest_sha256"):
            sample_info["auxiliary_sample_hash_matches"] = observed_hash == aux["sample_manifest_sha256"]
            if not sample_info["auxiliary_sample_hash_matches"]:
                warnings.append("auxiliary manifest fingerprint differs from the observed sample manifest")
    jobs = []
    for key, raw in state["jobs"].items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            raise ValueError("state jobs must map string keys to objects")
        jobs.append(_job(key, raw, state, run_dir, snapshot, sample_counts))
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    if isinstance(state.get("summary"), dict) and state["summary"] != counts:
        warnings.append("state summary does not match the job statuses; report uses job statuses")
    current = state.get("current")
    if current and current not in state["jobs"]:
        warnings.append("state current job is missing from the jobs mapping")
    elif current and state["jobs"][current].get("status") != "running":
        warnings.append("state current job is not marked running")
    if state.get("status") == "completed" and any(job["status"] != "completed" for job in jobs):
        warnings.append("completed queue contains jobs not marked completed")
    lineage = _lineage(state, run_dir, warnings)
    state_hash = hashlib.sha256(state_bytes).hexdigest()
    manifest = {
        "run_id": state.get("run_id"), "snapshot_at": snapshot.isoformat(),
        "state_file": {"path": str(state_path), "sha256_at_start": state_hash},
        "code": {"experiment": _git_info(project),
                 "reporter": {"checkout": str(Path(__file__).resolve().parent.parent),
                              "python_executable": sys.executable, "python_version": sys.version,
                              "reporting_source": _file_info(Path(__file__).resolve()),
                              "metrics_source": _file_info(Path(__file__).resolve().parent / "eval" / "metrics.py")},
                 "qualification": "Selected file hashes and HEAD are observed at collection time; they are neither a full code inventory nor launch-time frozen credentials."},
        "config": {"source_paths": _file_info(source), "auxiliary_paths": _file_info(auxiliary),
                   "validation_fixes": state.get("validation_fixes"), "roots": state.get("roots")},
        "environment": {"gpus_recorded": state.get("gpus"), "master_port_recorded": state.get("master_port"),
                        "execution_profile": state.get("execution_profile"),
                        "jobs": {job["key"]: job["environment"] for job in jobs}},
        "lineage": lineage,
        "sample": sample_info, "sample_counts": sample_counts,
        "jobs": {job["key"]: {**job["provenance"], "resources": job["resources"]} for job in jobs},
        "note": "The source profile identifies a sample manifest; each job's planned counts additionally require matching source-to-output pairs in its applicable observed effective paths. These files are not launch-time frozen credentials, and mappings do not prove consumption. Recorded sample hashes are not revalidated against data files. No model weights are scanned or hashed.",
    }
    try:
        changed = hashlib.sha256(state_path.read_bytes()).hexdigest() != state_hash
    except OSError:
        changed = None
    manifest["state_file"]["changed_during_collection"] = changed
    if changed:
        warnings.append("state.json changed during collection; jobs use the state read at the snapshot start")
    if changed is None:
        warnings.append("could not re-read state.json after collection")
    return {
        "schema_version": 1, "snapshot_at": snapshot.isoformat(), "collected_at": _now().isoformat(),
        "run_dir": str(run_dir), "run_id": state.get("run_id"), "status": state.get("status"), "current": current,
        **({"backbone": "internvl", "source_binding": {"path": state.get("source_binding"),
                                                       "sha256": state.get("source_binding_sha256")}}
           if state.get("backbone") == "internvl" else {}),
        "execution_profile": state.get("execution_profile"), "lineage": lineage,
        "started_at": state.get("started_at"), "updated_at": state.get("updated_at"), "finished_at": state.get("finished_at"),
        "summary": counts, "recorded_summary": state.get("summary"),
        "snapshot_note": "Live files are read sequentially, not atomically; the queue can continue while this snapshot is collected.",
        "warnings": warnings, "jobs": jobs,
        "manifest": manifest,
    }


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _md(value: Any) -> str:
    return _display(value).replace("|", "\\|").replace("\n", " ")


def _unique_values(items: list[dict[str, Any]] | None, key: str) -> str:
    if not items:
        return "N/A"
    return ", ".join(sorted({_display(item.get(key)) for item in items}))


def _allocations(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return "N/A"
    return "; ".join(sorted({f"{_display(item.get('num_chunks'))} chunks, index {_display(item.get('chunk_index'))}: "
                             f"CUDA_VISIBLE_DEVICES={_display(item.get('cuda_visible_devices'))}" for item in items}))


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# Validation queue report: {_md(report['run_id'])}", "",
             f"Snapshot (UTC): {report['snapshot_at']}  ", f"Collection finished (UTC): {report['collected_at']}  ",
             f"Run directory: `{report['run_dir']}`  ", f"Queue status: {_md(report['status'])}  ",
             f"Execution profile in state: {_md(report.get('execution_profile'))}  ",
             f"Current job: {_md(report['current'])}", "", report["snapshot_note"], "",
             f"Previous run: {_md(report['lineage'].get('previous_run'))}  ",
             f"Transition receipt: {_md((report['lineage'].get('transition_receipt') or {}).get('path'))}",
             "Related runs are linked only; their job states and metrics are not merged.", "",
             "This report may be incomplete. N/A means unavailable, not zero. Aggregate continual metrics are shown only for a completed job with a complete, consistent benchmark matrix.", "",
             "Sample validation does not establish formal reproduction. Planned sample counts come from existing metadata; observed counts come from result files.", "",
             "## Jobs", "", "| Method | Benchmark | Status | Phase | Task | Elapsed (s) | MFT | MFN | MAA | BWT | Failure |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    if report.get("backbone") == "internvl":
        source = report.get("source_binding") or {}
        lines[2:2] = ["Backbone: InternVL  ", f"Runtime source binding: `{_md(source.get('path'))}`  ",
                      f"Binding SHA-256: `{_md(source.get('sha256'))}`  ", ""]
    for job in report["jobs"]:
        row = [job["method"], job["benchmark"], job["status"], job["phase"], job["task"], job["elapsed_seconds"],
               *(job["metrics"][name] for name in _METRICS), job["error"]]
        lines.append("| " + " | ".join(_md(item) for item in row) + " |")
    if report.get("backbone") == "internvl":
        lines += ["", "## Observed result cells", "", "| Method | Benchmark | Available / required |",
                  "| --- | --- | ---: |"]
        for job in report["jobs"]:
            count = sum(value is not None for row in job["matrix"] for value in row)
            required = sum(len(row) for row in job["matrix"])
            lines.append(f"| {_md(job['method'])} | {_md(job['benchmark'])} | {count} / {required} |")
    if report["warnings"]:
        lines += ["", "## Snapshot warnings", ""] + [f"- {_md(warning)}" for warning in report["warnings"]]
    samples = report["manifest"]["sample_counts"]
    lines += ["", "## Planned sample counts", "", "These are dataset manifest counts, not proof that a job consumed or evaluated them.", "",
              "| Benchmark | Task | Train | Test | Caption images | Caption references |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for benchmark, tasks in samples.items():
        for task, item in tasks.items():
            caption = item.get("caption_reference") or {}
            lines.append("| " + " | ".join(_md(v) for v in (benchmark, task, item["train"]["sample_count"], item["test"]["sample_count"], caption.get("images"), caption.get("annotations"))) + " |")
    if not samples:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "## Job matrices and evidence", "", "Rows are training stages; columns are evaluation tasks. A dash marks cells outside the lower triangle. Numbers are reported result scores; N/A is unavailable."]
    for job in report["jobs"]:
        lines += ["", f"### {_md(job['key'])} — {_md(job['status'])}", ""]
        if job["result_source"] == "router":
            lines += [f"Matrix source: explicitly recorded router results at `{_md(job['results_dir'])}`. "
                      f"Expert cross-results remain separate at `{_md(job['expert_results_dir'])}`. "
                      f"The fixed {'six' if job['benchmark'] == 'ucit' else 'five'}-expert routing pool is retrospective "
                      "and is not a strictly online protocol.", ""]
        elif job["result_source"] == "invalid":
            lines += ["Matrix source: N/A; the explicitly recorded result directory is invalid or inconsistent.", ""]
        elif job["method"] == "mrlora" and job["benchmark"] == "mllm_dcl":
            lines += ["Matrix source: legacy DCL results; this older record does not identify a separate router directory.", ""]
        elif job["method"] == "mrlora" and job["benchmark"] == "ucit":
            lines += ["Matrix source: legacy expert results; routed UCIT scores are not established by this record.", ""]
        resources = job["resources"]
        training = resources["training"]
        lines += ["Recorded command resources (plans, not proof of execution):",
                  "- Training by command: " + ("; ".join(
                      f"task {_md(item.get('task'))}, command {_md(item.get('command_index'))}: "
                      f"world size {_md(item.get('world_size'))}, per-device batch {_md(item.get('per_device_batch_size'))}, "
                      f"gradient accumulation {_md(item.get('gradient_accumulation_steps'))}, "
                      f"effective batch {_md(item.get('effective_batch_size'))}" for item in training) if training else "N/A") + ".",
                  f"- Evaluation allocation: {_md(_allocations(resources['evaluation']))}.",
                  f"- Gradient preparation allocation: {_md(_allocations(resources['gradient_preparation']))}.", ""]
        if job["tasks"]:
            lines += ["| Stage | " + " | ".join(_md(t) for t in job["tasks"]) + " |",
                      "| --- | " + " | ".join("---:" for _ in job["tasks"]) + " |"]
            for index, row in enumerate(job["matrix"]):
                lines.append("| " + str(index + 1) + " | " + " | ".join([*(_md(v) for v in row), *(["—"] * (len(job["tasks"]) - len(row)))]) + " |")
        else:
            lines.append("Matrix: N/A")
        observed = [(str(i + 1), job["tasks"][j], count) for i, row in enumerate(job["observed_samples"]) for j, count in enumerate(row) if count is not None]
        if observed:
            lines += ["", "Observed Samples fields: " + "; ".join(f"stage {i}, {task}: {count}" for i, task, count in observed)]
        if job["failure_log"]:
            lines += ["", f"Failure log: `{job['failure_log']}`"]
        if job["doctor_failures"]:
            lines += ["", "Doctor failures:"] + [f"- {_md(item.get('name'))}: {_md(item.get('detail'))} (`{item['doctor_report']}`)" for item in job["doctor_failures"]]
        if job["error_tail"]:
            lines += ["", "Failure log excerpt:", "", "```text", job["error_tail"].replace("```", "` ` `"), "```"]
        if job["logs"]:
            lines += ["", "Logs: " + "; ".join(f"{_md(key)}: `{value}`" for key, value in job["logs"].items())]
        if job["warnings"]:
            lines += [""] + [f"- {_md(warning)}" for warning in job["warnings"]]
    lines += ["", "## Provenance", "", "See `manifest.json` for recorded code, environment, configuration, sample metadata, and fingerprints. Code fingerprints describe files observed at snapshot time; they do not prove the code used at run start.", ""]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, two CSVs, Markdown and a provenance manifest into a separate directory."""
    output = Path(output_dir).expanduser().resolve()
    protected = [Path(report["run_dir"]).resolve()]
    roots = report["manifest"]["config"].get("roots")
    if isinstance(roots, dict):
        protected.extend(path for path in (_resolve(roots.get(name)) for name in ("logs_root", "results_root", "ckpt_root")) if path)
    if any(output == root or root in output.parents for root in protected):
        raise ValueError("output directory must be outside the run, results, checkpoint and logs roots")
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: str(output / filename) for name, filename in
             (("json", "report.json"), ("csv", "report.csv"), ("cells_csv", "cells.csv"),
              ("markdown", "report.md"), ("manifest", "manifest.json"))}
    # Replace directory entries atomically, without following a pre-existing output symlink.
    def write(name: str, content: str) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output,
                                             prefix=".report-", delete=False) as handle:
                temporary = handle.name
                handle.write(content)
            os.replace(temporary, output / name)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    write("report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write("manifest.json", json.dumps(report["manifest"], indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write("report.md", _markdown(report))
    columns = ["method", "benchmark", "status", "phase", "task", "step_kind", "started_at", "finished_at",
               "elapsed_seconds", "matrix_complete", "available_cells", "required_cells", *_METRICS,
               "result_source", "results_dir", "expert_results_dir",
               "execution_profile", "train_world_size", "per_device_batch_size", "gradient_accumulation_steps",
               "effective_batch_size", "eval_num_chunks", "eval_allocations", "gradient_preparation_num_chunks",
               "error", "failure_log", "training_log", "evaluation_log", "worker_log"]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    for job in report["jobs"]:
        row = {key: job.get(key) for key in columns}
        row.update(job["metrics"])
        resources = job["resources"]
        training = resources["training"]
        row.update(step_kind=(job["step"] or {}).get("kind"),
                   available_cells=sum(value is not None for values in job["matrix"] for value in values),
                   required_cells=sum(len(values) for values in job["matrix"]),
                   execution_profile=report.get("execution_profile"),
                   train_world_size=_unique_values(training, "world_size"),
                   per_device_batch_size=_unique_values(training, "per_device_batch_size"),
                   gradient_accumulation_steps=_unique_values(training, "gradient_accumulation_steps"),
                   effective_batch_size=_unique_values(training, "effective_batch_size"),
                   eval_num_chunks=_unique_values(resources["evaluation"], "num_chunks"),
                   eval_allocations=_allocations(resources["evaluation"]),
                   gradient_preparation_num_chunks=_unique_values(resources["gradient_preparation"], "num_chunks"),
                   training_log=job["logs"].get("training"), evaluation_log=job["logs"].get("evaluation"),
                   worker_log=job["logs"].get("worker"))
        writer.writerow({key: "N/A" if value is None else value for key, value in row.items()})
    write("report.csv", handle.getvalue())
    cell_columns = ["method", "benchmark", "stage", "eval_task", "score", "samples", "availability"]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=cell_columns)
    writer.writeheader()
    for job in report["jobs"]:
        for i, values in enumerate(job["matrix"]):
            for j, value in enumerate(values):
                available = "observed" if value is not None else ("suppressed_by_status" if job["status"] in _NO_RESULTS else "unavailable")
                count = job["observed_samples"][i][j]
                writer.writerow({"method": job["method"], "benchmark": job["benchmark"], "stage": i + 1,
                                 "eval_task": job["tasks"][j], "score": value if value is not None else "N/A",
                                 "samples": count if count is not None else "N/A", "availability": available})
    write("cells.csv", handle.getvalue())
    return paths


def generate_report(run_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Build and write one report; return absolute artifact paths."""
    return write_report(build_report(run_dir), output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="queue directory containing state.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="separate output directory")
    args = parser.parse_args(argv)
    try:
        outputs = generate_report(args.run_dir, args.output_dir)
    except (ValueError, OSError) as exc:
        print(f"mcitlib.reporting: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
