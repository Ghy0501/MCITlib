# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Serial, failure-isolated queue for the UCIT/MLLM-DCL validation samples."""
from __future__ import annotations

import argparse
import copy
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .gpus import parse_gpus
from .paths import PathResolver, managed_path, process_env, repo_root
from .registry import METHODS, benchmark_supported, compatibility, get_benchmark, get_method

BENCHMARKS = ("ucit", "mllm_dcl")
ROOT_VARIABLES = ("MCITLIB_MODEL_ROOT", "MCITLIB_DATA_ROOT", "MCITLIB_CKPT_ROOT", "MCITLIB_RESULTS_ROOT",
                  "MCITLIB_LOGS_ROOT")
FIXES = ["modalprompt_ucit_task2_output_dir", "modalprompt_load_previous_task",
         "mrlora_dcl_expert_inference", "mrlora_dcl_router_selection",
         "mrlora_dcl_router_scoring", "keeplora_dcl_stage2_med", "mrlora_ucit_expert_inference",
         "mrlora_ucit_router_inference", "mrlora_ucit_router_selection", "mrlora_ucit_router_scoring"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def tail(path: str | Path | None, limit: int = 12000) -> str:
    if not path:
        return ""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def free_space_gib(roots: dict[str, str]) -> dict[str, float]:
    available = {}
    for root in ("ckpt_root", "results_root", "logs_root"):
        existing = Path(roots[root])
        while not existing.exists():
            existing = existing.parent
        available[root] = shutil.disk_usage(existing).free / 2**30
    return available


def disk_shortage(roots: dict[str, str], minimum: float) -> str | None:
    insufficient = {root: free for root, free in free_space_gib(roots).items() if free < minimum}
    if not insufficient:
        return None
    details = ", ".join(f"{root}: {free:.2f} GiB" for root, free in insufficient.items())
    return f"insufficient free disk space: {details} (minimum {minimum:.2f} GiB)"


def doctor_report(path: Path) -> dict[str, Any]:
    """Find the structured doctor result even if a dependency printed startup logs."""
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    for index in (i for i in range(len(content) - 1, -1, -1) if content[i] == "{"):
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("checks"), list) and "failed" in value:
            return value
    raise ValueError("doctor did not emit a structured report")


def proc_info(pid: int) -> tuple[int, int, int] | None:
    """Return process group, session and kernel start ticks for a Linux process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return int(fields[2]), int(fields[3]), int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def group_members(session: int, since: int) -> list[tuple[int, int]]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        info = proc_info(pid)
        if info and info[0] == session and info[1] == session and info[2] >= since:
            members.append((pid, info[2]))
    return members


def signal_member(pid: int, ticks: int, group: int, sig: int) -> None:
    """Use a pidfd and identity checks, so a recycled PID is never signalled."""
    try:
        fd = pidfd_open(pid)
        try:
            info = proc_info(pid)
            if info and info == (group, group, ticks):
                pidfd_signal(fd, sig)
        finally:
            os.close(fd)
    except (OSError, ProcessLookupError):
        pass


def _pidfd_syscall(number: int, *args: int) -> int:
    # Some conda Python 3.10 builds omit the os/signal pidfd wrappers even on
    # recent kernels. Both supported Linux architectures use these syscall IDs.
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise OSError("pidfd operations require Linux on x86_64 or aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(value) for value in args))
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def pidfd_open(pid: int) -> int:
    return os.pidfd_open(pid) if hasattr(os, "pidfd_open") else _pidfd_syscall(434, pid, 0)


def pidfd_signal(fd: int, sig: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, sig)
    else:
        _pidfd_syscall(424, fd, sig, 0, 0)


def remember_group(group: int, ticks: int, known: dict[int, int]) -> None:
    """Record members only while the original session leader still exists."""
    identity = (group, group, ticks)
    if proc_info(group) != identity:
        return
    members = group_members(group, ticks)
    if proc_info(group) == identity:
        known.update(members)


def live_known(group: int, known: dict[int, int]) -> list[tuple[int, int]]:
    return [(pid, ticks) for pid, ticks in known.items() if proc_info(pid) == (group, group, ticks)]


def cleanup_group(group: int, since: int, known: dict[int, int], grace: float = 5.0) -> None:
    """Stop only observed members of the session started for this child process."""
    remember_group(group, since, known)
    members = live_known(group, known)
    for pid, ticks in members:
        signal_member(pid, ticks, group, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while members and time.monotonic() < deadline:
        time.sleep(0.1)
        remember_group(group, since, known)
        members = live_known(group, known)
    for pid, ticks in members:
        signal_member(pid, ticks, group, signal.SIGKILL)


@dataclass
class Job:
    method: str
    benchmark: str
    env_name: str
    python: str | None
    directory: Path
    paths_file: Path
    profile: dict[str, Any]
    doctors: list[tuple[str, list[str]]]
    worker: list[str]
    metrics: list[str]
    metrics_result_dir: Path | None = None
    expected_eval_samples: int | None = None
    expected_eval_hashes: dict[str, str] | None = None

    @property
    def key(self) -> str:
        return f"{self.method}/{self.benchmark}"


def verified_dcl_sample_info(project: Path, resolver: PathResolver) -> dict[str, Any] | None:
    """Use the 3k/1k evidence only when all effective mappings and file hashes match."""
    manifest = project / "configs/validation_3k_1k/manifest.json"
    if not manifest.is_file():
        if any("/validation_3k_1k/" in target for scope in ("common", "mllm_dcl")
               for _, target in resolver.mappings.get(scope, [])):
            raise FileNotFoundError(f"DCL sample profile requires its manifest: {manifest}")
        return None
    manifest_bytes = manifest.read_bytes()
    payload = json.loads(manifest_bytes)
    records = {item["name"]: item["splits"]["test"] for item in payload["tasks"] if item["benchmark"] == "mllm_dcl"}
    names = ("RS", "Med", "AD", "Sci", "Fin")
    if set(records) != set(names):
        raise ValueError("validation sample manifest has incomplete DCL test coverage")
    matches = []
    for name in names:
        item = records[name]
        actual = Path(resolver.resolve_templates(item["source"], "mllm_dcl")).resolve()
        expected = Path(resolver.resolve_templates(item["output"])).resolve()
        matches.append((name, item, actual, actual == expected))
    if not any(match for _, _, _, match in matches):
        if any("/validation_3k_1k/" in target for scope in ("common", "mllm_dcl")
               for _, target in resolver.mappings.get(scope, [])):
            raise ValueError("DCL sample mappings do not match the validation manifest")
        return None
    if not all(match for _, _, _, match in matches):
        raise ValueError("validation sample profile maps only part of the DCL test manifest")
    counts = set()
    for name, item, path, _ in matches:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["output_sha256"]:
            raise ValueError(f"validation sample hash differs from manifest: {name}: {path}")
        data = json.loads(raw)
        count = item["sample_count"]
        if not isinstance(data, list) or type(count) is not int or count < 1 or len(data) != count:
            raise ValueError(f"validation sample count differs from manifest: {name}: {path}")
        counts.add(count)
    if len(counts) != 1:
        raise ValueError("DCL validation manifest has different per-domain test counts")
    return {"count": counts.pop(), "hashes": {name: item["output_sha256"] for name, item, _, _ in matches},
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def verified_dcl_sample_count(project: Path, resolver: PathResolver) -> int | None:
    info = verified_dcl_sample_info(project, resolver)
    return info["count"] if info else None


def verified_ucit_sample_info(project: Path, resolver: PathResolver) -> dict[str, Any] | None:
    """Bind UCIT test and caption inputs to their recorded sample manifest."""
    from .methods._mrlora.router_results.ucit_validation import EXPERTS, validate_caption_alignment

    manifest = project / "configs/validation_3k_1k/manifest.json"
    uses_sample = any("/validation_3k_1k/" in target for scope in ("common", "ucit")
                      for _, target in resolver.mappings.get(scope, []))
    if not manifest.is_file():
        if uses_sample:
            raise FileNotFoundError(f"UCIT sample profile requires its manifest: {manifest}")
        return None
    manifest_bytes = manifest.read_bytes()
    payload = json.loads(manifest_bytes)
    records = {item["name"]: item for item in payload["tasks"] if item["benchmark"] == "ucit"}
    if set(records) != set(EXPERTS):
        raise ValueError("validation sample manifest has incomplete UCIT coverage")
    mappings = []
    for domain in EXPERTS:
        item = records[domain]["splits"]["test"]
        expected = Path(resolver.resolve_templates(item["output"])).resolve()
        sources = [item["source"], *(alias["source"] for alias in item.get("source_aliases", []))]
        mappings.append((domain, item, expected, all(Path(resolver.resolve_templates(source, "ucit")).resolve() == expected
                                                   for source in sources)))
    if not any(match for _, _, _, match in mappings):
        if uses_sample:
            raise ValueError("UCIT sample mappings do not match the validation manifest")
        return None
    if not all(match for _, _, _, match in mappings):
        raise ValueError("validation sample profile maps only part of the UCIT test manifest")
    counts = set()
    hashes: dict[str, str] = {}
    for domain, item, path, _ in mappings:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["output_sha256"]:
            raise ValueError(f"UCIT validation sample hash differs from manifest: {domain}: {path}")
        annotations = json.loads(raw)
        count = item["sample_count"]
        if not isinstance(annotations, list) or type(count) is not int or count < 1 or len(annotations) != count:
            raise ValueError(f"UCIT validation sample count differs from manifest: {domain}: {path}")
        counts.add(count)
        hashes[domain] = item["output_sha256"]
        if domain in {"VizWiz", "Flickr30k"}:
            caption = records[domain].get("caption_reference")
            if not isinstance(caption, dict):
                raise ValueError(f"UCIT caption reference missing from manifest: {domain}")
            reference = Path(resolver.resolve_templates(caption["source"], "ucit")).resolve()
            expected = Path(resolver.resolve_templates(caption["output"])).resolve()
            if reference != expected:
                raise ValueError(f"UCIT caption reference mapping differs from manifest: {domain}")
            reference_bytes = reference.read_bytes()
            if hashlib.sha256(reference_bytes).hexdigest() != caption["output_sha256"]:
                raise ValueError(f"UCIT caption reference hash differs from manifest: {domain}")
            validate_caption_alignment(annotations, json.loads(reference_bytes), reference)
            hashes[f"{domain}:caption"] = caption["output_sha256"]
    if len(counts) != 1:
        raise ValueError("UCIT validation manifest has different per-domain test counts")
    return {"count": counts.pop(), "hashes": hashes, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def make_plan(project: Path, profile_file: Path, run_id: str, python_llava: str | None,
              python_t437: str | None, gpus: str, port: int, min_free_gib: float,
              auxiliary_paths: Path | None = None, execution_profile: str | None = None,
              job_select: list[str] | None = None, previous_run: Path | None = None,
              transition_receipt: Path | None = None, python_modalprompt: str | None = None,
              *, backbone_id: str = "llava", python_internvl: str | None = None,
              source_binding: Path | None = None, source_binding_sha256: str | None = None,
              gpu_lease_root: Path | None = None) -> dict[str, Any]:
    project = project.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be 1–80 letters, digits, dots, underscores or hyphens")
    ids = parse_gpus(gpus)
    if backbone_id not in {"llava", "internvl"}:
        raise ValueError(f"unknown validation backbone: {backbone_id}")
    if backbone_id == "internvl":
        if gpus != "0,1,2,3" or ids != ["0", "1", "2", "3"] or execution_profile != "four_gpu_validation":
            raise ValueError("InternVL sample validation requires four_gpu_validation on exactly GPUs 0,1,2,3")
        if min_free_gib < 32:
            raise ValueError("InternVL validation requires at least 32 GiB free disk")
        if bool(source_binding) != bool(source_binding_sha256):
            raise ValueError("source binding and its SHA-256 must be supplied together")
        if source_binding:
            source_binding = managed_path(source_binding, project, label="source_binding")
            if not re.fullmatch(r"[0-9a-f]{64}", source_binding_sha256 or ""):
                raise ValueError("source binding SHA-256 is invalid")
        from .internvl_validation import gpu_lease_root as checked_lease_root
        if gpu_lease_root is not None and not gpu_lease_root.is_absolute():
            gpu_lease_root = project / gpu_lease_root
        gpu_lease_root = checked_lease_root(project, gpu_lease_root)
    if execution_profile not in {None, "four_gpu_validation"}:
        raise ValueError(f"unknown execution profile: {execution_profile}")
    expected_gpus = 4 if execution_profile == "four_gpu_validation" else 2
    if ids is None or len(ids) != expected_gpus:
        raise ValueError(f"validation experiments require exactly {expected_gpus} distinct GPUs")
    if execution_profile == "four_gpu_validation" and ids != ["0", "1", "2", "3"]:
        raise ValueError("four_gpu_validation requires GPU IDs 0,1,2,3 in that order")
    if not 1 <= port <= 65535 or min_free_gib < 0:
        raise ValueError("master port must be 1..65535 and minimum free GiB must be non-negative")
    available = {f"{method.id}/{benchmark}" for method in METHODS for benchmark in BENCHMARKS
                 if not (method.id == "dgg" and execution_profile == "four_gpu_validation")
                 and compatibility(method.id, backbone_id)[0]
                 and benchmark_supported(method.id, benchmark)}
    if job_select is not None:
        if execution_profile == "four_gpu_validation" and any(key.split("/", 1)[0] == "dgg" for key in job_select):
            raise ValueError("DGG does not support the four-GPU execution profile")
        if not job_select or len(set(job_select)) != len(job_select) or set(job_select) - available:
            raise ValueError("job selection must contain distinct registered method/benchmark pairs")
        selected_jobs = set(job_select)
    else:
        selected_jobs = available
    if (backbone_id == "internvl" and any(key.split("/", 1)[0] in {"replay", "mrlora"} for key in selected_jobs)
            and auxiliary_paths is None):
        raise ValueError("InternVL Replay/MR-LoRA jobs require the reviewed auxiliary sample mappings")
    profile_file = profile_file if profile_file.is_absolute() else project / profile_file
    profile_file = profile_file.resolve()
    original = yaml.safe_load(profile_file.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise ValueError("paths profile must be a YAML mapping")
    paths = PathResolver.from_file(profile_file, repo_root=project)
    expected_dcl = (verified_dcl_sample_info(project, paths) if "mrlora/mllm_dcl" in selected_jobs else None)
    expected_ucit = (verified_ucit_sample_info(project, paths) if "mrlora/ucit" in selected_jobs else None)
    if bool(previous_run) != bool(transition_receipt):
        raise ValueError("previous_run and transition_receipt must be specified together")
    previous_run = (managed_path(previous_run, project, within=paths.logs_root, label="previous_run")
                    if previous_run else None)
    transition_receipt = (managed_path(transition_receipt, project, label="transition_receipt")
                          if transition_receipt else None)
    if auxiliary_paths is not None:
        auxiliary_paths = (auxiliary_paths if auxiliary_paths.is_absolute() else project / auxiliary_paths).resolve()
        auxiliary = yaml.safe_load(auxiliary_paths.read_text(encoding="utf-8"))
        original = apply_auxiliary_mappings(original, auxiliary, paths)
    namespace = Path("internvl") if backbone_id == "internvl" else Path()
    roots = {
        "model_root": str(paths.paths.model_root), "data_root": str(paths.paths.data_root),
        "ckpt_root": str(managed_path(paths.paths.ckpt_root, project, label="ckpt_root") / namespace / run_id),
        "results_root": str(managed_path(paths.results_root, project, label="results_root") / namespace / run_id),
        "logs_root": str(paths.logs_root / namespace / run_id),
    }
    run_dir = Path(roots["logs_root"])
    if previous_run == run_dir:
        raise ValueError("previous_run must differ from the new run directory")
    jobs = []
    for method in METHODS:
        for benchmark in BENCHMARKS:
            if f"{method.id}/{benchmark}" not in selected_jobs:
                continue
            job_dir = run_dir / "jobs" / method.id / benchmark
            effective = dict(original)
            effective.update(roots)
            effective["ckpt_root"] = str(Path(roots["ckpt_root"]) / method.id / benchmark)
            effective["results_root"] = str(Path(roots["results_root"]) / method.id / benchmark)
            # One logs root per queue lets execute_commands enforce its usual boundary.
            paths_file = job_dir / "paths.effective.yaml"
            if method.env == "llava-modalprompt-ds126":
                selected = python_modalprompt
            else:
                selected = (python_t437 if method.env == "llava-t437" else
                            ((python_internvl or python_llava or sys.executable) if backbone_id == "internvl" else python_llava))
            # Preserve venv symlinks: resolving them would select the base interpreter.
            python = os.path.abspath(os.path.join(project, os.path.expanduser(selected))) if selected else None
            executable = python or "<unconfigured>"
            # DGG's guidance callback is audited for one GPU. Give it the first
            # physical GPU from the queue allocation and serialize its evaluation.
            job_gpus = ids[0] if method.id == "dgg" else gpus
            base = [executable, "-m", "mcitlib.cli", "doctor", "--method", method.id, "--benchmark", benchmark,
                    "--paths", str(paths_file), "--json", "--validation-fixes",
                    "--gpus", job_gpus, "--master-port", str(port)]
            if method.id == "dgg":
                base += ["--dgg-sample-profile"]
            if backbone_id == "internvl":
                base += ["--backbone", "internvl"]
            if execution_profile:
                base += ["--execution-profile", execution_profile]
            sample_info = ({"ucit": expected_ucit, "mllm_dcl": expected_dcl}.get(benchmark)
                           if method.id == "mrlora" else None)
            expected_eval_samples = sample_info["count"] if sample_info else None
            expected_eval_hashes = sample_info["hashes"] if sample_info else None
            if expected_eval_samples is not None:
                base += ["--expected-eval-samples", str(expected_eval_samples),
                         "--expected-eval-hashes", json.dumps(expected_eval_hashes, sort_keys=True, separators=(",", ":"))]
            doctors = [("train", base + ["--sequence", "train"])]
            if method.id == "mrlora":
                doctors.append(("router", base + ["--sequence", "router"]))
            worker = [executable, "-m", "mcitlib.validation_batch", "_worker", "--method", method.id,
                      "--benchmark", benchmark, "--paths", str(paths_file), "--run-dir", str(job_dir),
                      "--gpus", job_gpus, "--master-port", str(port), "--min-free-gib", str(min_free_gib)]
            if backbone_id == "internvl":
                worker += ["--backbone", "internvl"]
                if source_binding:
                    worker += ["--source-binding", str(source_binding), "--source-binding-sha256", str(source_binding_sha256)]
            if execution_profile:
                worker += ["--execution-profile", execution_profile]
            if expected_eval_samples is not None:
                worker += ["--expected-eval-samples", str(expected_eval_samples),
                           "--expected-eval-hashes", json.dumps(expected_eval_hashes, sort_keys=True, separators=(",", ":"))]
            router_result = {"ucit": "UCIT_router", "mllm_dcl": "DCL_router"}.get(benchmark)
            metrics_result_dir = (Path(effective["results_root"]) / ("internvl" if backbone_id == "internvl" else "") / method.id / router_result
                                  if method.id == "mrlora" and router_result else None)
            if metrics_result_dir is not None:
                metrics = [executable, "-m", "mcitlib.cli", "metrics", "--method", method.id, "--run-dir", str(metrics_result_dir),
                           "--benchmark", benchmark, "--output", str(job_dir / "metrics.json")]
            else:
                metrics = [executable, "-m", "mcitlib.cli", "metrics", "--method", method.id, "--benchmark", benchmark,
                           "--paths", str(paths_file), "--output", str(job_dir / "metrics.json")]
            if backbone_id == "internvl":
                metrics += ["--backbone", "internvl"]
            jobs.append(Job(method.id, benchmark, method.env, python, job_dir, paths_file, effective,
                            doctors, worker, metrics, metrics_result_dir, expected_eval_samples, expected_eval_hashes))
    if backbone_id == "internvl":
        # The full merged SEFE checkpoints have the largest transient disk requirement.
        jobs.sort(key=lambda item: item.method == "sefe")
    return {"run_id": run_id, "project": project, "profile_file": profile_file, "roots": roots,
            "run_dir": run_dir, "lock_file": paths.logs_root / namespace / ".validation_queue.lock", "jobs": jobs,
            "gpus": gpus, "port": port, "min_free_gib": min_free_gib, "auxiliary_paths": auxiliary_paths,
            "execution_profile": execution_profile, "previous_run": previous_run,
            "transition_receipt": transition_receipt, **({"backbone": backbone_id,
            "source_binding": source_binding, "source_binding_sha256": source_binding_sha256,
            "gpu_lease_root": gpu_lease_root} if backbone_id == "internvl" else {})}


def apply_auxiliary_mappings(base: dict[str, Any], auxiliary: Any, resolver: PathResolver) -> dict[str, Any]:
    """Replace only exact logical replay/router mappings; preserve the base profile."""
    if not isinstance(auxiliary, dict) or set(auxiliary) != {"path_mappings"}:
        raise ValueError("auxiliary paths must contain only path_mappings")
    scopes = auxiliary["path_mappings"]
    if not isinstance(scopes, dict) or set(scopes) - set(BENCHMARKS):
        raise ValueError("auxiliary mappings support only ucit and mllm_dcl")
    result = copy.deepcopy(base)
    mappings = result.setdefault("path_mappings", {})
    for scope, entries in scopes.items():
        if not isinstance(entries, list):
            raise ValueError(f"auxiliary path_mappings.{scope} must be a list")
        replacements: dict[str, dict[str, str]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"from", "to"}:
                raise ValueError("auxiliary entries must contain exactly from and to")
            source, target = entry["from"], entry["to"]
            if not isinstance(source, str) or not isinstance(target, str) or not source.startswith("${data_root}/") \
                    or not source.endswith(("/replay.json", "/router.json")) or not target.startswith("${repo_root}/"):
                raise ValueError("auxiliary entries must map logical replay/router JSON files to project files")
            source_key = os.path.normpath(resolver.resolve_templates(source))
            managed_path(resolver.resolve_templates(target), resolver.paths.repo_root, label="auxiliary target")
            if source_key in replacements:
                raise ValueError(f"duplicate auxiliary mapping for {source}")
            replacements[source_key] = entry
        existing = mappings.get(scope, [])
        mappings[scope] = [entry for entry in existing
                           if os.path.normpath(resolver.resolve_templates(entry["from"])) not in replacements]
        mappings[scope].extend(replacements.values())
    return result


def plan_description(plan: dict[str, Any]) -> dict[str, Any]:
    return {"status": "preview", "run_id": plan["run_id"], "project": str(plan["project"]),
            **({"backbone": plan["backbone"], "source_binding": str(plan["source_binding"]) if plan["source_binding"] else None,
                "source_binding_sha256": plan["source_binding_sha256"], "gpu_lease_root": str(plan["gpu_lease_root"])}
               if plan.get("backbone") == "internvl" else {}),
            "source_paths": str(plan["profile_file"]),
            "auxiliary_paths": str(plan["auxiliary_paths"]) if plan["auxiliary_paths"] else None,
            "previous_run": str(plan["previous_run"]) if plan.get("previous_run") else None,
            "transition_receipt": str(plan["transition_receipt"]) if plan.get("transition_receipt") else None,
            "roots": plan["roots"], "gpus": plan["gpus"], "execution_profile": plan.get("execution_profile"),
            "master_port": plan["port"], "minimum_free_gib": plan["min_free_gib"], "validation_fixes": FIXES,
            "lock_file": str(plan["lock_file"]), "jobs": [
                {"method": job.method, "benchmark": job.benchmark, "environment": job.env_name,
                 "python": job.python, "directory": str(job.directory), "effective_paths": str(job.paths_file),
                 "doctors": [{"sequence": name, "argv": argv} for name, argv in job.doctors],
                 "worker": job.worker, "metrics": job.metrics,
                 **({"metrics_result_dir": str(job.metrics_result_dir)} if job.metrics_result_dir else {}),
                 **({"expected_eval_samples": job.expected_eval_samples, "expected_eval_hashes": job.expected_eval_hashes}
                    if job.expected_eval_samples else {})}
                for job in plan["jobs"]]}


def worker_commands(project: Path, paths: PathResolver, method: str, benchmark: str,
                    gpus: str, port: int, execution_profile: str | None = None,
                    expected_eval_samples: int | None = None,
                    expected_eval_hashes: dict[str, str] | None = None,
                    *, backbone_id: str = "llava") -> list[Any]:
    from .runner import build_sequence_commands

    method_id = get_method(method).id
    common = dict(config_root=project / "configs", backbone_id=backbone_id, method_id=method_id, benchmark_id=benchmark,
                  paths=paths, gpus=gpus, master_port=port, validation_fixes=True)
    if execution_profile:
        common["execution_profile"] = execution_profile
    if expected_eval_samples is not None:
        common["expected_eval_samples"] = expected_eval_samples
    if expected_eval_hashes is not None:
        common["expected_eval_hashes"] = expected_eval_hashes
    if method_id == "dgg" and (execution_profile or len(parse_gpus(gpus) or []) != 1):
        raise ValueError("DGG sample training requires exactly one GPU without the four-GPU profile")
    ordinary = build_sequence_commands(**common)
    if method_id == "dgg":
        from .methods._dgg.sample import apply_dgg_sample_profile

        apply_dgg_sample_profile(ordinary, benchmark, project)
    if get_method(method_id).separate_train_and_infer:
        # Keep any task-0 preparation before task 1, and infer after each task.
        commands = [item for item in ordinary if item.task_id == 0]
        for task in range(1, len(get_benchmark(benchmark).tasks) + 1):
            commands.extend(item for item in ordinary if item.task_id == task)
            commands.extend(build_sequence_commands(**common, task_id=task, only_eval=True))
    else:
        commands = ordinary
    if method_id == "mrlora":
        commands.extend(build_sequence_commands(**common, sequence="router"))
    if method_id in {"pclr", "same", "sigma", "octopus"}:
        # This queue is specifically the 3k/1k sample run. Its common budget
        # is independent of the full-data method configurations.
        world = len(gpus.split(","))
        if world not in {2, 4}:
            raise ValueError("sample training requires two or four GPUs")
        for command in commands:
            if (command.kind != "train" or command.sequence != "train" or
                    (method_id == "octopus" and command.metadata.get("octopus_stage") != "stage1")):
                continue
            values = {"--learning_rate": "0.0002", "--num_train_epochs": "1",
                      "--per_device_train_batch_size": "4", "--gradient_accumulation_steps": str(64 // (world * 4))}
            for flag, value in values.items():
                if command.argv.count(flag) != 1:
                    raise ValueError(f"sample command must have exactly one {flag}")
                command.argv[command.argv.index(flag) + 1] = value
            command.metadata["training_profile"] = {"id": "llava_sample_unified", "role": "main_train"}
            command.metadata["effective_batch"] = 64
    return commands


def verify_transition_receipt(plan: dict[str, Any]) -> None:
    """Require the reviewed, installed source and exact job selection before handoff execution."""
    receipt_path = plan.get("transition_receipt")
    if receipt_path is None:
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {"status": "applied", "runtime_project": str(plan["project"]),
                "previous_run": str(plan["previous_run"]), "new_run_id": plan["run_id"],
                "execution_profile": plan["execution_profile"], "gpus": plan["gpus"],
                "jobs": [job.key for job in plan["jobs"]]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("transition receipt does not match the new queue plan")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("transition receipt contains no installed source files")
    snapshot_path = managed_path(receipt["snapshot"], plan["project"], label="transition snapshot")
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != receipt["snapshot_sha256"]:
        raise ValueError("transition snapshot differs from its receipt")
    ledger = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if ledger.get("runtime_project") != str(plan["project"]) or not isinstance(ledger.get("runtime_hashes"), dict):
        raise ValueError("transition snapshot is not for this project")
    expected_hashes = dict(ledger["runtime_hashes"])
    for item in files:
        expected_hashes[item["path"]] = item["source_sha256"]
    for name, expected_hash in expected_hashes.items():
        path = managed_path(name, plan["project"], label="receipt source file")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"transition source differs from its receipt: {path}")


def checkpoint_error(argv: list[str], env: dict[str, str] | None = None) -> str | None:
    if "--output_dir" not in argv:
        return None
    output = Path(argv[argv.index("--output_dir") + 1])
    if not output.is_dir() or not any(entry.is_file() and entry.stat().st_size for entry in output.iterdir()):
        return f"training exited successfully but output directory is missing or empty: {output}"
    if "mcitlib.backbones.llava.train_entry" not in argv:
        return None
    if (env or {}).get("MCITLIB_METHOD") == "pclr":
        missing = [name for name in ("config.json", "pclr.pt")
                   if not (output / name).is_file() or not (output / name).stat().st_size]
        return f"PCLR checkpoint {output} is missing or has empty files: {', '.join(missing)}" if missing else None
    # These are the files written by the normal and prompt LoRA save branches,
    # and subsequently read by their inference and next-task loaders.
    required = ["config.json", "non_lora_trainables.bin"]
    lora = "--lora_enable" in argv and argv[argv.index("--lora_enable") + 1].lower() == "true"
    prompt = "--pt_enable" in argv and argv[argv.index("--pt_enable") + 1].lower() == "true"
    if not (lora or prompt):
        return None
    missing = [name for name in required if not (output / name).is_file() or not (output / name).stat().st_size]
    if lora and not any((output / name).is_file() and (output / name).stat().st_size
                        for name in ("adapter_model.bin", "adapter_model.safetensors")):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    return f"training checkpoint {output} is missing or has empty files: {', '.join(missing)}" if missing else None


def run_worker(args: argparse.Namespace) -> int:
    from .runner import execute_commands

    project = repo_root()
    paths = PathResolver.from_file(args.paths, repo_root=project)
    run_dir = paths.managed_run_dir(args.run_dir)
    event_file = run_dir / "events.jsonl"
    execution_profile = getattr(args, "execution_profile", None)
    expected_eval_samples = getattr(args, "expected_eval_samples", None)
    expected_eval_hashes = getattr(args, "expected_eval_hashes", None)
    backbone = getattr(args, "backbone", "llava")
    if backbone == "internvl":
        from .internvl_validation import audit_commands, verify_binding, verify_gpu_leases
        if args.gpus != "0,1,2,3" or execution_profile != "four_gpu_validation" or args.min_free_gib < 32:
            raise ValueError("InternVL worker requires GPUs 0,1,2,3, the four-GPU profile and a 32 GiB floor")
        source_binding = verify_binding(args.source_binding, args.source_binding_sha256, project)
        verify_gpu_leases(project)
        commands = worker_commands(project, paths, args.method, args.benchmark, args.gpus, args.master_port,
                                   execution_profile, expected_eval_samples, expected_eval_hashes, backbone_id="internvl")
        budget_audit = audit_commands(commands, method=args.method, task_count=len(get_benchmark(args.benchmark).tasks),
                                      source_files=source_binding["source_files"])
    else:
        commands = worker_commands(project, paths, args.method, args.benchmark, args.gpus, args.master_port,
                                   execution_profile, expected_eval_samples, expected_eval_hashes)
    router_result = {"ucit": "UCIT_router", "mllm_dcl": "DCL_router"}.get(get_benchmark(args.benchmark).id)
    metrics_result_dir = ((paths.method_results_dir("mrlora", backbone="internvl") if backbone == "internvl"
                           else paths.method_results_dir("mrlora")) / router_result
                          if get_method(args.method).id == "mrlora" and router_result else None)
    command_rows = [c.as_dict() for c in commands]
    command_digest = hashlib.sha256(json.dumps(command_rows, sort_keys=True, ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()
    atomic_json(run_dir / "commands.json", {"validation_fixes": FIXES, "execution_profile": execution_profile,
                                             "commands": command_rows,
                                             **({"backbone": "internvl", "command_sha256": command_digest,
                                                 "source_binding_sha256": args.source_binding_sha256,
                                                 "training_budget_audit": budget_audit}
                                                if backbone == "internvl" else {}),
                                             **({"metrics_result_dir": str(metrics_result_dir)} if metrics_result_dir else {}),
                                             **({"expected_eval_samples": expected_eval_samples,
                                                 "expected_eval_hashes": expected_eval_hashes}
                                                if expected_eval_samples is not None else {})})
    command_file_sha = hashlib.sha256((run_dir / "commands.json").read_bytes()).hexdigest() if backbone == "internvl" else None
    retention = None
    if backbone == "internvl" and args.method in {"sefe", "keeplora"}:
        from .internvl_retention import FreshMergedRetention
        retention = FreshMergedRetention(commands, project=project, ckpt_root=paths.paths.ckpt_root,
                                         job_dir=run_dir, benchmark=args.benchmark, method=args.method)

    def event(item: dict[str, Any]) -> None:
        if item["status"] == "completed" and item["kind"] == "train":
            problem = checkpoint_error(item["command"]["argv"], item["command"].get("env"))
            if problem:
                item["status"] = "failed"
                item["error"] = problem
        with event_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
        if item.get("error"):
            raise RuntimeError(item["error"])

    def before_step(index: int, command: Any) -> None:
        roots = {"ckpt_root": str(paths.paths.ckpt_root), "results_root": str(paths.results_root),
                 "logs_root": str(paths.logs_root)}
        minimum = getattr(args, "min_free_gib", 0)
        if backbone == "internvl":
            verify_binding(args.source_binding, args.source_binding_sha256, project)
            verify_gpu_leases(project)
            observed = hashlib.sha256(json.dumps([item.as_dict() for item in commands], sort_keys=True,
                                                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if (observed != command_digest or
                    hashlib.sha256((run_dir / "commands.json").read_bytes()).hexdigest() != command_file_sha):
                raise ValueError("InternVL command plan changed during execution")
            if retention is not None:
                retention.before_step(index)
            # Leave the floor intact even while a full merged model or growing top-P state is written.
            argv = " ".join(command.argv)
            if args.method in {"sefe", "keeplora"} and ("merge.py" in argv or "--save-model-path" in argv):
                minimum += 25
            elif args.method == "sefe" and "key_elements.py" in argv:
                minimum += 1 + 2 * max(1, int(command.task_id) - 1)
        problem = disk_shortage(roots, minimum)
        if problem:
            event({"status": "blocked_disk", "task": command.task_id, "kind": command.kind,
                   "index": index, "sequence": command.sequence, "error": problem})

    try:
        execute_commands(commands, run_dir, project, logs_root=paths.logs_root, on_event=event,
                         before_step=before_step)
        if retention is not None:
            retention.before_step(len(commands))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"worker failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


class Queue:
    def __init__(self, plan: dict[str, Any]):
        self.plan = plan
        self.state_path = plan["run_dir"] / "state.json"
        self.stopping = False
        self.state: dict[str, Any] = {}

    def save(self) -> None:
        counts: dict[str, int] = {}
        for item in self.state["jobs"].values():
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        self.state["summary"] = counts
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)

    def child(self, argv: list[str], stdout: Path, stderr: Path, job: dict[str, Any],
              *, doctor: bool = False, events: Path | None = None, cpu_probe: bool = False) -> tuple[int, str | None]:
        env = process_env({}, self.plan["project"])
        for name in ROOT_VARIABLES:
            env.pop(name, None)
        env["PATH"] = str(Path(argv[0]).parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        if self.plan.get("backbone") == "internvl":
            env.update(PYTHONPATH=os.pathsep.join(str(self.plan["project"] / path) for path in
                       (".", "third_party/LLaVA", "third_party/peft")), PYTHONNOUSERSITE="1",
                       MCITLIB_BACKBONE="internvl", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
            env.pop("NVIDIA_VISIBLE_DEVICES", None)
        if cpu_probe:
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif doctor:
            env["CUDA_VISIBLE_DEVICES"] = (argv[argv.index("--gpus") + 1]
                                           if "--gpus" in argv else self.plan["gpus"])
        else:
            # Assignment in the recorded commands is expressed in physical GPU IDs.
            env.pop("CUDA_VISIBLE_DEVICES", None)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        child: subprocess.Popen | None = None
        ticks: int | None = None
        known: dict[int, int] = {}
        lease_fds: tuple[int, ...] = ()
        tracker = None
        if self.plan.get("backbone") == "internvl":
            from .internvl_process_guard import OwnedProcesses
            from .internvl_validation import verify_gpu_leases
            lease_fds = tuple(row["fd"] for row in verify_gpu_leases(self.plan["project"]))
            tracker = OwnedProcesses()
            tracker.assert_no_existing_children()
        try:
            with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                child = subprocess.Popen(argv, cwd=self.plan["project"], env=env, stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=err, start_new_session=True, pass_fds=lease_fds)
                if tracker is not None:
                    tracker.start(child)
                info = proc_info(child.pid)
                if not info:
                    child.wait()
                    if tracker is not None:
                        tracker.cleanup(lambda: verify_gpu_leases(self.plan["project"]))
                    return child.returncode, None
                ticks = info[2]
                self.state["control"]["child"] = {"pid": child.pid, "start_ticks": ticks, "session": child.pid}
                self.save()
                position = 0
                terminated = False
                termination_time = 0.0
                next_group_scan = 0.0
                def read_events() -> None:
                    nonlocal position
                    if events and events.is_file():
                        with events.open(encoding="utf-8") as handle:
                            handle.seek(position)
                            while True:
                                line = handle.readline()
                                if not line or not line.endswith("\n"):
                                    break
                                event = json.loads(line)
                                job["task"] = event.get("task")
                                job["step"] = {key: value for key, value in event.items()
                                               if key not in {"command", "executed_argv"}}
                                if event.get("error"):
                                    job["error"] = event["error"]
                                if event.get("status") in {"completed", "failed"} and event.get("log"):
                                    aggregate = job["logs"]["training" if event.get("kind") == "train" else "evaluation"]
                                    with Path(aggregate).open("ab") as destination, Path(event["log"]).open("rb") as source:
                                        heading = (f"\n=== {event.get('sequence')} task {event.get('task')} "
                                                   f"{event.get('kind')} step {event.get('index')} "
                                                   f"{event.get('status')}: {event['log']} ===\n")
                                        destination.write(heading.encode("utf-8"))
                                        shutil.copyfileobj(source, destination)
                                self.save()
                                position = handle.tell()

                while True:
                    read_events()
                    if self.stopping or time.monotonic() >= next_group_scan:
                        if tracker is not None:
                            tracker.refresh()
                        else:
                            remember_group(child.pid, ticks, known)
                        next_group_scan = time.monotonic() + 1.0
                    code = child.poll()
                    if self.stopping and not terminated:
                        self.state["status"] = "stopping"
                        self.save()
                        if tracker is not None:
                            tracker.signal_owned(signal.SIGTERM)
                        else:
                            for pid, start in live_known(child.pid, known):
                                signal_member(pid, start, child.pid, signal.SIGTERM)
                        terminated = True
                        termination_time = time.monotonic()
                    if terminated and time.monotonic() - termination_time >= 5:
                        if tracker is not None:
                            tracker.signal_owned(signal.SIGKILL)
                        else:
                            for pid, start in live_known(child.pid, known):
                                signal_member(pid, start, child.pid, signal.SIGKILL)
                    if code is not None:
                        read_events()
                        if tracker is not None:
                            tracker.cleanup(lambda: verify_gpu_leases(self.plan["project"]))
                        else:
                            cleanup_group(child.pid, ticks, known)
                        self.state["control"]["child"] = None
                        self.save()
                        return code, None
                    time.sleep(0.25)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            # Popen failures must leave a useful log and cannot abort subsequent jobs.
            try:
                with stderr.open("a", encoding="utf-8") as handle:
                    handle.write(f"{type(error).__name__}: {error}\n")
            except OSError:
                pass
            return 1, f"{type(error).__name__}: {error}"
        finally:
            if tracker is not None and tracker.child is not None:
                tracker.cleanup(lambda: verify_gpu_leases(self.plan["project"]))
                self.state["control"]["child"] = None
            elif tracker is None and child is not None and ticks is not None and child.poll() is None:
                cleanup_group(child.pid, ticks, known, grace=1.0)
                child.wait()
                self.state["control"]["child"] = None

    def run_job(self, job_spec: Job) -> None:
        job = self.state["jobs"][job_spec.key]
        job["status"] = "running"
        job["logs"].update(training=str(job_spec.directory / "training.log"),
                           evaluation=str(job_spec.directory / "evaluation.log"))
        job["started_at"] = now()
        self.state["current"] = job_spec.key
        self.save()
        try:
            for name in ("training", "evaluation"):
                Path(job["logs"][name]).touch(exist_ok=False)
            free_space = free_space_gib(job_spec.profile)
            job["free_gib_before"] = free_space
            insufficient = {root: free for root, free in free_space.items() if free < self.plan["min_free_gib"]}
            if insufficient:
                details = ", ".join(f"{root}: {free:.2f} GiB" for root, free in insufficient.items())
                job.update(status="blocked_disk", error=f"insufficient free disk space: {details}")
                return
            if not job_spec.python or not os.path.isfile(job_spec.python) or not os.access(job_spec.python, os.X_OK):
                job.update(status="blocked", error=f"Python environment is unavailable: {job_spec.python or job_spec.env_name}")
                error_log = job_spec.directory / "doctor-train.stderr.log"
                error_log.write_text(job["error"] + "\n", encoding="utf-8")
                job["failure_log"] = str(error_log)
                job["logs"]["doctor_train_stderr"] = str(error_log)
                return
            if self.plan.get("backbone") == "internvl":
                from .internvl_validation import verify_binding
                binding = verify_binding(self.plan["source_binding"], self.plan["source_binding_sha256"], self.plan["project"])
                job["phase"] = "fresh_import_probe"
                probe_log = job_spec.directory / "import-probe.stdout.log"
                probe_err = job_spec.directory / "import-probe.stderr.log"
                code = ("import hashlib,importlib,json,sys;from pathlib import Path;"
                        "n=['mcitlib','mcitlib.validation_batch','mcitlib.backbones.internvl'];"
                        "d={x:{'path':str(Path(importlib.import_module(x).__file__).resolve()),"
                        "'sha256':hashlib.sha256(Path(importlib.import_module(x).__file__).read_bytes()).hexdigest()} for x in n};"
                        "print('INTERNVL_IMPORT='+json.dumps({'python':sys.executable,'modules':d},sort_keys=True))")
                self.save()
                code_result, error = self.child([job_spec.python, "-B", "-c", code], probe_log, probe_err, job, cpu_probe=True)
                job["logs"].update(import_probe=str(probe_log), import_probe_stderr=str(probe_err))
                if self.stopping:
                    job["status"] = "stopped"
                    return
                try:
                    rows = [line[len("INTERNVL_IMPORT="):] for line in probe_log.read_text().splitlines()
                            if line.startswith("INTERNVL_IMPORT=")]
                    if code_result or len(rows) != 1:
                        raise ValueError(error or f"fresh import probe exited {code_result}")
                    probe = json.loads(rows[0])
                    if not os.path.samefile(probe["python"], job_spec.python):
                        raise ValueError("fresh import used a different interpreter")
                    for name, rel in {"mcitlib": "mcitlib/__init__.py", "mcitlib.validation_batch": "mcitlib/validation_batch.py",
                                      "mcitlib.backbones.internvl": "mcitlib/backbones/internvl/__init__.py"}.items():
                        expected = {"path": str(self.plan["project"] / rel), "sha256": binding["source_files"][rel]}
                        if probe["modules"].get(name) != expected:
                            raise ValueError(f"fresh import used a different source: {name}")
                    atomic_json(job_spec.directory / "import-probe.json", {"status": "passed", **probe})
                    job["logs"]["import_probe_json"] = str(job_spec.directory / "import-probe.json")
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    job.update(status="blocked", error=f"fresh import proof failed: {exc}", failure_log=str(probe_err))
                    return
            for sequence, argv in job_spec.doctors:
                job["phase"] = f"doctor:{sequence}"
                job["task"] = None
                job["step"] = None
                log = job_spec.directory / f"doctor-{sequence}.json"
                err = job_spec.directory / f"doctor-{sequence}.stderr.log"
                job["logs"][f"doctor_{sequence}"] = str(log)
                job["logs"][f"doctor_{sequence}_stderr"] = str(err)
                self.save()
                code, error = self.child(argv, log, err, job, doctor=True)
                if self.stopping:
                    job["status"] = "stopped"
                    return
                failures = []
                hardware = []
                try:
                    report = doctor_report(log)
                    checks = report["checks"]
                    if not isinstance(checks, list):
                        raise ValueError("doctor checks are not a list")
                    failures = [c for c in checks if c.get("status") == "FAIL"]
                    if self.plan.get("backbone") == "internvl":
                        hardware = [c for c in checks if c.get("name") == "KeepLoRA memory" and c.get("status") in {"WARN", "FAIL"}]
                        failures += [c for c in hardware if c not in failures]
                        if not checks or any(c.get("status") not in {"OK", "WARN", "FAIL"} for c in checks):
                            raise ValueError("doctor did not return a complete recognized check list")
                    passed = code == 0 and report.get("failed") is False and not failures
                    explanation = "; ".join(f"{c.get('name')}: {c.get('detail')}" for c in failures[:4])
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    passed, explanation = False, f"invalid doctor report: {exc}"
                if not passed:
                    failure_log = log if failures or not err.is_file() or not err.stat().st_size else err
                    job.update(status="blocked_hardware" if hardware else "blocked", error=error or explanation or f"doctor exited {code}",
                               failure_log=str(failure_log))
                    if hardware:
                        job["hardware_evidence"] = {"checks": hardware, "doctor_report": str(log),
                                                    "policy": "KeepLoRA gradient accumulation must fit per GPU"}
                    return
            job["phase"] = "sequence"
            worker_log = job_spec.directory / "worker.stdout.log"
            worker_err = job_spec.directory / "worker.stderr.log"
            job["logs"].update(worker=str(worker_log), worker_stderr=str(worker_err),
                               events=str(job_spec.directory / "events.jsonl"),
                               commands=str(job_spec.directory / "commands.json"))
            self.save()
            code, error = self.child(job_spec.worker, worker_log, worker_err, job,
                                     events=job_spec.directory / "events.jsonl")
            if self.stopping:
                job["status"] = "stopped"
                return
            if code:
                step = job.get("step") or {}
                job.update(status="blocked_disk" if step.get("status") == "blocked_disk" else "failed",
                           error=job.get("error") or error or f"sequence exited {code}",
                           failure_log=(step.get("log") if step.get("status") == "failed" and not job.get("error") else None)
                           or str(worker_err))
                return
            job["phase"] = "metrics"
            problem = disk_shortage(job_spec.profile, self.plan["min_free_gib"])
            if problem:
                job.update(status="blocked_disk", error=problem)
                return
            metric_log = job_spec.directory / "metrics.stdout.log"
            metric_err = job_spec.directory / "metrics.stderr.log"
            job["logs"].update(metrics=str(metric_log), metrics_stderr=str(metric_err),
                               metrics_json=str(job_spec.directory / "metrics.json"))
            self.save()
            code, error = self.child(job_spec.metrics, metric_log, metric_err, job)
            if self.stopping:
                job["status"] = "stopped"
            elif code:
                job.update(status="failed", error=error or f"metrics exited {code}", failure_log=str(metric_err))
            else:
                job["status"] = "completed"
        except Exception as error:
            job.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            if job.get("failure_log"):
                job["error_tail"] = tail(job["failure_log"])
            job["finished_at"] = now()
            self.save()

    def execute(self) -> int:
        if self.plan.get("backbone") != "internvl":
            return self._execute_locked()
        from .internvl_validation import acquire_gpu_leases, gpu_idle, verify_gpu_leases
        names = ("MCITLIB_INTERNVL_LEASE_ROOT", "MCITLIB_INTERNVL_LEASE_FDS")
        saved = {name: os.environ.get(name) for name in names}
        fds: list[int] = []
        owns_leases = False
        started = False
        try:
            configured = str(self.plan["gpu_lease_root"])
            if saved[names[0]] is not None and saved[names[0]] != configured:
                raise ValueError("inherited GPU lease root differs from the queue plan")
            os.environ[names[0]] = configured
            if saved[names[1]]:
                verify_gpu_leases(self.plan["project"])
            else:
                _, fds = acquire_gpu_leases(self.plan["project"], self.plan["gpu_lease_root"])
                owns_leases = True
                os.environ[names[1]] = ",".join(map(str, fds))
            gpu_idle()
            started = True
            return self._execute_locked()
        finally:
            if owns_leases and started:
                # No owned subprocess may still be running (Queue.child proves this).
                # Do not release even our own locks while a GPU context remains.
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
                for sig in handlers:
                    signal.signal(sig, lambda _signal, _frame: None)
                try:
                    while True:
                        try:
                            verify_gpu_leases(self.plan["project"])
                            gpu_idle()
                            break
                        except Exception as error:
                            print(f"retaining GPU 0–3 leases until idle: {error}", file=sys.stderr, flush=True)
                            time.sleep(30)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
            for fd in fds:
                os.close(fd)
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _execute_locked(self) -> int:
        plan = self.plan
        if plan.get("backbone") == "internvl":
            from .internvl_validation import verify_binding, verify_gpu_leases
            if not plan.get("source_binding") or not plan.get("source_binding_sha256"):
                raise ValueError("InternVL execution requires a reviewed runtime source binding")
            binding = verify_binding(plan["source_binding"], plan["source_binding_sha256"], plan["project"])
            lease_evidence = verify_gpu_leases(plan["project"])
            if binding["profile"]["path"] != str(plan["profile_file"]):
                raise ValueError("runtime binding identifies a different paths profile")
            if (binding.get("auxiliary") or {}).get("path") != (str(plan["auxiliary_paths"]) if plan["auxiliary_paths"] else None):
                raise ValueError("runtime binding identifies a different auxiliary profile")
        plan["lock_file"].parent.mkdir(parents=True, exist_ok=True)
        with plan["lock_file"].open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another validation queue holds {plan['lock_file']}") from error
            verify_transition_receipt(plan)
            for name in ("logs_root", "ckpt_root", "results_root"):
                path = Path(plan["roots"][name])
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite an existing queue output: {path}")
            plan["run_dir"].mkdir(parents=True, exist_ok=False)
            for job in plan["jobs"]:
                job.directory.mkdir(parents=True)
                job.paths_file.write_text(yaml.safe_dump(job.profile, sort_keys=False, allow_unicode=True), encoding="utf-8")
            info = proc_info(os.getpid())
            if info is None:
                raise RuntimeError("cannot read queue process identity from /proc")
            self.state = {"version": 1, "status": "running", "run_id": plan["run_id"],
                          **({"backbone": "internvl", "source_binding": str(plan["source_binding"]),
                              "source_binding_sha256": plan["source_binding_sha256"], "lease_evidence": lease_evidence,
                              "gpu_lease_root": str(plan["gpu_lease_root"])}
                             if plan.get("backbone") == "internvl" else {}),
                          "started_at": now(), "project": str(plan["project"]), "roots": plan["roots"],
                          "source_paths": str(plan["profile_file"]), "validation_fixes": FIXES,
                          "previous_run": str(plan["previous_run"]) if plan.get("previous_run") else None,
                          "transition_receipt": str(plan["transition_receipt"]) if plan.get("transition_receipt") else None,
                          "auxiliary_paths": str(plan["auxiliary_paths"]) if plan["auxiliary_paths"] else None,
                          "gpus": plan["gpus"], "master_port": plan["port"],
                          "execution_profile": plan.get("execution_profile"),
                          "minimum_free_gib": plan["min_free_gib"], "current": None,
                          "control": {"pid": os.getpid(), "start_ticks": info[2],
                                      "lock_file": str(plan["lock_file"]), "child": None},
                          "jobs": {job.key: {"method": job.method, "benchmark": job.benchmark,
                                             "environment": job.env_name, "python": job.python,
                                             "status": "pending", "effective_paths": str(job.paths_file),
                                             "directory": str(job.directory), "logs": {},
                                             **({"metrics_result_dir": str(job.metrics_result_dir)}
                                                if job.metrics_result_dir else {}),
                                             **({"expected_eval_samples": job.expected_eval_samples,
                                                 "expected_eval_hashes": job.expected_eval_hashes}
                                                if job.expected_eval_samples else {})} for job in plan["jobs"]}}
            self.save()
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            def request_stop(_signum: int, _frame: Any) -> None:
                self.stopping = True
            for sig in previous:
                signal.signal(sig, request_stop)
            try:
                for job in plan["jobs"]:
                    if self.stopping:
                        break
                    if plan.get("backbone") == "internvl":
                        try:
                            verify_binding(plan["source_binding"], plan["source_binding_sha256"], plan["project"])
                            verify_gpu_leases(plan["project"])
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            self.state.update(status="blocked_source_or_lease", error=f"{type(error).__name__}: {error}",
                                              finished_at=now(), current=None)
                            self.save()
                            return 1
                    self.run_job(job)
                if self.stopping:
                    for job in self.state["jobs"].values():
                        if job["status"] == "pending":
                            job["status"] = "not_run"
                    self.state["status"] = "stopped"
                elif all(job["status"] == "completed" for job in self.state["jobs"].values()) and plan["jobs"]:
                    self.state["status"] = "completed"
                else:
                    self.state["status"] = "completed_with_failures"
                self.state["finished_at"] = now()
                self.state["current"] = None
                self.save()
                print(json.dumps({"status": self.state["status"], "summary": self.state["summary"],
                                  "state": str(self.state_path)}, ensure_ascii=False), flush=True)
                return 0 if self.state["status"] == "completed" else 1
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def stop_queue(run_dir: Path, project: Path) -> None:
    state_path = run_dir.resolve() / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["project"] != str(project.resolve()) or state["status"] not in {"running", "stopping"}:
        raise ValueError("state is not a running queue in this project")
    control = state["control"]
    pid, ticks = int(control["pid"]), int(control["start_ticks"])
    lock = managed_path(control["lock_file"], project, label="queue lock")
    if proc_info(pid) is None or proc_info(pid)[2] != ticks:
        raise ValueError("queue PID is no longer the recorded process")
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if b"mcitlib.validation_batch" not in cmd or b"execute" not in cmd:
        raise ValueError("recorded process is not a validation queue")
    if Path(f"/proc/{pid}/cwd").resolve() != project.resolve():
        raise ValueError("queue process has a different working directory")
    lock_stat = lock.stat()
    owns_fd = any(_same_file(entry, lock_stat) for entry in Path(f"/proc/{pid}/fd").iterdir())
    if not owns_fd:
        raise ValueError("queue process does not hold its recorded lock file")
    with lock.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("queue lock is not held")
    fd = pidfd_open(pid)
    try:
        info = proc_info(pid)
        if info is None or info[2] != ticks:
            raise ValueError("queue PID changed while validating identity")
        pidfd_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def _same_file(path: Path, expected: os.stat_result) -> bool:
    try:
        actual = path.stat()
        return (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    except OSError:
        return False


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    for action in ("preview", "execute"):
        command = sub.add_parser(action)
        command.add_argument("--backbone", choices=("llava", "internvl"), default="llava")
        command.add_argument("--paths", type=Path, default=Path("configs/paths.validation.yaml"))
        command.add_argument("--auxiliary-paths", type=Path, help="exact replay/router mapping overlay")
        command.add_argument("--run-id", default=None)
        command.add_argument("--python-llava", default=None)
        command.add_argument("--python-internvl", default=None)
        command.add_argument("--python-t437", default=None)
        command.add_argument("--python-modalprompt", default=None,
                             help="Python from the isolated llava-modalprompt-ds126 environment")
        command.add_argument("--gpus", default="0,1")
        command.add_argument("--execution-profile", choices=("four_gpu_validation",))
        command.add_argument("--job", action="append", dest="job_select", metavar="METHOD/BENCHMARK",
                             help="select an exact job; repeat to select several")
        command.add_argument("--previous-run", type=Path, help="prior queue directory for a reviewed handoff")
        command.add_argument("--transition-receipt", type=Path, help="successful source handoff receipt")
        command.add_argument("--master-port", type=int, default=9001)
        command.add_argument("--min-free-gib", type=float, default=20)
        command.add_argument("--source-binding", type=Path, help="reviewed InternVL source and sample-input binding")
        command.add_argument("--source-binding-sha256", help="SHA-256 of the reviewed source binding")
        command.add_argument("--gpu-lease-root", type=Path, help="shared project GPU lease directory")
    binder = sub.add_parser("bind-internvl", help="freeze InternVL source and sample evidence without using GPUs")
    binder.add_argument("--paths", type=Path, default=Path("configs/paths.validation.yaml"))
    binder.add_argument("--auxiliary-paths", type=Path)
    binder.add_argument("--output", type=Path, required=True)
    for action in ("status", "stop"):
        command = sub.add_parser(action)
        command.add_argument("--run-dir", type=Path, required=True)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--backbone", choices=("llava", "internvl"), default="llava")
    worker.add_argument("--method", required=True)
    worker.add_argument("--benchmark", required=True)
    worker.add_argument("--paths", type=Path, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--gpus", required=True)
    worker.add_argument("--execution-profile", choices=("four_gpu_validation",))
    worker.add_argument("--expected-eval-samples", type=int)
    worker.add_argument("--expected-eval-hashes", type=json.loads)
    worker.add_argument("--master-port", type=int, required=True)
    worker.add_argument("--min-free-gib", type=float, required=True)
    worker.add_argument("--source-binding", type=Path)
    worker.add_argument("--source-binding-sha256")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        project = repo_root()
        if args.action == "bind-internvl":
            from .internvl_validation import build_binding, source_inventory, verify_binding
            profile = args.paths if args.paths.is_absolute() else project / args.paths
            auxiliary = (args.auxiliary_paths if args.auxiliary_paths.is_absolute() else project / args.auxiliary_paths) if args.auxiliary_paths else None
            output = managed_path(args.output, project, label="binding output")
            evidence = build_binding(project, profile, auxiliary)
            if source_inventory(project) != evidence["source_files"]:
                raise ValueError("source changed while preparing the binding")
            payload = (json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
            output.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            expected = hashlib.sha256(payload).hexdigest()
            verify_binding(output, expected, project)
            print(json.dumps({"status": "bound", "path": str(output), "sha256": expected}, ensure_ascii=False))
            return 0
        if args.action == "_worker":
            return run_worker(args)
        if args.action in {"status", "stop"}:
            if args.action == "status":
                print((args.run_dir / "state.json").read_text(encoding="utf-8"))
            else:
                stop_queue(args.run_dir, project)
                print(f"stop requested for {args.run_dir}")
            return 0
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        plan = make_plan(project, args.paths, run_id, args.python_llava, args.python_t437,
                         args.gpus, args.master_port, args.min_free_gib, args.auxiliary_paths,
                         args.execution_profile, args.job_select, args.previous_run, args.transition_receipt,
                         python_modalprompt=args.python_modalprompt, backbone_id=args.backbone,
                         python_internvl=args.python_internvl, source_binding=args.source_binding,
                         source_binding_sha256=args.source_binding_sha256, gpu_lease_root=args.gpu_lease_root)
        if args.action == "preview":
            print(json.dumps(plan_description(plan), ensure_ascii=False, indent=2))
            return 0
        print(f"queue state: {plan['run_dir'] / 'state.json'}", flush=True)
        os.chdir(project)
        return Queue(plan).execute()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"validation queue: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
