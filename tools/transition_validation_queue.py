# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Safely hand off an existing validation queue between independent jobs.

The monitor never signals a sequence worker. It holds only a verified doctor
child while requesting the existing queue's normal stop and has a separate
watchdog to release that child if the monitor dies or stalls.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mcitlib.paths import managed_path
from mcitlib.validation_batch import (
    ROOT_VARIABLES,
    _same_file,
    atomic_json,
    make_plan,
    now,
    pidfd_open,
    pidfd_signal,
    plan_description,
    proc_info,
    stop_queue,
)

TERMINAL = {"completed", "completed_with_failures", "stopped"}
ATTEMPTED = {"completed", "failed", "blocked", "blocked_disk"}


def read_state(run_dir: Path) -> dict[str, Any]:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("jobs"), dict):
        raise ValueError("invalid queue state")
    return state


def check_identity(state: dict[str, Any], project: Path, run_id: str, pid: int, ticks: int,
                   *, running: bool) -> None:
    control = state["control"]
    if state.get("project") != str(project) or state.get("run_id") != run_id:
        raise ValueError("queue project or run id changed")
    if (control.get("pid"), control.get("start_ticks")) != (pid, ticks):
        raise ValueError("queue identity differs from the expected PID and start ticks")
    if not running:
        return
    info = proc_info(pid)
    if info is None or info[2] != ticks:
        raise ValueError("queue PID no longer identifies the expected process")
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if b"mcitlib.validation_batch" not in argv or b"execute" not in argv:
        raise ValueError("expected process is not a validation queue")
    if Path(f"/proc/{pid}/cwd").resolve() != project:
        raise ValueError("queue working directory changed")
    lock = managed_path(control["lock_file"], project, label="queue lock")
    stat = lock.stat()
    if not any(_same_file(entry, stat) for entry in Path(f"/proc/{pid}/fd").iterdir()):
        raise ValueError("queue does not hold its recorded lock file")
    with lock.open("r", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("queue lock is not locked")


def process_stat(pid: int) -> tuple[str, int, int, int, int] | None:
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = line[line.rfind(")") + 2:].split()
        return fields[0], int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def doctor_child(state: dict[str, Any], pid: int) -> tuple[str, int, int] | None:
    key = state.get("current")
    job = state["jobs"].get(key, {})
    child = state["control"].get("child")
    if not (job.get("status") == "running" and str(job.get("phase", "")).startswith("doctor:")
            and isinstance(child, dict)):
        return None
    child_pid, ticks = int(child["pid"]), int(child["start_ticks"])
    info = process_stat(child_pid)
    if (info is None or info[0] in {"Z", "X"} or info[1:] != (pid, child_pid, child_pid, ticks)
            or child.get("session") != child_pid):
        return None
    argv = Path(f"/proc/{child_pid}/cmdline").read_bytes().split(b"\0")
    if b"mcitlib.cli" not in argv or b"doctor" not in argv or b"_worker" in argv:
        return None
    return str(key), child_pid, ticks


@contextmanager
def doctor_hold(pid: int, ticks: int, timeout: float = 10.0) -> Iterator[int]:
    """Stop a pidfd-pinned doctor with a watchdog which cannot release it too early.

    The watchdog resumes the child on normal exit. If the monitor stalls, the
    watchdog first kills the monitor and waits until it is gone before resuming
    the doctor, so a delayed stop request cannot race the resumed child.
    """
    if timeout <= 0:
        raise ValueError("watchdog timeout must be positive")
    child_fd = pidfd_open(pid)
    monitor_fd = pidfd_open(os.getpid())
    read_fd, write_fd = os.pipe()
    watcher = os.fork()
    if watcher == 0:
        try:
            os.close(write_fd)
            os.setsid()
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1)
            os.dup2(null, 2)
            os.close(null)
            ready, _, _ = select.select([read_fd], [], [], timeout)
            if not ready:
                try:
                    pidfd_signal(monitor_fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # A pidfd becomes readable only when this particular process
                # exits, regardless of subsequent numeric PID reuse.
                select.select([monitor_fd], [], [])
            try:
                pidfd_signal(child_fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
        finally:
            os._exit(0)
    os.close(read_fd)
    os.close(monitor_fd)
    try:
        info = process_stat(pid)
        if info is None or info[-1] != ticks:
            raise ValueError("doctor PID changed before the hold")
        pidfd_signal(child_fd, signal.SIGSTOP)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            info = process_stat(pid)
            if info is None or info[-1] != ticks:
                raise ValueError("doctor exited during the hold")
            if info[0] in {"T", "t"}:
                break
            time.sleep(0.01)
        else:
            raise ValueError("doctor did not enter the stopped state")
        yield child_fd
    finally:
        os.close(write_fd)
        os.waitpid(watcher, 0)
        os.close(child_fd)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **payload}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def monitor(project: Path, run_dir: Path, run_id: str, pid: int, ticks: int, finish_job: str,
            status_file: Path, log_file: Path, poll: float = 0.5) -> dict[str, Any]:
    project, run_dir = project.resolve(), run_dir.resolve()
    if not run_dir.is_relative_to(project) or not 0.05 <= poll <= 10:
        raise ValueError("run directory or poll interval is invalid")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.with_suffix(status_file.suffix + ".lock").open("a+", encoding="utf-8") as monitor_lock:
        try:
            fcntl.flock(monitor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another transition monitor is active") from error
        record: dict[str, Any] = {"project": str(project), "run_dir": str(run_dir), "run_id": run_id,
                                  "pid": pid, "start_ticks": ticks, "finish_job": finish_job,
                                  "status": "watching", "stop_attempt": None}
        if status_file.is_file():
            previous = json.loads(status_file.read_text(encoding="utf-8"))
            if any(previous.get(k) != record[k] for k in ("project", "run_dir", "run_id", "pid", "start_ticks",
                                                           "finish_job")):
                raise ValueError("existing monitor status belongs to a different queue")
            record["stop_attempt"] = previous.get("stop_attempt")

        def update(status: str, **details: Any) -> None:
            record.update(status=status, updated_at=now(), **details)
            atomic_json(status_file, record)
            append_event(log_file, {"status": status, **details})

        update("watching")
        last_phase = None
        while True:
            state = read_state(run_dir)
            terminal = state.get("status") in TERMINAL
            check_identity(state, project, run_id, pid, ticks, running=not terminal)
            if finish_job not in state["jobs"]:
                raise ValueError("the job that must finish is absent from the old queue")
            if terminal:
                update("old_queue_terminal", old_status=state["status"])
                return record
            if state.get("status") == "stopping":
                time.sleep(poll)
                continue
            current = state.get("current")
            phase = state["jobs"].get(current, {}).get("phase")
            seen = (current, phase, state["jobs"][finish_job].get("status"))
            if seen != last_phase:
                update("watching", current_job=current, phase=phase,
                       finish_job_status=state["jobs"][finish_job].get("status"))
                last_phase = seen
            candidate = doctor_child(state, pid)
            if state["jobs"][finish_job].get("status") in ATTEMPTED and candidate and candidate[0] != finish_job:
                key, child_pid, child_ticks = candidate
                with doctor_hold(child_pid, child_ticks):
                    latest = read_state(run_dir)
                    check_identity(latest, project, run_id, pid, ticks, running=True)
                    if doctor_child(latest, pid) != candidate or process_stat(child_pid)[0] not in {"T", "t"}:
                        continue
                    attempt = {"job": key, "phase": latest["jobs"][key]["phase"], "child_pid": child_pid,
                               "child_start_ticks": child_ticks, "recorded_at": now()}
                    update("stop_about_to_send", stop_attempt=attempt)
                    stop_queue(run_dir, project)
                    update("stop_requested", stop_attempt=attempt)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        current_state = read_state(run_dir)
                        check_identity(current_state, project, run_id, pid, ticks,
                                       running=current_state.get("status") not in TERMINAL)
                        if current_state.get("status") in TERMINAL | {"stopping"}:
                            break
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("queue did not acknowledge the stop request")
            time.sleep(poll)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(name: str) -> Path:
    path = Path(name)
    allowed = {"mcitlib", "configs", "third_party", "tools", "tests", "docs", "envs", ".github",
               ".gitignore", "README.md", "pyproject.toml"}
    if (not name or path.is_absolute() or any(part in {".", ".."} for part in path.parts)
            or not path.parts or path.parts[0] not in allowed):
        raise ValueError(f"source path is outside the handoff allowlist: {name}")
    sample_readme = path.parts == ("configs", "validation_3k_1k", "README.md")
    if (any(part in {".git", ".workspaces", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
            or part.endswith(".egg-info") for part in path.parts)
            or (path.parts[0] == "envs" and path.parts not in {
                ("envs", "llava.txt"), ("envs", "llava-modalprompt-ds126.txt")})
            or (path.parts[0] == "configs" and len(path.parts) > 1
                and (path.parts[1].startswith("paths") or
                     (path.parts[1].startswith("validation_") and not sample_readme)))):
        raise ValueError(f"protected path cannot be synchronized: {name}")
    return path


def safe_join(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"path escapes its root: {relative}")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlink is not permitted in a synchronized path: {cursor}")
    if not cursor.resolve().is_relative_to(root):
        raise ValueError(f"path escapes its root: {cursor}")
    return cursor


def verify_ledger(project: Path, ledger: dict[str, Any]) -> None:
    expected = ledger.get("runtime_hashes")
    if ledger.get("runtime_project") != str(project) or not isinstance(expected, dict) or not expected:
        raise ValueError("snapshot ledger is not for this runtime project")
    for name, value in expected.items():
        path = safe_join(project, Path(name))
        if not path.is_file() or digest(path) != value:
            raise ValueError(f"runtime source differs from snapshot ledger: {name}")
    baseline = Path(ledger["baseline_snapshot"]).resolve()
    unexpected = source_files(project) - source_files(baseline)
    if unexpected:
        raise ValueError(f"unrecorded runtime source appeared after the snapshot: {sorted(unexpected)[0]}")


@contextmanager
def idle_queue(project: Path, run_dir: Path, run_id: str, pid: int, ticks: int) -> Iterator[dict[str, Any]]:
    state = read_state(run_dir)
    check_identity(state, project, run_id, pid, ticks, running=False)
    if state.get("status") not in TERMINAL or state["control"].get("child") is not None:
        raise ValueError("old queue is not terminal or still records a child")
    info = proc_info(pid)
    if info and info[2] == ticks:
        raise ValueError("old queue process has not exited")
    lock = managed_path(state["control"]["lock_file"], project, label="queue lock")
    with lock.open("r", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("old queue lock is still held") from error
        latest = read_state(run_dir)
        if latest != state:
            raise ValueError("old queue state changed during lock acquisition")
        yield state


def remaining_jobs(state: dict[str, Any], monitor_state: dict[str, Any], run_dir: Path) -> list[str]:
    selected = []
    attempt = monitor_state.get("stop_attempt") or {}
    for key, job in state["jobs"].items():
        status = job.get("status")
        if status in {"pending", "not_run"}:
            selected.append(key)
        elif status == "stopped":
            if (attempt.get("job") != key or not str(attempt.get("phase", "")).startswith("doctor:")
                    or job.get("phase") != attempt.get("phase") or job.get("step") is not None):
                raise ValueError(f"cannot prove the stopped job was still in doctor: {key}")
            child = process_stat(int(attempt["child_pid"]))
            if child and child[-1] == int(attempt["child_start_ticks"]) and child[0] not in {"Z", "X"}:
                raise ValueError("stopped doctor child is still alive")
            directory = Path(job["directory"]).resolve()
            if not directory.is_relative_to(run_dir):
                raise ValueError("stopped job directory is outside the old run")
            events = directory / "events.jsonl"
            if events.exists() and events.stat().st_size:
                raise ValueError("stopped doctor job already has worker events")
            selected.append(key)
        elif status not in ATTEMPTED:
            raise ValueError(f"cannot select a job with unexpected status {status!r}: {key}")
    return selected


def source_files(root: Path) -> set[str]:
    found: set[str] = set()
    for name in ("mcitlib", "configs", "third_party", "tools", "tests", "docs", ".github",
                 ".gitignore", "README.md", "pyproject.toml"):
        start = root / name
        if not start.exists():
            continue
        if start.is_file():
            found.add(name)
            continue
        for directory, dirs, files in os.walk(start, followlinks=False):
            base = Path(directory)
            dirs[:] = [part for part in dirs if not (base / part).is_symlink()]
            for filename in files:
                path = base / filename
                relative = str(path.relative_to(root))
                try:
                    source_path(relative)
                except ValueError:
                    continue
                if path.is_symlink():
                    raise ValueError(f"source file is a symlink: {relative}")
                found.add(relative)
    return found


def proposed_files(project: Path, baseline: Path, development: Path, ledger: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_files, development_files = source_files(baseline), source_files(development)
    missing = baseline_files - development_files
    if missing:
        raise ValueError(f"source deletion requires separate review: {sorted(missing)[0]}")
    result = []
    for name in sorted(development_files):
        relative = source_path(name)
        source = safe_join(development, relative)
        old = safe_join(baseline, relative)
        before = digest(old) if old.is_file() else None
        after = digest(source)
        if before == after:
            continue
        recorded = ledger["runtime_hashes"].get(name)
        if before is not None and before != recorded:
            raise ValueError(f"baseline is not covered by the snapshot ledger: {name}")
        target = safe_join(project, relative)
        if before is None and (recorded is not None or target.exists()):
            raise ValueError(f"new source collides with an existing runtime file: {name}")
        result.append({"path": name, "baseline_sha256": before, "source_sha256": after})
    return result


def write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    staged = stage_file(path, content, 0o600)
    try:
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def preview(project: Path, run_dir: Path, run_id: str, pid: int, ticks: int, finish_job: str, snapshot: Path,
            new_run_id: str, output: Path, monitor_status: Path, monitor_log: Path, plan_path: Path,
            receipt_path: Path, launch_log: Path, *, port: int = 9001) -> dict[str, Any]:
    project, run_dir, snapshot, output, monitor_status, monitor_log, plan_path, receipt_path, launch_log = (
        value.resolve() for value in (project, run_dir, snapshot, output, monitor_status, monitor_log, plan_path,
                                      receipt_path, launch_log)
    )
    if not run_dir.is_relative_to(project):
        raise ValueError("old run is outside the runtime project")
    execution_status = output.with_name(output.stem + ".execution.json")
    destinations = (output, monitor_status, monitor_log, plan_path, receipt_path, launch_log, execution_status)
    if (len(set(destinations)) != len(destinations) or any(not p.is_relative_to(project / ".workspaces")
                                                            or p.exists() for p in destinations)):
        raise ValueError("handoff files must be distinct new paths inside the runtime .workspaces directory")
    if any(name in os.environ for name in ROOT_VARIABLES):
        raise ValueError("clear MCITLIB_*_ROOT environment overrides before preview")
    state = read_state(run_dir)
    check_identity(state, project, run_id, pid, ticks, running=False)
    if finish_job not in state["jobs"]:
        raise ValueError("the required baseline job is not present in the old queue")
    ledger = json.loads(snapshot.read_text(encoding="utf-8"))
    development = Path(ledger["development_project"]).resolve()
    baseline = Path(ledger["baseline_snapshot"]).resolve()
    verify_ledger(project, ledger)
    files = proposed_files(project, baseline, development, ledger)
    if not files:
        raise ValueError("no engineering source changes to review")
    # This is only a path/argument preview; the final remaining jobs are decided after the old queue stops.
    sample = make_plan(project, Path(state["source_paths"]), new_run_id, sys.executable, sys.executable,
                       "0,1,2,3", port, float(state["minimum_free_gib"]),
                       Path(state["auxiliary_paths"]) if state.get("auxiliary_paths") else None,
                       "four_gpu_validation", [finish_job])
    if any(Path(sample["roots"][name]).exists() for name in ("ckpt_root", "results_root", "logs_root")):
        raise FileExistsError("a new queue output root already exists")
    approved = {"schema_version": 1, "kind": "validation_transition_source_review", "created_at": now(),
                "runtime_project": str(project), "development_project": str(development),
                "baseline_snapshot": str(baseline), "snapshot": str(snapshot), "snapshot_sha256": digest(snapshot),
                "old_run_dir": str(run_dir), "old_run_id": run_id, "old_pid": pid, "old_start_ticks": ticks,
                "finish_job": finish_job, "new_run_id": new_run_id, "execution_profile": "four_gpu_validation",
                "gpus": "0,1,2,3", "master_port": port, "source_paths": state["source_paths"],
                "auxiliary_paths": state.get("auxiliary_paths"), "minimum_free_gib": state["minimum_free_gib"],
                "new_roots": sample["roots"], "monitor_status": str(monitor_status), "monitor_log": str(monitor_log),
                "plan_path": str(plan_path), "receipt": str(receipt_path), "launch_log": str(launch_log),
                "selection_rule": "pending/not_run plus a stopped job proven to be held in doctor; "
                                  "completed/failed/blocked/blocked_disk are not retried",
                "files": files}
    write_new(output, approved)
    return {"status": "review_ready", "manifest": str(output), "manifest_sha256": digest(output),
            "source_files": len(files), "old_run_id": run_id, "pid": pid, "start_ticks": ticks,
            "finish_job": finish_job, "new_run_id": new_run_id, "gpus": "0,1,2,3",
            "selection_rule": approved["selection_rule"]}


def approved_manifest(path: Path, expected_digest: str) -> dict[str, Any]:
    if digest(path) != expected_digest:
        raise ValueError("reviewed source manifest SHA-256 does not match")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("kind") != "validation_transition_source_review":
        raise ValueError("unsupported source review manifest")
    return data


def prepare(project: Path, run_dir: Path, run_id: str, pid: int, ticks: int, snapshot: Path,
            monitor_status: Path, new_run_id: str, receipt_path: Path, output: Path,
            approved_path: Path, approved_sha256: str, *, gpus: str = "0,1,2,3", port: int = 9001) -> dict[str, Any]:
    project, run_dir, snapshot, monitor_status, receipt_path, output = (
        value.resolve() for value in (project, run_dir, snapshot, monitor_status, receipt_path, output)
    )
    if not run_dir.is_relative_to(project) or not receipt_path.is_relative_to(project / ".workspaces"):
        raise ValueError("run directory or receipt location is invalid")
    if output == receipt_path or output.exists() or receipt_path.exists():
        raise FileExistsError("handoff manifest or receipt already exists")
    if any(name in os.environ for name in ROOT_VARIABLES):
        raise ValueError("clear MCITLIB_*_ROOT environment overrides before preparing the handoff")
    approved_path = approved_path.resolve()
    approved = approved_manifest(approved_path, approved_sha256)
    expected = {"runtime_project": str(project), "old_run_dir": str(run_dir), "old_run_id": run_id,
                "old_pid": pid, "old_start_ticks": ticks, "snapshot": str(snapshot), "new_run_id": new_run_id,
                "receipt": str(receipt_path), "monitor_status": str(monitor_status), "plan_path": str(output),
                "gpus": gpus, "master_port": port}
    if any(approved.get(key) != value for key, value in expected.items()):
        raise ValueError("handoff arguments do not match the reviewed source manifest")
    ledger = json.loads(snapshot.read_text(encoding="utf-8"))
    if digest(snapshot) != approved["snapshot_sha256"]:
        raise ValueError("snapshot ledger changed after source review")
    development = Path(ledger["development_project"]).resolve()
    baseline = Path(ledger["baseline_snapshot"]).resolve()
    observed_monitor = json.loads(monitor_status.read_text(encoding="utf-8"))
    expected_monitor = {"project": str(project), "run_dir": str(run_dir), "run_id": run_id,
                        "pid": pid, "start_ticks": ticks, "status": "old_queue_terminal",
                        "finish_job": approved["finish_job"]}
    if any(observed_monitor.get(key) != value for key, value in expected_monitor.items()):
        raise ValueError("monitor status does not confirm this terminated queue")
    with idle_queue(project, run_dir, run_id, pid, ticks) as state:
        verify_ledger(project, ledger)
        jobs = remaining_jobs(state, observed_monitor, run_dir)
        files = proposed_files(project, baseline, development, ledger)
        if files != approved["files"]:
            raise ValueError("development source changed after review")
        if (state["source_paths"] != approved["source_paths"]
                or state.get("auxiliary_paths") != approved["auxiliary_paths"]
                or state["minimum_free_gib"] != approved["minimum_free_gib"]):
            raise ValueError("old queue settings differ from the reviewed source manifest")
        interpreters: dict[str, set[str]] = {}
        for job in state["jobs"].values():
            if job.get("python"):
                interpreters.setdefault(job["environment"], set()).add(job["python"])
        if any(len(values) != 1 for values in interpreters.values()):
            raise ValueError("old queue used inconsistent Python paths for the same environment")
        python_llava = next(iter(interpreters.get("llava", ())), None)
        python_t437 = next(iter(interpreters.get("llava-t437", ())), None)
        python_modalprompt = next(iter(interpreters.get("llava-modalprompt-ds126", ())), None)
        new_plan = None
        argv = None
        if jobs:
            new_plan = make_plan(project, Path(state["source_paths"]), new_run_id, python_llava, python_t437,
                                 gpus, port, float(state["minimum_free_gib"]),
                                 Path(state["auxiliary_paths"]) if state.get("auxiliary_paths") else None,
                                 "four_gpu_validation", jobs, run_dir, receipt_path,
                                 python_modalprompt=python_modalprompt)
            if new_plan["roots"] != approved["new_roots"]:
                raise ValueError("new queue roots differ from the reviewed source manifest")
            for name in ("model_root", "data_root"):
                if new_plan["roots"][name] != state["roots"][name]:
                    raise ValueError(f"new {name} differs from the old queue")
            for name in ("ckpt_root", "results_root", "logs_root"):
                if Path(new_plan["roots"][name]).parent != Path(state["roots"][name]).parent:
                    raise ValueError(f"new {name} does not share the old queue's parent")
            if any(Path(new_plan["roots"][name]).exists() for name in ("ckpt_root", "results_root", "logs_root")):
                raise FileExistsError("a new queue output root already exists")
            argv = [python_llava or python_t437 or python_modalprompt or sys.executable,
                    "-m", "mcitlib.validation_batch", "execute", "--paths", state["source_paths"],
                    "--run-id", new_run_id, "--gpus", gpus, "--master-port", str(port),
                    "--min-free-gib", str(state["minimum_free_gib"]), "--execution-profile", "four_gpu_validation",
                    "--previous-run", str(run_dir), "--transition-receipt", str(receipt_path)]
            if python_llava:
                argv += ["--python-llava", python_llava]
            if python_t437:
                argv += ["--python-t437", python_t437]
            if python_modalprompt:
                argv += ["--python-modalprompt", python_modalprompt]
            if state.get("auxiliary_paths"):
                argv += ["--auxiliary-paths", state["auxiliary_paths"]]
            for job in new_plan["jobs"]:
                argv += ["--job", job.key]
        manifest = {"schema_version": 1, "created_at": now(), "runtime_project": str(project),
                    "approved_manifest": str(approved_path), "approved_manifest_sha256": approved_sha256,
                    "development_project": str(development), "baseline_snapshot": str(baseline),
                    "snapshot": str(snapshot), "snapshot_sha256": digest(snapshot),
                    "old_run_dir": str(run_dir), "old_run_id": run_id, "old_pid": pid, "old_start_ticks": ticks,
                    "old_state_sha256": digest(run_dir / "state.json"), "monitor_status": str(monitor_status),
                    "monitor_status_sha256": digest(monitor_status), "receipt": str(receipt_path),
                    "new_run_id": new_run_id, "execution_profile": "four_gpu_validation", "gpus": gpus,
                    "jobs": [job.key for job in new_plan["jobs"]] if new_plan else [], "files": files,
                    "new_plan": plan_description(new_plan) if new_plan else None, "new_queue_argv": argv,
                    "new_queue_shell": shlex.join(argv) if argv else None}
        write_new(output, manifest)
        return {"status": "prepared", "manifest": str(output), "manifest_sha256": digest(output),
                "receipt": str(receipt_path), "files": len(files), "jobs": manifest["jobs"],
                "new_queue_argv": argv, "new_queue_shell": manifest["new_queue_shell"]}


def stage_file(target: Path, content: bytes, mode: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.handoff-", dir=target.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(mode)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def apply(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    if digest(plan_path) != expected_sha256:
        raise ValueError("exact handoff plan SHA-256 does not match")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    approved_path = Path(plan["approved_manifest"]).resolve()
    approved = approved_manifest(approved_path, plan["approved_manifest_sha256"])
    project = Path(plan["runtime_project"]).resolve()
    development = Path(plan["development_project"]).resolve()
    baseline = Path(plan["baseline_snapshot"]).resolve()
    run_dir = Path(plan["old_run_dir"]).resolve()
    snapshot = Path(plan["snapshot"]).resolve()
    monitor_path = Path(plan["monitor_status"]).resolve()
    receipt_path = Path(plan["receipt"]).resolve()
    if not receipt_path.is_relative_to(project / ".workspaces") or receipt_path.exists():
        raise FileExistsError("receipt location is invalid or already exists")
    for name in ("runtime_project", "development_project", "baseline_snapshot", "snapshot", "snapshot_sha256",
                 "old_run_dir", "old_run_id", "old_pid", "old_start_ticks", "new_run_id", "gpus", "receipt"):
        if plan.get(name) != approved.get(name):
            raise ValueError(f"exact handoff plan differs from source review: {name}")
    if plan["files"] != approved["files"] or digest(snapshot) != approved["snapshot_sha256"]:
        raise ValueError("source files or snapshot differ from the reviewed source manifest")
    if digest(monitor_path) != plan["monitor_status_sha256"]:
        raise ValueError("monitor status changed since exact planning")
    ledger = json.loads(snapshot.read_text(encoding="utf-8"))
    monitor_state = json.loads(monitor_path.read_text(encoding="utf-8"))
    staged: list[tuple[Path, Path, Path | None, dict[str, Any]]] = []
    replaced: list[tuple[Path, Path | None]] = []
    with idle_queue(project, run_dir, plan["old_run_id"], plan["old_pid"], plan["old_start_ticks"]) as state:
        if digest(run_dir / "state.json") != plan["old_state_sha256"]:
            raise ValueError("old queue state changed since exact planning")
        if remaining_jobs(state, monitor_state, run_dir) != plan["jobs"]:
            raise ValueError("remaining jobs changed since exact planning")
        verify_ledger(project, ledger)
        if proposed_files(project, baseline, development, ledger) != approved["files"]:
            raise ValueError("development source changed after review")
        try:
            for item in plan["files"]:
                relative = source_path(item["path"])
                source, target = safe_join(development, relative), safe_join(project, relative)
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != item["source_sha256"]:
                    raise ValueError(f"source changed while staging: {relative}")
                before = item["baseline_sha256"]
                if before is None:
                    if target.exists():
                        raise FileExistsError(f"new source target already exists: {relative}")
                    backup = None
                    mode = source.stat().st_mode & 0o777
                else:
                    if not target.is_file() or digest(target) != before:
                        raise ValueError(f"runtime target changed while staging: {relative}")
                    mode = target.stat().st_mode & 0o777
                    backup = stage_file(target, target.read_bytes(), mode)
                try:
                    staged_file = stage_file(target, data, mode)
                except BaseException:
                    if backup is not None:
                        backup.unlink(missing_ok=True)
                    raise
                staged.append((target, staged_file, backup, item))
            for target, staged_file, backup, item in staged:
                before = item["baseline_sha256"]
                safe_join(project, source_path(item["path"]))
                if (before is None and target.exists()) or (before is not None and digest(target) != before):
                    raise ValueError(f"runtime target changed before replacement: {item['path']}")
                if before is None:
                    # link(2) is exclusive: never replace an unrecorded new target.
                    os.link(staged_file, target)
                    replaced.append((target, backup))
                    staged_file.unlink()
                else:
                    os.replace(staged_file, target)
                    replaced.append((target, backup))
            receipt = {"schema_version": 1, "status": "applied", "applied_at": now(),
                       "runtime_project": str(project), "previous_run": str(run_dir), "old_run_id": plan["old_run_id"],
                       "new_run_id": plan["new_run_id"], "execution_profile": plan["execution_profile"],
                       "gpus": plan["gpus"], "jobs": plan["jobs"], "manifest": str(plan_path),
                       "manifest_sha256": expected_sha256, "approved_manifest": str(approved_path),
                       "approved_manifest_sha256": plan["approved_manifest_sha256"], "snapshot": str(snapshot),
                       "snapshot_sha256": plan["snapshot_sha256"], "files": plan["files"]}
            write_new(receipt_path, receipt)
        except BaseException:
            for target, backup in reversed(replaced):
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise
        finally:
            for _, staged_file, backup, _ in staged:
                staged_file.unlink(missing_ok=True)
                if backup is not None:
                    backup.unlink(missing_ok=True)
    return {"status": "applied", "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path),
            "jobs": plan["jobs"], "new_queue_argv": plan["new_queue_argv"]}


def go(approved_path: Path, approved_sha256: str, *, poll: float = 0.5) -> dict[str, Any]:
    """Use one frozen source review through monitor, installation, and queue launch."""
    approved_path = approved_path.resolve()
    approved = approved_manifest(approved_path, approved_sha256)
    project = Path(approved["runtime_project"]).resolve()
    snapshot = Path(approved["snapshot"]).resolve()
    baseline = Path(approved["baseline_snapshot"]).resolve()
    development = Path(approved["development_project"]).resolve()
    run_dir = Path(approved["old_run_dir"]).resolve()
    execution_status = approved_path.with_name(approved_path.stem + ".execution.json")
    if execution_status.exists():
        raise FileExistsError("execution status already exists; inspect the earlier attempt")
    status: dict[str, Any] = {"approved_manifest": str(approved_path), "approved_manifest_sha256": approved_sha256}
    write_new(execution_status, {**status, "phase": "starting", "updated_at": now()})

    def update(phase: str, **details: Any) -> None:
        status.update(phase=phase, updated_at=now(), **details)
        atomic_json(execution_status, status)

    update("verifying_source")
    try:
        if any(name in os.environ for name in ROOT_VARIABLES):
            raise ValueError("clear MCITLIB_*_ROOT environment overrides before executing the handoff")
        if digest(snapshot) != approved["snapshot_sha256"]:
            raise ValueError("snapshot ledger changed after source review")
        ledger = json.loads(snapshot.read_text(encoding="utf-8"))
        verify_ledger(project, ledger)
        if proposed_files(project, baseline, development, ledger) != approved["files"]:
            raise ValueError("development source changed after review")
        update("monitoring_old_queue")
        monitor(project, run_dir, approved["old_run_id"], approved["old_pid"], approved["old_start_ticks"],
                approved["finish_job"], Path(approved["monitor_status"]), Path(approved["monitor_log"]), poll=poll)
        update("waiting_for_old_exit")
        deadline = time.monotonic() + 60
        while True:
            try:
                with idle_queue(project, run_dir, approved["old_run_id"], approved["old_pid"],
                                approved["old_start_ticks"]):
                    break
            except ValueError as error:
                if str(error) not in {"old queue process has not exited", "old queue lock is still held"}:
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError("old queue did not exit and release its lock") from error
                time.sleep(poll)
        update("preparing_exact_jobs")
        exact = prepare(project, run_dir, approved["old_run_id"], approved["old_pid"], approved["old_start_ticks"],
                        snapshot, Path(approved["monitor_status"]), approved["new_run_id"],
                        Path(approved["receipt"]), Path(approved["plan_path"]), approved_path, approved_sha256,
                        gpus=approved["gpus"], port=approved["master_port"])
        update("installing_reviewed_source", exact_plan=exact["manifest"], exact_plan_sha256=exact["manifest_sha256"])
        result = apply(Path(exact["manifest"]), exact["manifest_sha256"])
        update("source_installed", receipt=result["receipt"], receipt_sha256=result["receipt_sha256"],
               selected_jobs=result["jobs"])
        argv = result["new_queue_argv"]
        if not argv:
            update("no_jobs_remaining")
            return status
        env = {key: value for key, value in os.environ.items()
               if key not in set(ROOT_VARIABLES) | {"PYTHONPATH", "PYTHONHOME", "MCITLIB_NO_VENDORED_PATH"}}
        env.update(CUDA_VISIBLE_DEVICES=approved["gpus"], PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        launch_log = Path(approved["launch_log"])
        launch_log.parent.mkdir(parents=True, exist_ok=True)
        with launch_log.open("x", encoding="utf-8") as handle:
            child = subprocess.Popen(argv, cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        info = proc_info(child.pid)
        update("queue_launching", new_pid=child.pid, new_start_ticks=info[2] if info else None,
               launch_log=str(launch_log))
        state_path = Path(approved["new_roots"]["logs_root"]) / "state.json"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if state_path.is_file():
                new_state = json.loads(state_path.read_text(encoding="utf-8"))
                if (new_state.get("run_id") != approved["new_run_id"]
                        or new_state.get("execution_profile") != approved["execution_profile"]
                        or new_state.get("gpus") != approved["gpus"]
                        or new_state.get("previous_run") != str(run_dir)
                        or new_state.get("transition_receipt") != result["receipt"]
                        or list(new_state.get("jobs", {})) != result["jobs"]
                        or new_state.get("control", {}).get("pid") != child.pid):
                    raise RuntimeError("new queue state does not match the approved launch")
                update("queue_started", new_state=str(state_path), new_start_ticks=new_state["control"]["start_ticks"])
                return status
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"new queue exited before recording its state (exit {code}); see {launch_log}")
            time.sleep(0.1)
        raise RuntimeError(f"new queue did not record its state within 30 seconds; inspect {launch_log}")
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    review = actions.add_parser("preview", help="freeze and inspect a read-only source and handoff plan")
    review.add_argument("--project", type=Path, required=True)
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--run-id", required=True)
    review.add_argument("--pid", type=int, required=True)
    review.add_argument("--start-ticks", type=int, required=True)
    review.add_argument("--finish-job", required=True, metavar="METHOD/BENCHMARK")
    review.add_argument("--snapshot", type=Path, required=True)
    review.add_argument("--new-run-id", required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--monitor-status", type=Path, required=True)
    review.add_argument("--monitor-log", type=Path, required=True)
    review.add_argument("--plan", type=Path, required=True)
    review.add_argument("--receipt", type=Path, required=True)
    review.add_argument("--launch-log", type=Path, required=True)
    review.add_argument("--master-port", type=int, default=9001)
    execute = actions.add_parser("go", help="perform the handoff described by the reviewed source manifest")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--sha256", required=True)
    execute.add_argument("--poll", type=float, default=0.5)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "preview":
            result = preview(args.project, args.run_dir, args.run_id, args.pid, args.start_ticks, args.finish_job,
                             args.snapshot, args.new_run_id, args.output, args.monitor_status, args.monitor_log,
                             args.plan, args.receipt, args.launch_log, port=args.master_port)
        else:
            result = go(args.manifest, args.sha256, poll=args.poll)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"validation transition: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
