# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise queue handoff with real, disposable CPU processes."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcitlib.validation_batch import make_plan, proc_info, verify_transition_receipt
from tools.transition_validation_queue import apply, prepare, preview, process_stat, source_path


def _wait(predicate, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for CPU test subprocess")


def test_watchdog_releases_doctor_when_monitor_stalls(repo_root: Path, tmp_path: Path) -> None:
    doctor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    try:
        info = proc_info(doctor.pid)
        assert info is not None
        marker = tmp_path / "held"
        code = (
            "import time; from pathlib import Path; from tools.transition_validation_queue import doctor_hold\n"
            f"with doctor_hold({doctor.pid}, {info[2]}, timeout=0.75):\n"
            f"    Path({str(marker)!r}).write_text('held')\n"
            "    time.sleep(60)\n"
        )
        env = {**os.environ, "PYTHONPATH": str(repo_root), "PYTHONDONTWRITEBYTECODE": "1"}
        monitor = subprocess.Popen([sys.executable, "-c", code], cwd=tmp_path, env=env)
        _wait(marker.exists)
        assert process_stat(doctor.pid)[0] in {"T", "t"}
        assert monitor.wait(timeout=10) == -9
        _wait(lambda: process_stat(doctor.pid)[0] not in {"T", "t"})
        assert doctor.poll() is None
        assert unrelated.poll() is None
    finally:
        doctor.terminate()
        doctor.wait(timeout=10)
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_watchdog_releases_doctor_after_monitor_crash(repo_root: Path, tmp_path: Path) -> None:
    doctor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    try:
        info = proc_info(doctor.pid)
        assert info is not None
        code = ("import os; from tools.transition_validation_queue import doctor_hold\n"
                f"with doctor_hold({doctor.pid}, {info[2]}):\n"
                "    os._exit(17)\n")
        env = {**os.environ, "PYTHONPATH": str(repo_root), "PYTHONDONTWRITEBYTECODE": "1"}
        monitor = subprocess.Popen([sys.executable, "-c", code], cwd=tmp_path, env=env)
        assert monitor.wait(timeout=10) == 17
        _wait(lambda: process_stat(doctor.pid)[0] not in {"T", "t"})
        assert doctor.poll() is None
    finally:
        doctor.terminate()
        doctor.wait(timeout=10)


def test_frozen_source_review_and_receipt_refuse_drift_and_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "runtime"
    baseline = project / ".workspaces/baseline"
    development = project / ".workspaces/development"
    control = project / ".workspaces/control"
    for root in (project, baseline, development):
        (root / "mcitlib").mkdir(parents=True)
        (root / "configs/validation_3k_1k").mkdir(parents=True)
    (project / "configs").mkdir(exist_ok=True)
    (project / "configs/paths.yaml").write_text("{}\n")
    for root in (project, baseline):
        (root / "mcitlib/example.py").write_text("old\n")
        (root / "configs/validation_3k_1k/README.md").write_text("old documentation\n")
        (root / "configs/validation_3k_1k/manifest.json").write_text("original manifest\n")
        (root / "configs/validation_3k_1k/recipe.json").write_text("original recipe\n")
    (development / "mcitlib/example.py").write_text("new\n")
    (development / "mcitlib/new.py").write_text("added\n")
    (development / "configs/validation_3k_1k/README.md").write_text("new documentation\n")
    (development / "configs/validation_3k_1k/manifest.json").write_text("modified manifest\n")
    (development / "configs/validation_3k_1k/recipe.json").write_text("modified recipe\n")
    (development / "configs/paths.yaml").write_text("modified machine paths\n")
    old_plan = make_plan(project, project / "configs/paths.yaml", "old", sys.executable, None,
                         "0,1", 29513, 0, job_select=["lora_ft/ucit", "lora_ft/mllm_dcl",
                                                        "replay/ucit", "replay/mllm_dcl"])
    run_dir = old_plan["run_dir"]
    run_dir.mkdir(parents=True)
    old_plan["lock_file"].touch()
    jobs = {}
    for index, job in enumerate(old_plan["jobs"]):
        jobs[job.key] = {"status": ("completed", "not_run", "failed", "blocked")[index], "environment": job.env_name,
                         "python": job.python, "directory": str(job.directory)}
    old_state = {"project": str(project), "run_id": "old", "status": "stopped", "jobs": jobs,
                 "roots": old_plan["roots"], "source_paths": str(project / "configs/paths.yaml"),
                 "minimum_free_gib": 0, "auxiliary_paths": None,
                 "control": {"pid": 99999999, "start_ticks": 123, "child": None,
                             "lock_file": str(old_plan["lock_file"])}}
    (run_dir / "state.json").write_text(json.dumps(old_state))
    hashes = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
              for name in ("mcitlib/example.py", "configs/paths.yaml", "configs/validation_3k_1k/README.md",
                           "configs/validation_3k_1k/manifest.json", "configs/validation_3k_1k/recipe.json")}
    snapshot = control / "snapshot.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps({"runtime_project": str(project), "baseline_snapshot": str(baseline),
                                    "development_project": str(development), "runtime_hashes": hashes}))
    review = control / "review.json"
    status, log = control / "monitor.json", control / "monitor.jsonl"
    exact, receipt, launch = control / "exact.json", control / "receipt.json", control / "launch.log"
    reviewed = preview(project, run_dir, "old", 99999999, 123, "lora_ft/ucit", snapshot, "new", review,
                       status, log, exact, receipt, launch, port=29514)
    included = {item["path"] for item in json.loads(review.read_text())["files"]}
    assert "configs/validation_3k_1k/README.md" in included
    assert not included.intersection({"configs/paths.yaml", "configs/validation_3k_1k/manifest.json",
                                      "configs/validation_3k_1k/recipe.json"})
    status.write_text(json.dumps({"project": str(project), "run_dir": str(run_dir), "run_id": "old",
                                  "pid": 99999999, "start_ticks": 123, "status": "old_queue_terminal",
                                  "finish_job": "lora_ft/ucit", "stop_attempt": None}))
    args = (project, run_dir, "old", 99999999, 123, snapshot, status, "new", receipt, exact,
            review, reviewed["manifest_sha256"])
    (development / "mcitlib/example.py").write_text("changed after review\n")
    with pytest.raises(ValueError, match="development source changed after review"):
        prepare(*args, port=29514)
    (development / "mcitlib/example.py").write_text("new\n")
    (project / "mcitlib/unrecorded.py").write_text("unexpected\n")
    with pytest.raises(ValueError, match="unrecorded runtime source"):
        prepare(*args, port=29514)
    (project / "mcitlib/unrecorded.py").unlink()
    prepared = prepare(*args, port=29514)
    assert prepared["jobs"] == ["lora_ft/mllm_dcl"]
    (project / "mcitlib/new.py").write_text("unrelated existing file\n")
    with pytest.raises(ValueError, match="unrecorded runtime source|collides"):
        apply(exact, prepared["manifest_sha256"])
    assert (project / "mcitlib/example.py").read_text() == "old\n"
    assert (project / "mcitlib/new.py").read_text() == "unrelated existing file\n"
    (project / "mcitlib/new.py").unlink()
    link = os.link
    def fail_new_file(source, destination):
        if Path(destination) == project / "mcitlib/new.py":
            raise OSError("injected installation failure")
        return link(source, destination)
    with monkeypatch.context() as context:
        context.setattr(os, "link", fail_new_file)
        with pytest.raises(OSError, match="injected installation failure"):
            apply(exact, prepared["manifest_sha256"])
    assert (project / "mcitlib/example.py").read_text() == "old\n"
    assert (project / "configs/validation_3k_1k/README.md").read_text() == "old documentation\n"
    assert not (project / "mcitlib/new.py").exists() and not receipt.exists()
    result = apply(exact, prepared["manifest_sha256"])
    assert result["status"] == "applied"
    assert (project / "mcitlib/example.py").read_text() == "new\n"
    assert (project / "mcitlib/new.py").read_text() == "added\n"
    assert (project / "configs/validation_3k_1k/README.md").read_text() == "new documentation\n"
    assert (project / "configs/validation_3k_1k/manifest.json").read_text() == "original manifest\n"
    assert (project / "configs/validation_3k_1k/recipe.json").read_text() == "original recipe\n"
    assert (project / "configs/paths.yaml").read_text() == "{}\n"
    next_plan = make_plan(project, project / "configs/paths.yaml", "new", sys.executable, None,
                          "0,1,2,3", 29514, 0, execution_profile="four_gpu_validation",
                          job_select=["lora_ft/mllm_dcl"], previous_run=run_dir, transition_receipt=receipt)
    verify_transition_receipt(next_plan)
    (project / "mcitlib/new.py").write_text("changed after install\n")
    with pytest.raises(ValueError, match="differs from its receipt"):
        verify_transition_receipt(next_plan)
    (project / "mcitlib/new.py").write_text("added\n")
    (project / "configs/paths.yaml").write_text("data_root: elsewhere\n")
    with pytest.raises(ValueError, match="differs from its receipt"):
        verify_transition_receipt(next_plan)


def test_only_the_specific_sample_readme_is_allowed_for_synchronization() -> None:
    name = "configs/validation_3k_1k/README.md"
    assert source_path(name) == Path(name)
    for protected in ("configs/paths.yaml", "configs/validation_3k_1k/manifest.json",
                      "configs/validation_3k_1k/recipe.json", "configs/validation_3k_1k/paths.yaml",
                      "configs/validation_3k_1k/nested/README.md", "configs/validation_other/README.md",
                      "data/validation_3k_1k/RS/test.json"):
        with pytest.raises(ValueError):
            source_path(protected)


def test_detached_handoff_applies_reviewed_bytes_and_runs_only_remaining_jobs(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("MODEL", "DATA", "CKPT", "RESULTS", "LOGS"):
        monkeypatch.delenv(f"MCITLIB_{name}_ROOT", raising=False)
    project = tmp_path / "runtime"
    project.mkdir()
    shutil.copytree(repo_root / "mcitlib", project / "mcitlib", ignore=shutil.ignore_patterns("__pycache__"))
    (project / "third_party").mkdir()
    (project / "configs").mkdir()
    (project / "configs/paths.yaml").write_text("{}\n")
    baseline, development = project / ".workspaces/baseline", project / ".workspaces/development"
    for root in (baseline, development):
        shutil.copytree(project / "mcitlib", root / "mcitlib")
    (development / "mcitlib/handoff_marker.py").write_text("# reviewed source\n")
    fake = project / "fake-python"
    fake.write_text(f"""#!{sys.executable}
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if 'mcitlib.validation_batch' in args and 'execute' in args:
    os.execv(sys.executable, [sys.executable, *args])
def option(name): return args[args.index(name) + 1]
project = Path({str(project)!r})
method, benchmark = option('--method'), option('--benchmark')
old = '/old/' in option('--paths')
if 'doctor' in args:
    if old and benchmark == 'mllm_dcl': time.sleep(60)
    failed = not old and method == 'lora_ft' and benchmark == 'mllm_dcl'
    print(json.dumps({{'failed': failed, 'checks': [{{'status': 'FAIL', 'name': 'fixture', 'detail': 'no'}}] if failed else []}}))
elif '_worker' in args:
    if old and method == 'lora_ft' and benchmark == 'ucit':
        (project / 'old-worker').write_text('started')
        while not (project / 'release').exists(): time.sleep(0.05)
    elif old:
        (project / 'unexpected-old-worker').write_text('started')
    else:
        (project / (method + '-' + benchmark + '-new-worker')).write_text('started')
elif 'metrics' in args:
    Path(option('--output')).write_text('{{}}')
""", encoding="utf-8")
    fake.chmod(0o755)
    controller = project / "controller.py"
    controller.write_text(f"""from pathlib import Path
from mcitlib.validation_batch import Queue, make_plan
project = Path({str(project)!r})
plan = make_plan(project, project / 'configs/paths.yaml', 'old', {str(fake)!r}, None, '0,1', 29513, 0,
                 job_select=['lora_ft/ucit', 'lora_ft/mllm_dcl', 'replay/ucit'])
raise SystemExit(Queue(plan).execute())
""", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
    queue = subprocess.Popen([sys.executable, str(controller), "-m", "mcitlib.validation_batch", "execute"],
                             cwd=project, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    transition = None
    try:
        _wait(lambda: (project / "old-worker").exists())
        info = proc_info(queue.pid)
        assert info is not None
        run_dir, control = project / "logs/old", project / ".workspaces/control"
        control.mkdir(parents=True)
        hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (project / "mcitlib").rglob("*") if path.is_file()}
        hashes["configs/paths.yaml"] = hashlib.sha256((project / "configs/paths.yaml").read_bytes()).hexdigest()
        snapshot = control / "snapshot.json"
        snapshot.write_text(json.dumps({"runtime_project": str(project), "baseline_snapshot": str(baseline),
                                        "development_project": str(development), "runtime_hashes": hashes}))
        review = control / "review.json"
        status, log = control / "monitor.json", control / "monitor.jsonl"
        exact, receipt, launch = control / "exact.json", control / "receipt.json", control / "launch.log"
        reviewed = preview(project, run_dir, "old", queue.pid, info[2], "lora_ft/ucit", snapshot, "new", review,
                           status, log, exact, receipt, launch, port=29514)
        transition = subprocess.Popen(
            [sys.executable, "-m", "tools.transition_validation_queue", "go", "--manifest", str(review),
             "--sha256", reviewed["manifest_sha256"], "--poll", "0.05"], cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _wait(status.exists)
        assert queue.poll() is None and transition.poll() is None
        (project / "release").write_text("done")
        assert transition.wait(timeout=25) == 0, transition.stderr.read().decode()
        assert queue.wait(timeout=10) == 1, queue.stderr.read().decode()
        launch_state = json.loads(transition.stdout.read().decode())
        assert launch_state["phase"] == "queue_started"
        assert launch_state["selected_jobs"] == ["lora_ft/mllm_dcl", "replay/ucit"]
        new_state = project / "logs/new/state.json"
        _wait(lambda: json.loads(new_state.read_text())["status"] in {"completed", "completed_with_failures"})
        state = json.loads(new_state.read_text())
        assert state["execution_profile"] == "four_gpu_validation" and state["gpus"] == "0,1,2,3"
        assert state["previous_run"] == str(run_dir) and state["transition_receipt"] == str(receipt)
        assert state["jobs"]["lora_ft/mllm_dcl"]["status"] == "blocked"
        assert state["jobs"]["replay/ucit"]["status"] == "completed"
        assert not (project / "unexpected-old-worker").exists()
        assert not (project / "lora_ft-mllm_dcl-new-worker").exists()
        assert (project / "replay-ucit-new-worker").exists()
        assert (project / "mcitlib/handoff_marker.py").read_text() == "# reviewed source\n"
    finally:
        if transition and transition.poll() is None:
            transition.kill()
            transition.wait(timeout=10)
        if queue.poll() is None:
            queue.terminate()
            queue.wait(timeout=10)


def test_monitor_waits_for_worker_then_stops_only_next_doctor(repo_root: Path, tmp_path: Path) -> None:
    profile = tmp_path / "paths.yaml"
    profile.write_text("{}\n", encoding="utf-8")
    fake = tmp_path / "fake-python"
    fake.write_text(f"""#!{sys.executable}
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
def option(name): return args[args.index(name) + 1]
project = Path({str(tmp_path)!r})
benchmark = option('--benchmark')
if 'doctor' in args:
    if benchmark == 'mllm_dcl':
        (project / 'second-doctor').write_text('started')
        time.sleep(60)
    print(json.dumps({{'failed': False, 'checks': []}}))
elif '_worker' in args:
    if benchmark != 'ucit':
        (project / 'unexpected-second-worker').write_text('started')
        sys.exit(3)
    (project / 'first-worker').write_text('started')
    while not (project / 'release-first').exists(): time.sleep(0.05)
elif 'metrics' in args:
    Path(option('--output')).write_text('{{}}')
""", encoding="utf-8")
    fake.chmod(0o755)
    controller = tmp_path / "controller.py"
    controller.write_text(f"""from pathlib import Path
from mcitlib.validation_batch import Queue, make_plan
project = Path({str(tmp_path)!r})
plan = make_plan(project, project / 'paths.yaml', 'old', {str(fake)!r}, None, '0,1', 29513, 0,
                 job_select=['lora_ft/ucit', 'lora_ft/mllm_dcl'])
raise SystemExit(Queue(plan).execute())
""", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(repo_root), "PYTHONDONTWRITEBYTECODE": "1"}
    queue = subprocess.Popen([sys.executable, str(controller), "-m", "mcitlib.validation_batch", "execute"],
                             cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    watcher = None
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    run_dir = tmp_path / "logs/old"
    try:
        _wait(lambda: (tmp_path / "first-worker").is_file())
        identity = proc_info(queue.pid)
        assert identity is not None
        status = tmp_path / "handoff/status.json"
        log = tmp_path / "handoff/events.jsonl"
        monitor_code = (
            "from pathlib import Path; from tools.transition_validation_queue import monitor; "
            f"monitor(Path({str(tmp_path)!r}), Path({str(run_dir)!r}), 'old', {queue.pid}, {identity[2]}, "
            f"'lora_ft/ucit', Path({str(status)!r}), Path({str(log)!r}), poll=0.05)"
        )
        watcher = subprocess.Popen([sys.executable, "-c", monitor_code], cwd=tmp_path, env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _wait(status.exists)
        time.sleep(0.5)
        assert queue.poll() is None and watcher.poll() is None and unrelated.poll() is None
        state = json.loads((run_dir / "state.json").read_text())
        assert state["jobs"]["lora_ft/ucit"]["phase"] == "sequence"
        (tmp_path / "release-first").write_text("done")
        assert watcher.wait(timeout=15) == 0, watcher.stderr.read().decode()
        assert queue.wait(timeout=15) == 1, queue.stderr.read().decode()
        state = json.loads((run_dir / "state.json").read_text())
        result = json.loads(status.read_text())
        assert state["status"] == "stopped"
        assert state["jobs"]["lora_ft/ucit"]["status"] == "completed"
        assert state["jobs"]["lora_ft/mllm_dcl"]["status"] == "stopped"
        assert result["stop_attempt"]["job"] == "lora_ft/mllm_dcl"
        assert result["stop_attempt"]["phase"] == "doctor:train"
        assert not (tmp_path / "unexpected-second-worker").exists()
        assert unrelated.poll() is None
    finally:
        if watcher and watcher.poll() is None:
            watcher.kill()
            watcher.wait(timeout=10)
        if queue.poll() is None:
            queue.terminate()
            queue.wait(timeout=10)
        unrelated.terminate()
        unrelated.wait(timeout=10)
