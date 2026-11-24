# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise queue isolation, real child processes, ordering and safe controls without GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from mcitlib.config.schema import Command
from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands, execute_commands
from mcitlib.validation_batch import Queue, cleanup_group, make_plan, proc_info, stop_queue, worker_commands


def profile(project: Path) -> Path:
    path = project / "paths.yaml"
    path.write_text(yaml.safe_dump({"model_root": "./models", "data_root": "./data",
                                    "ckpt_root": "./ckpts/validation_3k_1k",
                                    "results_root": "./results/validation_3k_1k",
                                    "logs_root": "./logs/validation_3k_1k"}), encoding="utf-8")
    return path


def fake_python(project: Path, *, hang: bool = False) -> Path:
    executable = project / "fake-python"
    executable.write_text(f'''#!{sys.executable}
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
method, benchmark = option("--method"), option("--benchmark")
if "doctor" in args:
    fail = method == "lora_ft" and benchmark == "ucit"
    print("dependency startup message")
    print(json.dumps({{"failed": fail, "checks": [{{"status": "FAIL" if fail else "OK", "name": "test", "detail": "doctor failure"}}]}}))
    sys.exit(0)
if "_worker" in args:
    directory = Path(option("--run-dir"))
    (directory / "worker-started").write_text("started")
    log = directory / "task1.log"
    log.write_text("synthetic training failure\\n")
    blocked_disk = method == "disco" and benchmark == "ucit"
    event = {{"status": "blocked_disk" if blocked_disk else "failed" if benchmark == "mllm_dcl" and method == "lora_ft" else "started", "task": 1, "kind": "train", "index": 0, "log": str(log)}}
    if blocked_disk:
        event["error"] = "insufficient free disk space"
    (directory / "events.jsonl").write_text(json.dumps(event) + "\\n")
    if {hang!r}:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        (directory / "child.pid").write_text(str(child.pid))
        time.sleep(300)
    if benchmark == "mllm_dcl" and method == "lora_ft":
        sys.exit(3)
    if blocked_disk:
        sys.exit(1)
    log.write_text("synthetic evaluation completed\\n")
    with (directory / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({{"status": "completed", "task": 1, "kind": "eval", "index": 1, "log": str(log)}}) + "\\n")
    sys.exit(0)
if "metrics" in args:
    Path(option("--output")).write_text('{{"metrics": {{}}}}')
    print("metrics complete")
    sys.exit(0)
sys.exit(4)
''', encoding="utf-8")
    executable.chmod(0o755)
    return executable


def plan_for(project: Path, run_id: str, executable: Path) -> dict:
    return make_plan(project, profile(project), run_id, str(executable), None, "0,1", 29511, 0)


def test_preview_has_all_jobs_and_complete_doctors(tmp_path: Path) -> None:
    plan = plan_for(tmp_path, "preview", tmp_path / "python")
    assert len(plan["jobs"]) == 34  # 17 methods, two benchmarks each
    assert all(Path(plan["roots"][key]).is_relative_to(tmp_path) for key in ("logs_root", "ckpt_root", "results_root"))
    assert not plan["run_dir"].exists()
    assert len({job.profile["ckpt_root"] for job in plan["jobs"]}) == 34
    assert len({job.profile["results_root"] for job in plan["jobs"]}) == 34
    for job in plan["jobs"]:
        for _, argv in job.doctors:
            assert "--validation-fixes" in argv and "--json" in argv
            assert "--skip-gpu" not in argv and "--skip-entrypoints" not in argv
    assert len(next(job for job in plan["jobs"] if job.method == "mrlora").doctors) == 2


def test_auxiliary_mapping_overlay_replaces_only_exact_inputs(tmp_path: Path) -> None:
    source = profile(tmp_path)
    base = yaml.safe_load(source.read_text())
    base["path_mappings"] = {"ucit": [
        {"from": "${data_root}/UCIT/A/replay.json", "to": "${repo_root}/missing.json"},
        {"from": "${data_root}/UCIT/A/train.json", "to": "${repo_root}/sample.json"},
    ]}
    source.write_text(yaml.safe_dump(base))
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"path_mappings": {"ucit": [
        {"from": "${data_root}/UCIT/A/replay.json", "to": "${repo_root}/data/validation_aux/replay.json"}]}}))
    plan = make_plan(tmp_path, source, "overlay", "/none", None, "0,1", 29511, 0, overlay)
    mappings = plan["jobs"][0].profile["path_mappings"]["ucit"]
    assert len(mappings) == 2
    assert next(entry["to"] for entry in mappings if entry["from"].endswith("replay.json")) == \
        "${repo_root}/data/validation_aux/replay.json"
    assert next(entry["to"] for entry in mappings if entry["from"].endswith("train.json")) == \
        "${repo_root}/sample.json"
    assert yaml.safe_load(source.read_text()) == base
    overlay.write_text(yaml.safe_dump({"path_mappings": {"ucit": [
        {"from": "${data_root}/UCIT", "to": "${repo_root}/data"}]}}))
    with pytest.raises(ValueError, match="logical replay/router"):
        make_plan(tmp_path, source, "bad", "/none", None, "0,1", 29511, 0, overlay)


def test_job_failures_do_not_stop_later_jobs(tmp_path: Path) -> None:
    fake = fake_python(tmp_path)
    plan = plan_for(tmp_path, "isolation", fake)
    jobs = plan["jobs"][:6]
    # An unconfigured environment and an executable with an invalid format are independent failures.
    jobs[2].python = None
    bad = tmp_path / "bad-executable"
    bad.write_text("not an executable format", encoding="utf-8")
    bad.chmod(0o755)
    jobs[4].python = str(bad)
    jobs[4].doctors = [(seq, [str(bad), *argv[1:]]) for seq, argv in jobs[4].doctors]
    plan["jobs"] = jobs
    queue = Queue(plan)
    assert queue.execute() == 1
    state = json.loads(queue.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed_with_failures"
    assert [state["jobs"][job.key]["status"] for job in jobs] == [
        "blocked", "failed", "blocked", "completed", "blocked", "completed"]
    assert not (jobs[0].directory / "worker-started").exists()
    assert state["jobs"][jobs[1].key]["step"]["task"] == 1
    assert "synthetic training failure" in state["jobs"][jobs[1].key]["error_tail"]
    assert "synthetic training failure" in Path(state["jobs"][jobs[1].key]["logs"]["training"]).read_text()
    assert (jobs[5].directory / "metrics.json").exists()
    assert "synthetic evaluation completed" in Path(state["jobs"][jobs[5].key]["logs"]["evaluation"]).read_text()
    assert state["summary"] == {"blocked": 3, "failed": 1, "completed": 2}
    assert Path(state["jobs"][jobs[4].key]["failure_log"]).is_file()


def test_disk_blocking_continues_to_next_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_python(tmp_path)
    plan = plan_for(tmp_path, "disk", fake)
    plan["jobs"] = plan["jobs"][2:4]
    plan["min_free_gib"] = 1
    import mcitlib.validation_batch as batch
    actual = batch.shutil.disk_usage
    calls = []
    def usage(path):
        calls.append(path)
        real = actual(path)
        return type(real)(real.total, real.used, 0 if len(calls) == 1 else 2**31)
    monkeypatch.setattr(batch.shutil, "disk_usage", usage)
    queue = Queue(plan)
    assert queue.execute() == 1
    assert [queue.state["jobs"][job.key]["status"] for job in plan["jobs"]] == ["blocked_disk", "completed"]


def test_disk_block_between_steps_prevents_next_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcitlib.validation_batch as batch

    paths_file = profile(tmp_path)
    job_dir = tmp_path / "logs/validation_3k_1k/run/jobs/disk"
    job_dir.mkdir(parents=True)
    first, second = tmp_path / "first", tmp_path / "second"
    commands = [
        Command([sys.executable, "-c", f"from pathlib import Path; p=Path({str(first)!r}); p.mkdir(); (p/'weight').write_text('ok')", "--output_dir", str(first)], kind="train", task_id=1),
        Command([sys.executable, "-c", f"from pathlib import Path; Path({str(second)!r}).write_text('bad')"], kind="train", task_id=2),
    ]
    monkeypatch.setattr(batch, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(batch, "worker_commands", lambda *args: commands)
    checks = iter([None, "insufficient free disk space"])
    monkeypatch.setattr(batch, "disk_shortage", lambda *args: next(checks))
    args = argparse.Namespace(paths=paths_file, run_dir=job_dir, method="lora_ft", benchmark="ucit",
                              gpus="0,1", master_port=29511, min_free_gib=32)
    assert batch.run_worker(args) == 1
    events = [json.loads(line) for line in (job_dir / "events.jsonl").read_text().splitlines()]
    assert [(event["task"], event["status"]) for event in events] == [
        (1, "started"), (1, "completed"), (2, "blocked_disk")]
    assert first.exists() and not second.exists()


def test_disk_block_within_job_does_not_stop_later_job(tmp_path: Path) -> None:
    fake = fake_python(tmp_path)
    plan = plan_for(tmp_path, "step-disk", fake)
    plan["jobs"] = [job for job in plan["jobs"] if job.method == "disco"]
    queue = Queue(plan)
    assert queue.execute() == 1
    assert [queue.state["jobs"][job.key]["status"] for job in plan["jobs"]] == ["blocked_disk", "completed"]
    assert "insufficient free disk space" in queue.state["jobs"][plan["jobs"][0].key]["error"]


def test_worker_stops_after_failed_train_and_missing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcitlib.validation_batch as batch

    paths_file = profile(tmp_path)
    job_dir = tmp_path / "logs/validation_3k_1k/run/jobs/job"
    job_dir.mkdir(parents=True)
    completed = tmp_path / "first-ckpt"
    after = tmp_path / "must-not-run"
    commands = [
        Command([sys.executable, "-c", f"from pathlib import Path; p=Path({str(completed)!r}); p.mkdir(); (p/'weight').write_text('ok')", "--output_dir", str(completed)], kind="train", task_id=1),
        Command([sys.executable, "-c", "raise RuntimeError('task two broke')", "--output_dir", str(tmp_path / "task2")], kind="train", task_id=2),
        Command([sys.executable, "-c", f"from pathlib import Path; Path({str(after)!r}).write_text('bad')"], kind="train", task_id=3),
    ]
    monkeypatch.setattr(batch, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(batch, "worker_commands", lambda *args: commands)
    args = argparse.Namespace(paths=paths_file, run_dir=job_dir, method="lora_ft", benchmark="ucit",
                              gpus="0,1", master_port=29511)
    assert batch.run_worker(args) == 1
    events = [json.loads(line) for line in (job_dir / "events.jsonl").read_text().splitlines()]
    assert [(event["task"], event["status"]) for event in events] == [(1, "started"), (1, "completed"), (2, "started"), (2, "failed")]
    assert "task two broke" in Path(events[-1]["log"]).read_text()
    assert not after.exists()
    # A zero exit code without a checkpoint is also a failure, before task 2 starts.
    commands[0] = Command([sys.executable, "-c", "pass", "--output_dir", str(tmp_path / "absent")], kind="train", task_id=1)
    another = job_dir.parent / "missing-output"
    another.mkdir()
    args.run_dir = another
    assert batch.run_worker(args) == 1
    missing_events = [json.loads(line) for line in (another / "events.jsonl").read_text().splitlines()]
    assert missing_events[-1]["status"] == "failed"
    assert "missing or empty" in missing_events[-1]["error"]
    markers = list((another / ".mcitlib/steps").glob("*.json"))
    assert len(markers) == 1
    marker = json.loads(markers[0].read_text())
    assert marker["status"] == "failed" and marker["return_code"] == 0
    assert "missing or empty" in marker["validation_error"]
    assert not after.exists()


def test_modalprompt_interleaves_inference_and_repairs_only_task2(repo_root: Path, tmp_path: Path) -> None:
    paths = PathResolver.from_file(profile(tmp_path), repo_root=tmp_path)
    commands = worker_commands(repo_root, paths, "modalprompt", "ucit", "0,1", 29511)
    tasks = [command.task_id for command in commands]
    assert tasks == sorted(tasks)
    for task in range(1, 7):
        group = [command for command in commands if command.task_id == task]
        assert group[0].kind == "train" and all(item.kind == "eval" for item in group[1:])
        assert len(group) > 1
        argv = group[0].argv
        assert argv[argv.index("--num_train_epochs") + 1] == "10"
        assert argv[argv.index("--output_dir") + 1].endswith(f"Task{task}_llava")
    historical = build_sequence_commands(repo_root / "configs", "llava", "modalprompt", "ucit", task_id=2)
    train = next(command for command in historical if command.kind == "train")
    assert train.argv[train.argv.index("--output_dir") + 1].endswith("Task1_llava")


def test_doctor_and_worker_share_the_same_validation_repair(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcitlib.doctor as doctor

    captured = []
    for name in ("check_environment", "check_gpus", "check_vendored_imports", "check_inputs", "check_prerequisites"):
        monkeypatch.setattr(doctor, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(doctor, "check_dataflow", lambda report, commands, *args: captured.extend(commands))
    source = profile(tmp_path)
    report = doctor.run_doctor("modalprompt", "ucit", source, repo_root, skip_gpu=True,
                               skip_entrypoints=True, validation_fixes=True)
    assert not report.failed
    doctor_train = next(item for item in captured if item.kind == "train" and item.task_id == 2)
    paths = PathResolver.from_file(source, repo_root=repo_root)
    worker_train = next(item for item in worker_commands(repo_root, paths, "modalprompt", "ucit", "0,1", 29511)
                        if item.kind == "train" and item.task_id == 2)
    assert doctor_train.argv[doctor_train.argv.index("--output_dir") + 1] == \
        worker_train.argv[worker_train.argv.index("--output_dir") + 1]
    assert doctor_train.metadata["validation_fix"] == "modalprompt_ucit_task2_output_dir"


def test_required_checkpoint_files_are_checked(tmp_path: Path) -> None:
    from mcitlib.validation_batch import checkpoint_error

    (tmp_path / "trainer_state.json").write_text("{}")
    argv = ["mcitlib.backbones.llava.train_entry", "--output_dir", str(tmp_path), "--lora_enable", "True"]
    assert "adapter_model" in checkpoint_error(argv)
    for name in ("config.json", "non_lora_trainables.bin", "adapter_model.bin"):
        (tmp_path / name).write_text("contents")
    assert checkpoint_error(argv) is None


def test_pclr_checkpoint_requires_actual_rank_pool_state(tmp_path: Path) -> None:
    from mcitlib.validation_batch import checkpoint_error

    (tmp_path / 'config.json').write_text('{}')
    argv = ['mcitlib.backbones.llava.train_entry', '--output_dir', str(tmp_path), '--lora_enable', 'False']
    assert 'pclr.pt' in checkpoint_error(argv, {'MCITLIB_METHOD': 'pclr'})
    (tmp_path / 'pclr.pt').write_bytes(b'checkpoint')
    assert checkpoint_error(argv, {'MCITLIB_METHOD': 'pclr'}) is None


def test_deepspeed_uses_worker_interpreter(tmp_path: Path) -> None:
    package = tmp_path / "fake/deepspeed/launcher"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "runner.py").write_text("import sys; print(sys.executable)\n")
    command = Command(["deepspeed", "--sentinel"], env={"PYTHONPATH": str(tmp_path / "fake")})
    execute_commands([command], tmp_path / "logs/run", tmp_path)
    marker = next((tmp_path / "logs/run/.mcitlib/steps").glob("*.json"))
    record = json.loads(marker.read_text())
    assert record["command"]["argv"] == ["deepspeed", "--sentinel"]
    assert record["executed_argv"] == [sys.executable, "-m", "deepspeed.launcher.runner", "--sentinel"]
    assert sys.executable in Path(record["log"]).read_text()


def wait_until(test, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if test():
            return
        time.sleep(0.1)
    raise AssertionError("timed out waiting for child process")


def test_lock_and_stop_only_controlled_session(tmp_path: Path) -> None:
    fake = fake_python(tmp_path, hang=True)
    source = profile(tmp_path)
    plan = make_plan(tmp_path, source, "controlled", str(fake), None, "0,1", 29511, 0)
    # Pick an always-passing doctor and start an actual Queue in another process.
    job = plan["jobs"][2]
    script = tmp_path / "controller.py"
    script.write_text(f'''from pathlib import Path
from mcitlib.validation_batch import Queue, make_plan
p = make_plan(Path({str(tmp_path)!r}), Path({str(source)!r}), "controlled", {str(fake)!r}, None, "0,1", 29511, 0)
p["jobs"] = p["jobs"][2:3]
raise SystemExit(Queue(p).execute())
''')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    controller = subprocess.Popen([sys.executable, str(script), "-m", "mcitlib.validation_batch", "execute"],
                                  cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], cwd=tmp_path)
    try:
        child_file = job.directory / "child.pid"
        wait_until(child_file.exists)
        child_pid = int(child_file.read_text())
        other_plan = make_plan(tmp_path, source, "duplicate", str(fake), None, "0,1", 29512, 0)
        with pytest.raises(RuntimeError, match="another validation queue"):
            Queue(other_plan).execute()
        stop_queue(plan["run_dir"], tmp_path)
        assert controller.wait(timeout=15) == 1
        wait_until(lambda: proc_info(child_pid) is None)
        assert unrelated.poll() is None
        state = json.loads((plan["run_dir"] / "state.json").read_text())
        assert state["status"] == "stopped"
        assert state["jobs"][job.key]["status"] == "stopped"
        with pytest.raises(ValueError, match="not a running queue"):
            stop_queue(plan["run_dir"], tmp_path)
    finally:
        if controller.poll() is None:
            controller.terminate()
            controller.wait(timeout=10)
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_cleanup_ignores_reused_session_identity(tmp_path: Path) -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                                 cwd=tmp_path, start_new_session=True)
    try:
        identity = proc_info(unrelated.pid)
        assert identity is not None
        # Simulate the same numeric session ID being assigned to a newer process.
        cleanup_group(unrelated.pid, identity[2] - 1, {}, grace=0)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
