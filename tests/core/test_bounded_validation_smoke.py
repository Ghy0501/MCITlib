# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Safety properties of the explicit, bounded GPU smoke planner."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tools import bounded_validation_smoke as smoke


def _minimal_plan(tmp_path: Path):
    sample_source = tmp_path / "source.json"
    sample_source.write_text("[]", encoding="utf-8")
    subset = tmp_path / "subset.json"
    subset.write_text("[]", encoding="utf-8")
    item = {"source": str(sample_source), "source_sha256": smoke._sha(sample_source),
            "subset": str(subset), "subset_sha256": smoke._sha(subset)}
    root = tmp_path.resolve()
    job_dir = root / "job"
    train = {"argv": ["deepspeed", "--include", "localhost:0,1,2,3", "--max_steps", "2",
                      "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
                      "--save_strategy", "no", "--logging_nan_inf_filter", "False",
                      "--data_path", str(subset), "--output_dir", str(root / "checkpoint")],
             "cwd": str(root), "env": {}, "kind": "train", "task_id": 1,
             "sequence": "train", "metadata": {}}
    stage = {"training_sample": item, "evaluation_sample": item, "commands": [train], "task": 1}
    plan = {"project": str(root), "run_dir": str(root), "limits": {"max_steps": 2},
            "jobs": {"moelora/ucit": {"job_dir": str(job_dir), "stages": [stage]}}}
    return plan, plan["jobs"]["moelora/ucit"]


def test_explicit_jobs_and_task_limit() -> None:
    assert smoke._selected(["hide/ucit", "modalprompt/mllm_dcl"]) == ["hide/ucit", "modalprompt/mllm_dcl"]
    assert len(smoke._selected(None)) == 16
    with pytest.raises(ValueError, match="unique"):
        smoke._selected(["hide/ucit", "hide/ucit"])
    with pytest.raises(ValueError, match="unique"):
        smoke._selected(["keeplora/ucit"])


def test_plan_checks_data_and_resource_bounds(tmp_path: Path) -> None:
    plan, job = _minimal_plan(tmp_path)
    assert len(smoke._commands(plan, job, 1)) == 1
    train = job["stages"][0]["commands"][0]
    position = train["argv"].index("--max_steps") + 1
    train["argv"][position] = "4"
    with pytest.raises(ValueError, match="violates"):
        smoke._commands(plan, job, 1)
    train["argv"][position] = "2"
    train["argv"][train["argv"].index("--output_dir") + 1] = str(tmp_path.parent / "outside")
    with pytest.raises(ValueError, match="escapes"):
        smoke._commands(plan, job, 1)
    train["argv"][train["argv"].index("--output_dir") + 1] = str(tmp_path / "checkpoint")
    (tmp_path / "subset.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        smoke._commands(plan, job, 1)


def test_doctor_failure_never_executes_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, job = _minimal_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(smoke, "_read_plan", lambda path: plan)
    monkeypatch.setattr(smoke, "_doctor", lambda *args: {"failed": True, "checks": []})
    import mcitlib.runner

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU execution was attempted after doctor failure")

    monkeypatch.setattr(mcitlib.runner, "execute_commands", forbidden)
    assert smoke._worker(plan_path, "moelora/ucit", 1, "execute", False, 20) == 1
    report = Path(job["job_dir"]) / "doctor-1-gpu.json"
    assert json.loads(report.read_text())["failed"]


def test_generation_target_is_restricted(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.bounded_eval_entry import _target

    monkeypatch.setenv("MCITLIB_SMOKE_EVAL_TARGET", "mcitlib.methods._hide.eval_runners.model_others")
    assert _target().endswith("model_others")
    monkeypatch.setenv("MCITLIB_SMOKE_EVAL_TARGET", "os.system")
    with pytest.raises(ValueError, match="invalid"):
        _target()


def test_timeout_kills_only_the_spawned_process_group(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    with (tmp_path / "job.log").open("w", encoding="utf-8") as log:
        code = smoke._run_isolated([sys.executable, "-c", script, str(child_pid)],
                                   os.environ.copy(), str(tmp_path), log, 0.5)
    assert code == 124 and child_pid.is_file()
    status = Path(f"/proc/{int(child_pid.read_text())}/stat")
    try:
        state = status.read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return  # already reaped
    assert state in {"X", "Z"}  # dead or zombie; never accept a live child
