# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise actual queue orchestration using CPU-only child processes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from mcitlib.validation_batch import Queue, make_plan


def test_four_gpu_profile_and_precise_job_selection(tmp_path: Path) -> None:
    profile = tmp_path / "paths.yaml"
    profile.write_text("{}\n", encoding="utf-8")
    common = (tmp_path, profile, "selected", sys.executable, sys.executable)
    default = make_plan(*common, "0,1", 29513, 0)
    assert len(default["jobs"]) == 34
    dgg = next(job for job in default["jobs"] if job.key == "dgg/ucit")
    assert dgg.worker[dgg.worker.index("--gpus") + 1] == "0"
    assert "--dgg-sample-profile" in dgg.doctors[0][1]
    assert dgg.doctors[0][1][dgg.doctors[0][1].index("--gpus") + 1] == "0"
    with pytest.raises(ValueError, match="four-GPU"):
        make_plan(*common, "0,1,2,3", 29513, 0, execution_profile="four_gpu_validation",
                  job_select=["dgg/ucit"])
    assert all("--execution-profile" not in argv for job in default["jobs"] for _, argv in job.doctors)
    selected = make_plan(*common, "0,1,2,3", 29513, 0, execution_profile="four_gpu_validation",
                         job_select=["mrlora/mllm_dcl", "lora_ft/ucit"])
    assert [job.key for job in selected["jobs"]] == ["lora_ft/ucit", "mrlora/mllm_dcl"]
    assert selected["execution_profile"] == "four_gpu_validation"
    assert len(selected["jobs"][1].doctors) == 2
    for job in selected["jobs"]:
        for _, argv in job.doctors:
            assert argv[argv.index("--execution-profile") + 1] == "four_gpu_validation"
            assert argv[argv.index("--gpus") + 1] == "0,1,2,3"
            assert "--skip-gpu" not in argv and "--skip-entrypoints" not in argv
        assert job.worker[job.worker.index("--execution-profile") + 1] == "four_gpu_validation"
        assert job.worker[job.worker.index("--gpus") + 1] == "0,1,2,3"
    with pytest.raises(ValueError, match="exactly 2"):
        make_plan(*common, "0,1,2,3", 29513, 0)
    with pytest.raises(ValueError, match="exactly 4"):
        make_plan(*common, "0,1", 29513, 0, execution_profile="four_gpu_validation")
    with pytest.raises(ValueError, match="0,1,2,3"):
        make_plan(*common, "1,2,3,4", 29513, 0, execution_profile="four_gpu_validation")
    for selection in ([], ["unknown/ucit"], ["lora_ft/ucit", "lora_ft/ucit"]):
        with pytest.raises(ValueError, match="job selection"):
            make_plan(*common, "0,1,2,3", 29513, 0, execution_profile="four_gpu_validation", job_select=selection)


def test_failed_job_does_not_contaminate_or_stop_next_job(tmp_path: Path, monkeypatch) -> None:
    for name in ("MODEL", "DATA", "CKPT", "RESULTS", "LOGS"):
        monkeypatch.setenv(f"MCITLIB_{name}_ROOT", "/inherited-must-not-be-used")
    profile = tmp_path / "paths.yaml"
    profile.write_text(yaml.safe_dump({"model_root": "./model", "data_root": "./data", "ckpt_root": "./ckpt",
                                      "results_root": "./results", "logs_root": "./logs"}), encoding="utf-8")
    # make_plan honors explicit environment overrides; clear them before making the plan,
    # then put one back to verify it is removed from the child processes.
    for name in ("MODEL", "DATA", "CKPT", "RESULTS", "LOGS"):
        monkeypatch.delenv(f"MCITLIB_{name}_ROOT")
    plan = make_plan(tmp_path, profile, "cpu-check", sys.executable, sys.executable, "0,1", 29513, 0)
    jobs = plan["jobs"][:2]
    assert len({job.profile["results_root"] for job in jobs}) == 2
    assert len({job.profile["ckpt_root"] for job in jobs}) == 2
    assert not plan["run_dir"].exists()

    child = tmp_path / "cpu_child.py"
    child.write_text("""import json, os, sys
from pathlib import Path
phase, root, outcome = sys.argv[1:]
root = Path(root)
assert 'MCITLIB_DATA_ROOT' not in os.environ
if phase == 'doctor':
    print(json.dumps({'failed': False, 'checks': []}))
elif phase == 'worker':
    if outcome == 'fail':
        print('injected worker failure', file=sys.stderr)
        sys.exit(7)
    (root / 'worker-completed').write_text('ok')
elif phase == 'metrics':
    (root / 'metrics.json').write_text('{"ok": true}')
""", encoding="utf-8")
    monkeypatch.setenv("MCITLIB_DATA_ROOT", "/inherited-must-not-be-used")
    for job, outcome in zip(jobs, ("fail", "pass")):
        command = [sys.executable, str(child)]
        job.doctors = [("train", [*command, "doctor", str(job.directory), outcome])]
        job.worker = [*command, "worker", str(job.directory), outcome]
        job.metrics = [*command, "metrics", str(job.directory), outcome]
    plan["jobs"] = jobs

    queue = Queue(plan)
    assert queue.execute() == 1
    state = json.loads(queue.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed_with_failures"
    assert state["summary"] == {"failed": 1, "completed": 1}
    assert state["jobs"][jobs[0].key]["status"] == "failed"
    assert "injected worker failure" in Path(state["jobs"][jobs[0].key]["failure_log"]).read_text()
    assert state["jobs"][jobs[1].key]["status"] == "completed"
    assert (jobs[1].directory / "worker-completed").read_text() == "ok"
    assert json.loads((jobs[1].directory / "metrics.json").read_text()) == {"ok": True}
    assert not (jobs[0].directory / "metrics.json").exists()


def test_exact_new_method_selection_uses_physical_gpu_four_for_dgg(tmp_path: Path) -> None:
    profile = tmp_path / 'paths.yaml'
    profile.write_text('{}\n')
    names = ('pclr', 'same', 'sigma', 'dgg', 'octopus')
    selection = [f'{method}/{benchmark}' for method in names for benchmark in ('ucit', 'mllm_dcl')]
    plan = make_plan(tmp_path, profile, 'new-only', sys.executable, None, '4,5', 29681, 0,
                     job_select=selection)
    assert {job.key for job in plan['jobs']} == set(selection)
    for job in plan['jobs']:
        expected = '4' if job.method == 'dgg' else '4,5'
        assert job.worker[job.worker.index('--gpus') + 1] == expected
        doctor = job.doctors[0][1]
        assert doctor[doctor.index('--gpus') + 1] == expected
        assert ('--dgg-sample-profile' in doctor) == (job.method == 'dgg')
        assert '--skip-gpu' not in doctor and '--skip-entrypoints' not in doctor


def test_doctor_receives_its_own_physical_gpu_visibility(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / 'paths.yaml'
    profile.write_text('{}\n')
    plan = make_plan(tmp_path, profile, 'visibility', sys.executable, None, '4,5', 29681, 0,
                     job_select=['dgg/ucit'])
    queue = Queue(plan)
    queue.state = {'control': {}, 'jobs': {}}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5')
    stdout, stderr = tmp_path / 'doctor.out', tmp_path / 'doctor.err'
    argv = [sys.executable, '-c', 'import os; print(os.environ.get("CUDA_VISIBLE_DEVICES"))', '--gpus', '4']
    code, _ = queue.child(argv, stdout, stderr, {}, doctor=True)
    assert code == 0 and stdout.read_text().strip() == '4'
    stdout = tmp_path / 'worker.out'
    code, _ = queue.child(argv, stdout, stderr, {}, doctor=False)
    assert code == 0 and stdout.read_text().strip() == 'None'
