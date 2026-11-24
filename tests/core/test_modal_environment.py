# SPDX-License-Identifier: Apache-2.0
"""ModalPrompt's isolated environment is explicit and failure-isolated without a GPU."""
from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from mcitlib import doctor
from mcitlib.registry import METHODS, get_method
from mcitlib.validation_batch import Queue, main, make_plan
from tools import bounded_validation_smoke as bounded
from tools.transition_validation_queue import source_path

PROFILE = "llava-modalprompt-ds126"


def paths_file(root: Path) -> Path:
    path = root / "paths.yaml"
    path.write_text(yaml.safe_dump({"model_root": "./models", "data_root": "./data", "ckpt_root": "./ckpts",
                                    "results_root": "./results", "logs_root": "./logs"}), encoding="utf-8")
    return path


def test_profile_changes_only_deepspeed_and_modal_method(repo_root: Path) -> None:
    standard = doctor._pinned(repo_root / "envs/llava.txt")
    modal = doctor._pinned(repo_root / f"envs/{PROFILE}.txt")
    assert standard["deepspeed"] == "0.9.5" and modal["deepspeed"] == "0.12.6"
    assert {k: v for k, v in standard.items() if k != "deepspeed"} == {
        k: v for k, v in modal.items() if k != "deepspeed"
    }
    assert get_method("modalprompt").env == PROFILE
    assert sum(method.env == "llava" for method in METHODS) == 14
    assert get_method("sefe").env == get_method("keeplora").env == "llava-t437"


def install_fake_deepspeed(monkeypatch: pytest.MonkeyPatch, *, present: bool, official: bool) -> None:
    package = types.ModuleType("deepspeed")
    package.__path__ = []
    utils = types.ModuleType("deepspeed.utils")
    utils.__path__ = []
    fragment = types.ModuleType("deepspeed.utils.tensor_fragment")

    def getter(_parameter):
        return None

    def setter(_parameter, _value):
        return None

    getter.__module__ = "deepspeed.utils.tensor_fragment"
    setter.__module__ = "deepspeed.utils.tensor_fragment" if official else "fake_override"
    utils.safe_get_full_fp32_param = fragment.safe_get_full_fp32_param = getter
    if present:
        utils.safe_set_full_fp32_param = fragment.safe_set_full_fp32_param = setter
    utils.tensor_fragment = fragment
    package.utils = utils
    for name, module in (("deepspeed", package), ("deepspeed.utils", utils),
                         ("deepspeed.utils.tensor_fragment", fragment)):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(("version", "present", "official", "expected"), [
    ("0.12.6", True, True, doctor.OK),
    ("0.9.5", False, False, doctor.FAIL),
    ("0.12.6", True, False, doctor.FAIL),
])
def test_doctor_requires_exact_pins_and_official_safe_state_api(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, version: str, present: bool, official: bool, expected: str,
) -> None:
    pins = doctor._pinned(repo_root / f"envs/{PROFILE}.txt")
    monkeypatch.setattr(doctor.metadata, "version", lambda name: version if name == "deepspeed" else pins[name])
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _name: object())
    install_fake_deepspeed(monkeypatch, present=present, official=official)
    report = doctor.Report()
    doctor.check_environment(report, PROFILE, repo_root)
    api = next(check for check in report.checks if check.name == "ModalPrompt DeepSpeed safe state API")
    packages = next(check for check in report.checks if check.name == f"packages for environment `{PROFILE}`")
    assert api.status == expected
    assert packages.status == (doctor.FAIL if version != "0.12.6" else doctor.OK)
    assert report.failed == (expected == doctor.FAIL)


def test_queue_cli_separates_interpreters_and_keeps_symlink(tmp_path: Path, monkeypatch, capsys) -> None:
    import mcitlib.validation_batch as batch

    monkeypatch.setattr(batch, "repo_root", lambda: tmp_path)
    modal = tmp_path / "venv/bin/python"
    modal.parent.mkdir(parents=True)
    modal.symlink_to(sys.executable)
    argv = ["preview", "--paths", str(paths_file(tmp_path)), "--run-id", "preview", "--gpus", "0,1",
            "--job", "modalprompt/ucit", "--job", "lora_ft/ucit", "--job", "sefe/ucit",
            "--python-llava", sys.executable, "--python-t437", "/t437/bin/python", "--python-modalprompt", str(modal)]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    jobs = {item["method"]: item for item in result["jobs"]}
    assert jobs["modalprompt"]["environment"] == PROFILE and jobs["modalprompt"]["python"] == str(modal)
    assert jobs["lora_ft"]["environment"] == "llava" and jobs["lora_ft"]["python"] == sys.executable
    assert jobs["sefe"]["environment"] == "llava-t437" and jobs["sefe"]["python"] == "/t437/bin/python"
    assert jobs["modalprompt"]["doctors"][0]["argv"][0] == str(modal)
    assert jobs["modalprompt"]["worker"][0] == jobs["modalprompt"]["metrics"][0] == str(modal)


def test_missing_modal_interpreter_blocks_only_that_queue_job(tmp_path: Path) -> None:
    fake = tmp_path / "fake-python"
    fake.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
args = sys.argv[1:]
if 'doctor' in args: print(json.dumps({{'failed': False, 'checks': []}}))
elif 'metrics' in args: Path(args[args.index('--output') + 1]).write_text('{{}}')
elif '_worker' in args: pass
else: sys.exit(2)
''', encoding="utf-8")
    fake.chmod(0o755)
    plan = make_plan(tmp_path, paths_file(tmp_path), "isolated", str(fake), None, "0,1", 29511, 0,
                     job_select=["modalprompt/ucit", "lora_ft/ucit"])
    jobs = {job.method: job for job in plan["jobs"]}
    assert jobs["modalprompt"].env_name == PROFILE and jobs["modalprompt"].python is None
    assert jobs["lora_ft"].python == str(fake)
    plan["jobs"] = [jobs["modalprompt"], jobs["lora_ft"]]
    queue = Queue(plan)
    assert queue.execute() == 1
    assert queue.state["jobs"]["modalprompt/ucit"]["status"] == "blocked"
    assert queue.state["jobs"]["lora_ft/ucit"]["status"] == "completed"


def test_bounded_environment_and_unavailable_python_isolation(tmp_path: Path, monkeypatch, capsys) -> None:
    old = {"python": "/old/bin/python", "environment": "llava"}
    assert bounded._job_environment(tmp_path, "disco", old, None) == ("/old/bin/python", "llava")
    with pytest.raises(ValueError, match="--python-modalprompt"):
        bounded._job_environment(tmp_path, "modalprompt", old, None)
    assert bounded._job_environment(tmp_path, "modalprompt", old, "venv/bin/python") == (
        str(tmp_path / "venv/bin/python"), PROFILE,
    )
    plan = {"project": str(tmp_path), "run_dir": str(tmp_path / "logs"), "jobs": {
        "modalprompt/ucit": {"python": str(tmp_path / "missing"), "environment": PROFILE,
                             "job_dir": str(tmp_path / "modal")},
        "disco/ucit": {"python": sys.executable, "environment": "llava", "job_dir": str(tmp_path / "disco")},
    }}
    monkeypatch.setattr(bounded, "_read_plan", lambda _path: plan)
    calls = []
    monkeypatch.setattr(bounded, "_run_isolated", lambda argv, *_: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["bounded", "doctor", "--plan", "ignored", "--job", "modalprompt/ucit",
                                     "--job", "disco/ucit"])
    assert bounded.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert "modalprompt/ucit" in result["failures"] and len(calls) == 1
    assert calls[0][calls[0].index("--job") + 1] == "disco/ucit"
    # A pre-migration plan cannot silently reuse the former DS 0.9.5 profile,
    # even via the internal worker; an unrelated job remains runnable.
    plan["jobs"]["modalprompt/ucit"].update(python=sys.executable, environment="llava")
    calls.clear()
    assert bounded.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert "isolated" in result["failures"]["modalprompt/ucit"] and len(calls) == 1
    with pytest.raises(ValueError, match="isolated"):
        bounded._worker(Path("ignored"), "modalprompt/ucit", 1, "doctor", False, 20)


def test_handoff_allows_only_reviewed_llava_environment_files() -> None:
    assert source_path(f"envs/{PROFILE}.txt") == Path(f"envs/{PROFILE}.txt")
    assert source_path("envs/llava.txt") == Path("envs/llava.txt")
    for forbidden in ("envs/llava-t437.txt", "envs/private.txt"):
        with pytest.raises(ValueError):
            source_path(forbidden)


@pytest.mark.parametrize("recording", ["new", "missing", "legacy"])
def test_handoff_forwards_only_recorded_modal_interpreter(tmp_path: Path, monkeypatch, recording: str) -> None:
    from tools import transition_validation_queue as transition

    for name in transition.ROOT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    profile = paths_file(project)
    regular, modal = "/regular/bin/python", "/modal/bin/python"
    old = make_plan(project, profile, "old", regular, None, "0,1", 29511, 0,
                    job_select=["modalprompt/ucit", "lora_ft/ucit"], python_modalprompt=modal)
    new = make_plan(project, profile, "new", regular, None, "0,1,2,3", 29512, 0,
                    execution_profile="four_gpu_validation", job_select=["modalprompt/ucit", "lora_ft/ucit"],
                    python_modalprompt=modal if recording == "new" else None)
    run_dir = old["run_dir"]
    run_dir.mkdir(parents=True)
    state_jobs = {
        "lora_ft/ucit": {"environment": "llava", "python": regular},
        "modalprompt/ucit": {"environment": "llava" if recording == "legacy" else PROFILE,
                             "python": modal if recording == "new" else regular if recording == "legacy" else None},
    }
    state = {"jobs": state_jobs, "roots": old["roots"], "source_paths": str(profile),
             "minimum_free_gib": 0, "auxiliary_paths": None}
    (run_dir / "state.json").write_text(json.dumps(state))
    control = project / ".workspaces/control"
    control.mkdir(parents=True)
    snapshot, monitor = control / "snapshot.json", control / "monitor.json"
    receipt, output, approved_path = control / "receipt.json", control / "output.json", control / "approved.json"
    snapshot.write_text(json.dumps({"development_project": str(control / "dev"),
                                    "baseline_snapshot": str(control / "baseline")}))
    monitor.write_text(json.dumps({"project": str(project), "run_dir": str(run_dir), "run_id": "old",
                                   "pid": 123, "start_ticks": 456, "status": "old_queue_terminal",
                                   "finish_job": "lora_ft/ucit"}))
    approved = {"runtime_project": str(project), "old_run_dir": str(run_dir), "old_run_id": "old", "old_pid": 123,
                "old_start_ticks": 456, "snapshot": str(snapshot), "new_run_id": "new", "receipt": str(receipt),
                "monitor_status": str(monitor), "plan_path": str(output), "gpus": "0,1,2,3", "master_port": 29512,
                "snapshot_sha256": transition.digest(snapshot), "finish_job": "lora_ft/ucit", "files": [],
                "source_paths": str(profile), "auxiliary_paths": None, "minimum_free_gib": 0,
                "new_roots": new["roots"]}

    @contextmanager
    def idle(*_args):
        yield state

    monkeypatch.setattr(transition, "approved_manifest", lambda *_: approved)
    monkeypatch.setattr(transition, "idle_queue", idle)
    monkeypatch.setattr(transition, "verify_ledger", lambda *_: None)
    monkeypatch.setattr(transition, "remaining_jobs", lambda *_: list(state_jobs))
    monkeypatch.setattr(transition, "proposed_files", lambda *_: [])
    result = transition.prepare(project, run_dir, "old", 123, 456, snapshot, monitor, "new", receipt, output,
                                approved_path, "reviewed", gpus="0,1,2,3", port=29512)
    argv = result["new_queue_argv"]
    recorded = json.loads(output.read_text())["new_plan"]["jobs"]
    modal_job = next(item for item in recorded if item["method"] == "modalprompt")
    if recording == "new":
        assert argv[argv.index("--python-modalprompt") + 1] == modal
        assert modal_job["python"] == modal
    else:
        assert "--python-modalprompt" not in argv
        assert modal_job["python"] is None
