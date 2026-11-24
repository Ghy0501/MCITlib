# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
import json
import os
import struct
import sys
from pathlib import Path

import pytest
import yaml

from mcitlib.config.schema import Command, PathSet
from mcitlib.internvl_retention import FreshMergedRetention, _identity, build_plan
from mcitlib.internvl_validation import (
    acquire_gpu_leases,
    audit_commands,
    sample_inputs,
    source_inventory,
    verify_gpu_leases,
)
from mcitlib.paths import PathResolver
from mcitlib.reporting import build_report, generate_report
from mcitlib.validation_batch import Queue, make_plan


def profile(root):
    path = root / "paths.yaml"
    path.write_text(yaml.safe_dump({"model_root": str(root / "model"), "data_root": str(root / "data"),
                                    "ckpt_root": str(root / "ckpt"), "results_root": str(root / "result"),
                                    "logs_root": str(root / "logs")}))
    return path


def test_internvl_plan_is_isolated_and_only_uses_four_reviewed_gpus(tmp_path):
    common = (tmp_path, profile(tmp_path), "fresh-run", "/bin/python", "/bin/python")
    plan = make_plan(*common, "0,1,2,3", 29511, 32, execution_profile="four_gpu_validation",
                     job_select=["lora_ft/ucit"], backbone_id="internvl")
    assert plan["roots"]["ckpt_root"] == str(tmp_path / "ckpt/internvl/fresh-run")
    assert plan["lock_file"] == tmp_path / "logs/internvl/.validation_queue.lock"
    assert "--backbone" in plan["jobs"][0].doctors[0][1]
    assert plan["jobs"][0].worker[plan["jobs"][0].worker.index("--backbone") + 1] == "internvl"
    for gpus in ("4,5", "0,1,2,4", "0,1,2,5", "3,2,1,0"):
        with pytest.raises(ValueError, match="InternVL"):
            make_plan(*common, gpus, 29511, 32, execution_profile="four_gpu_validation",
                      job_select=["lora_ft/ucit"], backbone_id="internvl")
    with pytest.raises(ValueError, match="32 GiB"):
        make_plan(*common, "0,1,2,3", 29511, 31, execution_profile="four_gpu_validation",
                  job_select=["lora_ft/ucit"], backbone_id="internvl")


def test_gpu_lease_proof_requires_four_actual_exclusive_fds(tmp_path):
    directory = tmp_path / ".workspaces/sample-completion-20261108-control"
    directory.mkdir(parents=True)
    fds = []
    try:
        for index in range(4):
            path = directory / f"gpu-{index}.lease"
            path.touch()
            fd = os.open(path, os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fds.append(fd)
        value = ",".join(map(str, fds))
        assert [row["gpu"] for row in verify_gpu_leases(tmp_path, value, directory)] == [0, 1, 2, 3]
        with pytest.raises(ValueError):
            verify_gpu_leases(tmp_path, ",".join(map(str, reversed(fds))), directory)
        fcntl.flock(fds[2], fcntl.LOCK_UN)
        with pytest.raises(ValueError, match="not exclusively held"):
            verify_gpu_leases(tmp_path, value, directory)
    finally:
        for fd in fds:
            os.close(fd)


def test_public_queue_acquires_and_releases_own_four_leases(tmp_path, monkeypatch):
    import mcitlib.internvl_validation as validation
    lease_root = tmp_path / "logs/.gpu-leases"
    monkeypatch.delenv("MCITLIB_INTERNVL_LEASE_ROOT", raising=False)
    monkeypatch.delenv("MCITLIB_INTERNVL_LEASE_FDS", raising=False)
    checks = []
    monkeypatch.setattr(validation, "gpu_idle", lambda: checks.append("idle"))
    plan = {"project": tmp_path, "backbone": "internvl", "gpu_lease_root": lease_root,
            "run_dir": tmp_path / "run"}
    def locked(_self):
        assert len(verify_gpu_leases(tmp_path)) == 4
        with pytest.raises(BlockingIOError):
            acquire_gpu_leases(tmp_path, lease_root)
        return 0
    monkeypatch.setattr(Queue, "_execute_locked", locked)
    assert Queue(plan).execute() == 0
    assert checks == ["idle", "idle"]
    _, after = acquire_gpu_leases(tmp_path, lease_root)
    for fd in after:
        os.close(fd)


def test_public_queue_reaps_its_detached_descendant_before_releasing_leases(tmp_path, monkeypatch):
    root, fds = acquire_gpu_leases(tmp_path, tmp_path / "logs/.gpu-leases")
    monkeypatch.setenv("MCITLIB_INTERNVL_LEASE_ROOT", str(root))
    monkeypatch.setenv("MCITLIB_INTERNVL_LEASE_FDS", ",".join(map(str, fds)))
    run = tmp_path / "run"
    run.mkdir()
    record = tmp_path / "orphan.pid"
    child_script = f"import os,time;os.setsid();open({str(record)!r},'w').write(str(os.getpid()));time.sleep(60)"
    script = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child_script!r}]);time.sleep(0.8)"
    plan = {"project": tmp_path, "backbone": "internvl", "gpu_lease_root": root, "run_dir": run,
            "gpus": "0,1,2,3"}
    q = Queue(plan)
    q.state = {"jobs": {"test/ucit": {"status": "running"}}, "control": {"child": None}}
    try:
        code, error = q.child([sys.executable, "-B", "-c", script], run / "out", run / "err", q.state["jobs"]["test/ucit"])
        assert code == 0 and error is None
        assert record.is_file()
        pid = int(record.read_text())
        assert not Path(f"/proc/{pid}").exists()
        assert len(verify_gpu_leases(tmp_path)) == 4
    finally:
        for fd in fds:
            os.close(fd)


def train_command(task, *, sequence="train", epochs="1", rate="0.0002"):
    argv = ["deepspeed", "--include", "localhost:0,1,2,3", "train.py",
            "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "16",
            "--learning_rate", rate, "--num_train_epochs", epochs]
    profile = {"id": "internvl_sample_unified", "backbone": "internvl",
               "path": "configs/training_profiles/internvl_sample_unified.yaml", "sha256": "a" * 64,
               "world_size": 4, "microbatch": 1, "gradient_accumulation": 16,
               "global_batch_size": 64, "learning_rate": rate, "epochs": epochs,
               "role": "main_train" if sequence == "train" else "auxiliary_preserved",
               "source_values": {"--learning_rate": rate, "--num_train_epochs": epochs}}
    key = "training_profile" if sequence == "train" else "auxiliary_training_profile"
    return Command(argv, kind="train", task_id=task, sequence=sequence, metadata={key: profile})


def test_budget_audit_rejects_drift_and_preserves_router():
    inputs = {"configs/training_profiles/internvl_sample_unified.yaml": "a" * 64}
    commands = [train_command(i) for i in range(1, 7)] + [train_command(1, sequence="router", epochs="30")]
    assert len(audit_commands(commands, method="mrlora", task_count=6, source_files=inputs)) == 7
    commands[0].argv[commands[0].argv.index("--learning_rate") + 1] = "0.0001"
    with pytest.raises(ValueError, match="differ"):
        audit_commands(commands, method="mrlora", task_count=6, source_files=inputs)
    commands[0] = train_command(1)
    commands[-1].argv[commands[-1].argv.index("--include") + 1] = "localhost:0,1,2,4"
    with pytest.raises(ValueError, match="GPU launcher"):
        audit_commands(commands, method="mrlora", task_count=6, source_files=inputs)


@pytest.mark.parametrize("task_count", [5, 6])
def test_budget_audit_requires_one_modalprompt_epoch(task_count):
    inputs = {"configs/training_profiles/internvl_sample_unified.yaml": "a" * 64}
    commands = [train_command(i) for i in range(1, task_count + 1)]
    assert len(audit_commands(commands, method="modalprompt", task_count=task_count,
                              source_files=inputs)) == task_count
    commands[-1] = train_command(task_count, epochs="10")
    with pytest.raises(ValueError, match="budget"):
        audit_commands(commands, method="modalprompt", task_count=task_count, source_files=inputs)


def test_source_inventory_detects_source_or_config_change(tmp_path):
    for name in ("mcitlib", "configs", "third_party", "envs", "assets", "resources"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "file.py").write_text(name)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    before = source_inventory(tmp_path)
    (tmp_path / "mcitlib/file.py").write_text("changed")
    assert before != source_inventory(tmp_path)


def test_internvl_report_finds_real_scored_cells_in_isolated_backbone_tree(tmp_path):
    run = tmp_path / "run"
    result = tmp_path / "results/lora_ft/ucit/internvl/lora_ft/UCIT"
    score = result / "each_dataset/ImageNet-R/LoRA-task1/Result.text"
    score.parent.mkdir(parents=True)
    score.write_text("Samples: 1000\nAccuracy: 80.6%\n")
    run.mkdir()
    state = {"run_id": "r1", "status": "running", "current": "lora_ft/ucit", "backbone": "internvl",
             "source_binding": "/proof/binding.json", "source_binding_sha256": "a" * 64,
             "project": str(tmp_path), "roots": {"results_root": str(tmp_path / "results"),
             "logs_root": str(run), "ckpt_root": str(tmp_path / "ckpts")},
             "jobs": {"lora_ft/ucit": {"method": "lora_ft", "benchmark": "ucit", "status": "running", "task": 1}}}
    (run / "state.json").write_text(json.dumps(state))
    report = build_report(run)
    assert report["backbone"] == "internvl"
    assert report["jobs"][0]["matrix"][0] == [80.6]
    assert report["jobs"][0]["results_dir"] == str(result)
    output = generate_report(run, tmp_path / "report")
    markdown = Path(output["markdown"]).read_text()
    assert "Backbone: InternVL" in markdown and "1 / 21" in markdown


def test_real_3k_1k_train_and_eval_mapping():
    main = Path(__file__).resolve().parents[2]
    profile_path = main / "configs/paths.validation.yaml"
    if not profile_path.is_file():
        pytest.skip("host validation sample is not installed")
    resolver = PathResolver.from_file(profile_path, repo_root=main)
    assert len(sample_inputs(main, resolver)) >= 24
    assert len(sample_inputs(main, resolver, main / "data/validation_aux/paths.yaml")) == 47


def test_sefe_removes_only_obsolete_merged_model_after_durable_consumers(tmp_path):
    project = tmp_path / "project"
    root = project / "ckpts/run1/sefe/mllm_dcl"
    job = project / "logs/run1/jobs/sefe/mllm_dcl"
    job.mkdir(parents=True)
    models = root / "MCITlib/MLLM-DCL/InternVL/RegLoRA"
    commands = []
    for task in range(1, 6):
        adapter = models / f"Task{task}_internvl_lora"
        merged = Path(str(adapter) + "_merged")
        base = models / f"Task{task - 1}_internvl_lora_merged" if task > 1 else project / "base"
        commands.append(Command(["deepspeed", "mcitlib.backbones.llava.train_entry", "--base", str(base)],
                                kind="train", task_id=task))
        commands.append(Command(["python", "mcitlib/methods/_sefe/merge.py", "--model-path", str(adapter),
                                 "--model-base", str(base), "--save-model-path", str(merged)],
                                kind="extra", task_id=task))
        commands.append(Command(["python", "-m", "mcitlib.backbones.llava.eval_runners.entry",
                                 "--model-path", str(merged)], kind="eval", task_id=task))
    manager = FreshMergedRetention(commands, project=project, ckpt_root=root, job_dir=job, benchmark="mllm_dcl")
    first, second = manager.plan["entries"][:2]
    target, successor = Path(first["path"]), Path(second["path"])
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"old")
    successor.mkdir(parents=True)
    (successor / "config.json").write_text("{}")
    (successor / "tokenizer_config.json").write_text("{}")
    (successor / "tokenizer.model").write_bytes(b"tokenizer")
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (successor / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    adapter = models / "Task1_internvl_lora"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"keep")
    assert manager.before_step(first["last_consumer"] + 1) == []
    assert target.exists()
    markers = job / ".mcitlib/steps"
    markers.mkdir(parents=True, exist_ok=True)
    events = []
    for index in range(first["last_consumer"] + 1):
        command = manager.plan["commands"][index]
        ident = _identity(command)
        event = {"index": index, "id": ident, "command": command, "status": "completed", "return_code": 0}
        events.append(event)
        (markers / (ident + "-attempt.json")).write_text(json.dumps({**event, "validation_error": None}))
    (job / "events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
    deleted = manager.before_step(first["last_consumer"] + 1)
    assert len(deleted) == 1 and deleted[0]["status"] == "deleted"
    assert not target.exists() and successor.is_dir() and (adapter / "adapter.bin").exists()


def test_keeplora_retention_accounts_for_model_references_inside_config(tmp_path):
    project = tmp_path / "project"
    root = project / "ckpts/run1/keeplora/mllm_dcl"
    project.mkdir()
    models = root / "MCITlib/MLLM-DCL/InternVL/KeepLoRA"
    commands = []
    for task in range(1, 6):
        adapter = models / f"Task{task}_internvl_lora"
        merged = Path(str(adapter) + "_merged")
        base = models / f"Task{task - 1}_internvl_lora_merged" if task > 1 else project / "base"
        commands.append(Command(["deepspeed", "mcitlib.backbones.llava.train_entry", "--base", str(base)],
                                kind="train", task_id=task))
        commands.append(Command(["python", "mcitlib/methods/_keeplora/merge.py", "--model-path", str(adapter),
                                 "--model-base", str(base), "--save-model-path", str(merged)],
                                kind="extra", task_id=task))
        commands.append(Command(["python", "-m", "mcitlib.backbones.llava.eval_runners.entry",
                                 "--model-path", str(merged)], kind="eval", task_id=task))
    config = project / "late-reference.json"
    config.write_text(json.dumps({"model_path": "/your_ckpts_path/MCITlib/MLLM-DCL/InternVL/KeepLoRA/Task1_internvl_lora_merged"}))
    paths = PathSet(project / "models", project / "data", root, project, project / "results", project / "logs")
    context = PathResolver(paths).context("mllm_dcl")
    commands.append(Command(["python", "-m", "mcitlib.methods._keeplora.extract_gradients",
                             "--train-config", str(config)], kind="extra", task_id=5,
                            env={"MCITLIB_PATH_CONTEXT": context}))
    plan = build_plan(commands, root, "mllm_dcl", "keeplora")
    assert plan["entries"][0]["last_consumer"] == len(commands) - 1
