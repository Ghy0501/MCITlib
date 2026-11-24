# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only checks for the explicit four-GPU validation execution profile."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcitlib.config.loader import read_yaml
from mcitlib.config.schema import Command
from mcitlib.dataflow import find_gaps
from mcitlib.gpus import (
    apply_execution_profile,
    apply_gpu_assignment,
    execution_profile,
    main,
    require_four_discovered_chunks,
)
from mcitlib.runner import build_sequence_commands


def option(command: Command, name: str) -> str:
    return command.argv[command.argv.index(name) + 1]


def effective_batch(command: Command) -> int:
    world = len(option(command, "--include").split(":", 1)[1].split(","))
    return world * int(option(command, "--per_device_train_batch_size")) * int(option(command, "--gradient_accumulation_steps"))


def test_all_239_four_gpu_compatible_train_and_router_batches_are_feasible(repo_root: Path) -> None:
    profile = execution_profile(repo_root / "configs", "four_gpu_validation", None)
    counts = {"accumulation": 0, "batch": 0}
    for path in sorted((repo_root / "configs/methods").glob("*.yaml")):
        if path.stem == "dgg":
            continue  # DGG's release requires its single-GPU path.
        values = read_yaml(path)
        groups = list(values["overrides"]["llava"].values())
        groups.extend(item["tasks"] for item in values.get("router", {}).get("benchmarks", {}).values())
        for tasks in groups:
            for key, task in tasks.items():
                if not key.startswith("task_"):
                    continue
                old = Command(argv=list(task["train_command"]["argv"]), kind="train", task_id=task["task_id"])
                original = list(old.argv)
                old_effective = effective_batch(old)
                old_batch = option(old, "--per_device_train_batch_size")
                old_accum = option(old, "--gradient_accumulation_steps")
                new = apply_execution_profile([old], profile)[0]
                assert option(new, "--include") == "localhost:0,1,2,3"
                assert effective_batch(new) == old_effective
                assert new.metadata["effective_batch"] == old_effective
                if option(new, "--per_device_train_batch_size") == old_batch:
                    assert int(option(new, "--gradient_accumulation_steps")) * 2 == int(old_accum)
                    counts["accumulation"] += 1
                else:
                    assert int(option(new, "--per_device_train_batch_size")) * 2 == int(old_batch)
                    assert option(new, "--gradient_accumulation_steps") == old_accum
                    counts["batch"] += 1
                changed = {"--include", "--per_device_train_batch_size", "--gradient_accumulation_steps"}
                for index, token in enumerate(original):
                    if index and original[index - 1] in changed:
                        continue
                    assert new.argv[index] == token
    assert counts == {"accumulation": 227, "batch": 12}


def test_four_gpu_training_inference_merge_and_two_gpu_default(repo_root: Path) -> None:
    root = repo_root / "configs"
    default = build_sequence_commands(root, "llava", "lora_ft", "ucit", task_id=1)
    explicit_none = build_sequence_commands(root, "llava", "lora_ft", "ucit", task_id=1, execution_profile=None)
    assert [c.as_dict() for c in default] == [c.as_dict() for c in explicit_none]
    train_before = next(c for c in default if c.kind == "train")
    plan = build_sequence_commands(root, "llava", "lora_ft", "ucit", task_id=1,
                                   gpus="0,1,2,3", execution_profile="four_gpu_validation")
    train = next(c for c in plan if c.kind == "train")
    assert option(train, "--include") == "localhost:0,1,2,3"
    assert train.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert effective_batch(train) == effective_batch(train_before)
    assert option(train, "--learning_rate") == option(train_before, "--learning_rate")
    assert option(train, "--num_train_epochs") == option(train_before, "--num_train_epochs")
    chunks = [c for c in plan if "--answers-file" in c.argv]
    assert [option(c, "--num-chunks") for c in chunks] == ["4"] * 4
    assert [option(c, "--chunk-idx") for c in chunks] == ["0", "1", "2", "3"]
    assert [c.env["CUDA_VISIBLE_DEVICES"] for c in chunks] == ["0", "1", "2", "3"]
    assert len({c.metadata["parallel_group"] for c in chunks}) == 1
    merge = next(c for c in plan if c.metadata.get("merge_chunks"))
    assert merge.argv[merge.argv.index("--inputs") + 1:] == [option(c, "--answers-file") for c in chunks]
    assert all(c.metadata["execution_profile"] == "four_gpu_validation" for c in plan)
    after = build_sequence_commands(root, "llava", "lora_ft", "ucit", task_id=1)
    assert [c.as_dict() for c in default] == [c.as_dict() for c in after]


def test_cli_train_and_doctor_accept_same_explicit_profile(capsys: pytest.CaptureFixture) -> None:
    import json

    from mcitlib.cli import build_parser
    from mcitlib.cli import main as cli_main

    opts = ["--method", "lora_ft", "--benchmark", "ucit", "--gpus", "0,1,2,3",
            "--master-port", "29873", "--execution-profile", "four_gpu_validation"]
    for command in ("train", "doctor"):
        parsed = build_parser().parse_args([command, *opts])
        assert parsed.execution_profile == "four_gpu_validation"
        assert parsed.gpus == "0,1,2,3" and parsed.master_port == 29873
    assert cli_main(["train", *opts, "--task", "1", "--dry-run"]) == 0
    commands = json.loads(capsys.readouterr().out)
    train = next(command for command in commands if command["kind"] == "train")
    assert train["argv"][train["argv"].index("--include") + 1] == "localhost:0,1,2,3"
    assert train["argv"][train["argv"].index("--master_port") + 1] == "29873"


def test_keep_gradient_preparation_remains_two_shards(repo_root: Path) -> None:
    from copy import deepcopy

    profile = execution_profile(repo_root / "configs", "four_gpu_validation", None)
    records = []
    for index in range(2):
        records.append(Command(argv=["python", "-m", "mcitlib.methods._keeplora.extract_gradients",
                                     "--num-chunks", "2", "--chunk-idx", str(index)], kind="extra", task_id=1,
                               env={"CUDA_VISIBLE_DEVICES": str(index)}, metadata={"order": index}))
    records.append(Command(argv=["python", "-m", "mcitlib.methods._keeplora.extract_gradients",
                                 "--num-chunks", "2", "--merge-only"], kind="extra", task_id=1,
                           env={"CUDA_VISIBLE_DEVICES": "0,1"}, metadata={"order": 2}))
    original = deepcopy(records)
    commands = apply_execution_profile(records, profile)
    apply_gpu_assignment(commands, "0,1,2,3", None)
    assert [c.argv for c in commands] == [c.argv for c in original]
    assert [c.env for c in commands] == [c.env for c in original]
    assert all(c.metadata["gradient_preparation_shards"] == 2 for c in commands)


def test_inexact_batch_and_bad_profile_are_rejected(repo_root: Path) -> None:
    profile = execution_profile(repo_root / "configs", "four_gpu_validation", None)
    command = Command(argv=["deepspeed", "--include", "localhost:0,1", "--per_device_train_batch_size", "1",
                            "--gradient_accumulation_steps", "1"], kind="train")
    original = list(command.argv)
    with pytest.raises(ValueError, match="cannot preserve effective batch"):
        apply_execution_profile([command], profile)
    assert command.argv == original
    with pytest.raises(ValueError, match="requires --gpus"):
        execution_profile(repo_root / "configs", "four_gpu_validation", "0,1")
    with pytest.raises(ValueError, match="unknown execution profile"):
        execution_profile(repo_root / "configs", "not_a_profile", None)
    for benchmark in ("mllm_acl", "general"):
        with pytest.raises(ValueError, match="supports only ucit and mllm_dcl"):
            build_sequence_commands(repo_root / "configs", "llava", "lora_ft", benchmark,
                                    only_eval=benchmark == "general", execution_profile="four_gpu_validation")


def test_discover_still_requires_dependencies_and_exactly_four_chunks(tmp_path: Path) -> None:
    directory = tmp_path / "chunks"
    directory.mkdir()
    command = Command(argv=["python", "-m", "mcitlib.eval.chunks", "merge", "--output", str(directory / "merge.jsonl"),
                            "--discover", str(directory)], kind="eval", task_id=1, metadata={"merge_chunks": True})
    require_four_discovered_chunks([command])
    assert "--discover" in command.argv
    assert any(gap.flag == "--discover" for gap in find_gaps([command], ()))
    for index in range(2):
        (directory / f"2_{index}.jsonl").write_text(str(index))
    assert main(["merge-expected-chunks", "--output", str(directory / "out"), "--discover", str(directory)]) == 2
    for path in directory.glob("2_*.jsonl"):
        path.unlink()
    for index in range(4):
        (directory / f"4_{index}.jsonl").write_text(str(index))
    assert main(["merge-expected-chunks", "--output", str(directory / "out"), "--discover", str(directory)]) == 0
    assert (directory / "out").read_text() == "0123"


def test_doctor_checks_same_profile_plan_and_needs_four(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcitlib.doctor as doctor
    from mcitlib.paths import PathResolver

    seen: dict[str, object] = {}
    monkeypatch.setattr(doctor, "check_environment", lambda *args: None)
    monkeypatch.setattr(doctor, "check_vendored_imports", lambda *args: None)
    monkeypatch.setattr(doctor, "check_inputs", lambda *args: None)
    monkeypatch.setattr(doctor, "check_prerequisites", lambda *args: None)
    monkeypatch.setattr(doctor, "_keeplora_config_inputs", lambda report, commands: (commands, ()))
    monkeypatch.setattr(doctor, "check_gpus", lambda report, needed, method, ids: seen.update(needed=needed, ids=ids))
    monkeypatch.setattr(doctor, "check_dataflow", lambda report, commands, *args: seen.update(commands=commands))
    doctor.run_doctor("lora_ft", "ucit", None, repo_root, skip_entrypoints=True,
                      execution_profile="four_gpu_validation", gpus="0,1,2,3", master_port=29873)
    paths = PathResolver.from_file(None, repo_root=repo_root)
    expected = build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "ucit", paths=paths,
                                       execution_profile="four_gpu_validation", gpus="0,1,2,3", master_port=29873)
    assert seen["needed"] == 4 and seen["ids"] == ["0", "1", "2", "3"]
    assert [c.as_dict() for c in seen["commands"]] == [c.as_dict() for c in expected]


@pytest.mark.parametrize("method", ["modalprompt", "mrlora"])
def test_doctor_segments_match_worker_order(repo_root: Path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    import mcitlib.doctor as doctor
    import mcitlib.runner as runner
    from mcitlib.paths import PathResolver
    from mcitlib.validation_batch import worker_commands

    def fake_build(*args, **kwargs):
        sequence = kwargs.get("sequence", "train")
        if kwargs.get("only_eval"):
            task = kwargs["task_id"]
            return [Command(["eval", str(task)], task_id=task, kind="eval")]
        return [Command([sequence, str(task)], task_id=task, kind="train", sequence=sequence)
                for task in range(1, 7)]

    monkeypatch.setattr(runner, "build_sequence_commands", fake_build)
    monkeypatch.setattr(doctor, "check_environment", lambda *args: None)
    monkeypatch.setattr(doctor, "check_vendored_imports", lambda *args: None)
    monkeypatch.setattr(doctor, "check_inputs", lambda *args: None)
    monkeypatch.setattr(doctor, "check_prerequisites", lambda *args: None)
    monkeypatch.setattr(doctor, "_keeplora_config_inputs", lambda report, commands: (commands, ()))
    captured: list[Command] = []
    monkeypatch.setattr(doctor, "check_dataflow", lambda report, commands, *args: captured.extend(commands))
    common = {"skip_gpu": True, "skip_entrypoints": True, "validation_fixes": True,
              "execution_profile": "four_gpu_validation", "gpus": "0,1,2,3", "master_port": 29873}
    doctor.run_doctor(method, "ucit", None, repo_root, sequence="train", **common)
    if method == "mrlora":
        doctor.run_doctor(method, "ucit", None, repo_root, sequence="router", **common)
    resolver = PathResolver.from_file(None, repo_root=repo_root)
    expected = worker_commands(repo_root, resolver, method, "ucit", "0,1,2,3", 29873, "four_gpu_validation")
    assert [command.as_dict() for command in captured] == [command.as_dict() for command in expected]


def test_doctor_profile_cli_uses_same_resource_environment(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    import mcitlib.doctor as doctor
    from mcitlib.cli import build_parser

    seen: dict[str, object] = {}
    monkeypatch.setenv("MCITLIB_GPUS", "0,1,2,3")
    monkeypatch.setenv("MCITLIB_MASTER_PORT", "29873")
    monkeypatch.setattr(doctor, "run_doctor", lambda *args, **kwargs: seen.update(kwargs) or doctor.Report())
    args = build_parser().parse_args(["doctor", "--method", "lora_ft", "--benchmark", "ucit",
                                      "--execution-profile", "four_gpu_validation"])
    assert doctor.main_doctor(args) == 0
    assert seen["gpus"] == "0,1,2,3" and seen["master_port"] == 29873
    capsys.readouterr()


def test_doctor_profile_preserves_mrlora_discovery_failure(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcitlib.doctor as doctor

    monkeypatch.setattr(doctor, "check_environment", lambda *args: None)
    monkeypatch.setattr(doctor, "check_vendored_imports", lambda *args: None)
    monkeypatch.setattr(doctor, "check_inputs", lambda *args: None)
    monkeypatch.setattr(doctor, "check_prerequisites", lambda *args: None)
    monkeypatch.setattr(doctor, "_keeplora_config_inputs", lambda report, commands: (commands, ()))
    report = doctor.run_doctor("mrlora", "mllm_dcl", None, repo_root, skip_gpu=True, skip_entrypoints=True,
                               execution_profile="four_gpu_validation", gpus="0,1,2,3")
    assert any(check.name == "dataflow" and check.status == doctor.FAIL and "--discover" in check.detail
               for check in report.checks)


def test_doctor_rejects_unavailable_fourth_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from mcitlib.doctor import FAIL, Report, check_gpus

    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 3)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    report = Report()
    check_gpus(report, 4, "lora_ft", ["0", "1", "2", "3"])
    assert report.failed
    assert any(check.name == "selected GPUs" and check.status == FAIL for check in report.checks)
