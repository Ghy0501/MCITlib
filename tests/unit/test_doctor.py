# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""`mcitlib doctor` reports what would stop a real run, with the fix, before GPU time is spent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcitlib.doctor import FAIL, OK, WARN, Report, gpus_needed, run_doctor
from mcitlib.runner import build_sequence_commands


def _server(tmp_path: Path) -> Path:
    for name in ("models", "data", "ckpts"):
        (tmp_path / name).mkdir(exist_ok=True)
    config = tmp_path / "paths.yaml"
    config.write_text(f"model_root: {tmp_path}/models\ndata_root: {tmp_path}/data\nckpt_root: {tmp_path}/ckpts\n")
    return config


def _status(report: Report, prefix: str) -> list[str]:
    return [check.status for check in report.checks if check.name.startswith(prefix)]


def _materialise_inputs(repo_root: Path, config: Path, method: str, benchmark: str, tmp_path: Path) -> None:
    from mcitlib.paths import PathResolver

    paths = PathResolver.from_file(config, repo_root=repo_root)
    roots = (str(paths.paths.model_root), str(paths.paths.data_root))
    for command in build_sequence_commands(repo_root / "configs", "llava", method, benchmark, paths=paths):
        for value in command.argv:
            if any(value.startswith(root) for root in roots):
                target = Path(value)
                if target.suffix in {".json", ".jsonl", ".bin"}:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                else:
                    target.mkdir(parents=True, exist_ok=True)


def test_missing_inputs_are_listed_with_the_fix(repo_root: Path, tmp_path: Path) -> None:
    report = run_doctor("lora_ft", "ucit", _server(tmp_path), repo_root, skip_gpu=True, skip_entrypoints=True)
    missing = [check for check in report.checks if check.name.startswith("missing under")]
    assert {Path(check.name.split("under ")[1]).name for check in missing} == {"data", "models"}
    assert all(check.status == FAIL and check.fix for check in missing)
    assert report.failed
    assert "OK" in report.render() and "failures" in report.render()


def test_inputs_pass_once_the_files_exist(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    from collections import namedtuple

    disk_usage = namedtuple("disk_usage", "total used free")
    monkeypatch.setattr("mcitlib.doctor.shutil.disk_usage", lambda _: disk_usage(500 * 2**30, 0, 500 * 2**30))
    config = _server(tmp_path)
    _materialise_inputs(repo_root, config, "lora_ft", "ucit", tmp_path)
    report = run_doctor("lora_ft", "ucit", config, repo_root, skip_gpu=True, skip_entrypoints=True)
    assert _status(report, "input files") == [OK]
    assert _status(report, "missing under") == []
    assert _status(report, "dataflow") == [OK]
    assert _status(report, "checkpoint directory") == [OK]


def test_acl_requires_the_math_judge_and_the_mathvista_table(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM", raising=False)
    report = run_doctor("lora_ft", "mllm_acl", _server(tmp_path), repo_root, skip_gpu=True, skip_entrypoints=True)
    judge = [check for check in report.checks if check.name.startswith("Math judge")]
    assert judge and judge[0].status == FAIL and "OPENAI_API_KEY" in judge[0].fix
    table = [check for check in report.checks if check.name == "llava_v1.5_7b_MathVista_MINI.xlsx"]
    assert table and table[0].status in {OK, FAIL}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    report = run_doctor("lora_ft", "mllm_acl", _server(tmp_path), repo_root, skip_gpu=True, skip_entrypoints=True)
    assert [check.status for check in report.checks if check.name == "Math judge"] == [OK]


def test_ucit_does_not_ask_for_the_judge(repo_root: Path, tmp_path: Path) -> None:
    report = run_doctor("lora_ft", "ucit", _server(tmp_path), repo_root, skip_gpu=True, skip_entrypoints=True)
    assert not [check for check in report.checks if "judge" in check.name]


def test_known_v3_gap_is_reported_as_a_dataflow_failure(repo_root: Path, tmp_path: Path) -> None:
    report = run_doctor("mrlora", "mllm_dcl", _server(tmp_path), repo_root, skip_gpu=True, skip_entrypoints=True)
    dataflow = [check for check in report.checks if check.name == "dataflow"]
    assert dataflow and dataflow[0].status == FAIL and "--discover" in dataflow[0].detail


def test_unknown_method_is_a_clean_failure(repo_root: Path, tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        run_doctor("nonexistent", "ucit", _server(tmp_path), repo_root, skip_gpu=True, skip_entrypoints=True)


def test_gpu_count_comes_from_the_training_command(repo_root: Path) -> None:
    commands = build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "ucit")
    assert gpus_needed(commands) == 2


@pytest.mark.parametrize("capacity,status", [(40, WARN), (80, OK)])
def test_internvl_keeplora_memory_bound_uses_actual_configs(tmp_path: Path, monkeypatch, capacity, status) -> None:
    from types import SimpleNamespace

    import torch

    from mcitlib.doctor import check_gpus, internvl_keeplora_memory_lower_bound

    model = {"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32,
             "num_key_value_heads": 32, "num_hidden_layers": 32}
    vision = {"hidden_size": 3200, "intermediate_size": 12800, "num_hidden_layers": 48}
    for directory, config in (("InternVL-Chat-ViT-6B-Vicuna-7B", model), ("InternViT-6B-224px", vision)):
        target = tmp_path / directory
        target.mkdir()
        (target / "config.json").write_text(json.dumps(config))
    selected, frozen_vision = internvl_keeplora_memory_lower_bound(model, vision)
    assert selected == 32 * (4 * 4096**2 + 3 * 4096 * 11008) * 2
    assert frozen_vision == 48 * (4 * 3200**2 + 2 * 3200 * 12800) * 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=capacity * 2**30))
    report = Report()
    check_gpus(report, 4, "keeplora", ["0", "1", "2", "3"], "internvl", tmp_path)
    check = next(check for check in report.checks if check.name == "KeepLoRA memory")
    assert check.status == status
    assert "q/k/v/o/gate/up/down" in check.detail and "extract_gradients.py" in check.detail
    assert "47." in check.detail and "excluding other weights and activations" in check.detail


@pytest.mark.parametrize('installed,expected', [('0.6.13', OK), ('0.9.12', FAIL), (None, FAIL)])
def test_internvl_timm_pin_is_checked(tmp_path: Path, monkeypatch, installed, expected) -> None:
    import importlib.metadata

    from mcitlib.doctor import check_internvl_model

    envs = tmp_path / 'envs'
    envs.mkdir()
    (envs / 'internvl-extra.txt').write_text('timm==0.6.13\n')

    def version(name):
        assert name == 'timm'
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr('mcitlib.doctor.metadata.version', version)
    monkeypatch.setattr('mcitlib.doctor.importlib.util.find_spec', lambda name: object())
    report = Report()
    check_internvl_model(report, tmp_path, tmp_path)
    dependency = next(check for check in report.checks if check.name == 'InternViT dependency')
    assert dependency.status == expected
    if expected == FAIL:
        assert 'envs/internvl-extra.txt' in dependency.fix


def test_report_is_machine_readable() -> None:
    report = Report()
    report.add(OK, "a")
    report.add(WARN, "b", "detail", "fix")
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["failed"] is False and [item["status"] for item in payload["checks"]] == [OK, WARN]


def test_write_probe_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    from mcitlib.doctor import Report, check_prerequisites

    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    existing = ckpt / ".mcitlib-write-test"
    existing.write_text("keep")
    report = Report()
    check_prerequisites(report, "lora_ft", "ucit", [], tmp_path, ckpt)
    assert existing.read_text() == "keep"
    assert sorted(path.name for path in ckpt.iterdir()) == [".mcitlib-write-test"]
