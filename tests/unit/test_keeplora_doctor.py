# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""KeepLoRA JSON dependencies are checked at their real positions without execution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mcitlib.config.schema import Command
from mcitlib.dataflow import find_gaps
from mcitlib.doctor import FAIL, Report, _keeplora_config_inputs, check_inputs
from mcitlib.paths import PathResolver


@pytest.fixture
def case(tmp_path: Path):
    path_file = tmp_path / "paths.yaml"
    path_file.write_text(yaml.safe_dump({"model_root": str(tmp_path / "models"), "data_root": str(tmp_path / "data"),
                                         "ckpt_root": str(tmp_path / "ckpts")}))
    paths = PathResolver.from_file(path_file, repo_root=tmp_path)
    base = tmp_path / "models/base"
    base.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.model", "pytorch_model.bin"):
        (base / name).write_text("fixture")
    image_dir = tmp_path / "data/images"
    image_dir.mkdir(parents=True)
    (image_dir / "image.jpg").write_text("fixture")
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps([{"image": "image.jpg"}]))
    data_config = tmp_path / "data-config.json"
    data_config.write_text(json.dumps({"train_path": str(data_file), "train_folder": str(image_dir)}))
    model_config = tmp_path / "model-config.json"
    model_config.write_text(json.dumps({"model_name": str(base)}))
    model = tmp_path / "ckpts/merged-model"
    space = tmp_path / "ckpts/previous-subspace"
    train_config = tmp_path / "train-config.json"
    train_config.write_text(json.dumps({"model_path": str(model), "space_path": str(space)}))
    context = {"MCITLIB_PATH_CONTEXT": paths.context("ucit")}
    producer = Command(argv=["python", "merge", "--save-model-path", str(model)], kind="extra", task_id=1)
    weights = Command(argv=["python", "-m", "mcitlib.methods._keeplora.extract_weights",
                            "--model-config", str(model_config), "--output-dir", str(space)],
                      env=context, kind="extra", task_id=1)
    shared = ["python", "-m", "mcitlib.methods._keeplora.extract_gradients",
              "--data-config", str(data_config), "--model-config", str(model_config),
              "--train-config", str(train_config)]
    chunk = Command(argv=shared + ["--output-dir", str(tmp_path / "ckpts/current"),
                                  "--num-chunks", "2", "--chunk-idx", "0"],
                    env=context, kind="extra", task_id=2)
    merge = Command(argv=shared + ["--output-dir", str(tmp_path / "ckpts/current"), "--merge-only",
                                  "--svd-mode", "fixed_rank"], env=context, kind="extra", task_id=2)
    return paths, producer, weights, chunk, merge, train_config, space


def analyze(paths, commands):
    report = Report()
    analyzed, external = _keeplora_config_inputs(report, commands)
    roots = paths.external_roots("ucit") + external
    outputs = (paths.paths.ckpt_root, paths.paths.results_root)
    gaps = find_gaps(analyzed, roots, internal_roots=tuple(map(str, outputs)))
    return report, analyzed, gaps


def test_actual_json_dependencies_have_correct_earlier_producers(case) -> None:
    paths, producer, weights, chunk, merge, _, space = case
    commands = [producer, weights, chunk, merge]
    original = [command.as_dict() for command in commands]
    report, analyzed, gaps = analyze(paths, commands)
    assert not report.failed and not gaps
    assert len(analyzed) == len(commands)
    assert [command.as_dict() for command in commands] == original
    assert ["--output", str(space / "gradients_principle_subspace.pt")] == analyzed[1].argv[-2:]
    assert "--model-path" in analyzed[2].argv
    assert "--regularization_info_path" in analyzed[3].argv


@pytest.mark.parametrize("field,flag", [("model_path", "--model-path"),
                                        ("space_path", "--regularization_info_path")])
def test_wrong_internal_json_path_fails_dataflow(case, field: str, flag: str) -> None:
    paths, producer, weights, chunk, merge, config, _ = case
    values = json.loads(config.read_text())
    values[field] = str(paths.paths.ckpt_root / "missing")
    config.write_text(json.dumps(values))
    _, _, gaps = analyze(paths, [producer, weights, chunk, merge])
    assert any(gap.flag == flag and "/missing" in gap.path for gap in gaps)


def test_later_producer_and_duplicate_json_reads_do_not_hide_earlier_model_gap(case) -> None:
    paths, producer, weights, chunk, merge, _, _ = case
    _, _, gaps = analyze(paths, [chunk, producer, chunk, weights, merge])
    model_gaps = [gap for gap in gaps if gap.flag == "--model-path"]
    assert len(model_gaps) == 1 and model_gaps[0].step == 0


def test_chunk_directory_is_not_a_subspace_producer(case) -> None:
    paths, producer, weights, chunk, merge, _, space = case
    chunk.argv[chunk.argv.index("--output-dir") + 1] = str(space)
    _, _, gaps = analyze(paths, [producer, chunk, merge, weights])
    assert any(gap.flag == "--regularization_info_path" and gap.step == 2
               and gap.path == str(space / "gradients_principle_subspace.pt") for gap in gaps)


def test_external_space_checks_the_actual_file_and_not_just_directory(case, tmp_path: Path) -> None:
    paths, producer, _, chunk, merge, config, _ = case
    external_dir = tmp_path / "external-space"
    external_dir.mkdir()
    values = json.loads(config.read_text())
    values["space_path"] = str(external_dir)
    config.write_text(json.dumps(values))
    report, analyzed, gaps = analyze(paths, [producer, chunk, merge])
    assert not gaps
    outputs = (paths.paths.ckpt_root, paths.paths.results_root)
    check_inputs(report, analyzed, paths.paths.model_root, paths.paths.data_root, outputs)
    assert any(check.status == FAIL and "gradients_principle_subspace.pt" in check.detail for check in report.checks)
    (external_dir / "gradients_principle_subspace.pt").write_text("fixture")
    report = Report()
    check_inputs(report, analyzed, paths.paths.model_root, paths.paths.data_root, outputs)
    assert not any(check.status == FAIL for check in report.checks)


def test_external_model_from_json_is_checked(case, tmp_path: Path) -> None:
    paths, producer, weights, chunk, merge, config, _ = case
    values = json.loads(config.read_text())
    values["model_path"] = str(tmp_path / "missing-external-model")
    config.write_text(json.dumps(values))
    report, analyzed, gaps = analyze(paths, [producer, weights, chunk, merge])
    assert not gaps
    outputs = (paths.paths.ckpt_root, paths.paths.results_root)
    check_inputs(report, analyzed, paths.paths.model_root, paths.paths.data_root, outputs)
    assert any(check.status == FAIL and "missing-external-model" in check.detail for check in report.checks)


def test_run_doctor_uses_analyzed_commands_for_dataflow(case, tmp_path: Path, monkeypatch) -> None:
    from mcitlib.doctor import OK, run_doctor

    _, producer, weights, chunk, merge, config, _ = case
    monkeypatch.setattr("mcitlib.runner.build_sequence_commands", lambda *a, **k: [producer, weights, chunk, merge])
    monkeypatch.setattr("mcitlib.doctor.check_environment", lambda *a, **k: None)
    monkeypatch.setattr("mcitlib.doctor.check_vendored_imports", lambda *a, **k: None)
    monkeypatch.setattr("mcitlib.doctor.check_prerequisites", lambda *a, **k: None)
    report = run_doctor("keeplora", "ucit", tmp_path / "paths.yaml", tmp_path, skip_gpu=True, skip_entrypoints=True)
    assert [item.status for item in report.checks if item.name == "dataflow"] == [OK]
    values = json.loads(config.read_text())
    values["model_path"] = str(tmp_path / "ckpts/missing-model")
    config.write_text(json.dumps(values))
    report = run_doctor("keeplora", "ucit", tmp_path / "paths.yaml", tmp_path, skip_gpu=True, skip_entrypoints=True)
    assert [item.status for item in report.checks if item.name == "dataflow"] == [FAIL]


def test_invalid_json_path_field_is_a_failure_not_a_crash(case) -> None:
    paths, producer, weights, chunk, merge, config, _ = case
    values = json.loads(config.read_text())
    values["space_path"] = 42
    config.write_text(json.dumps(values))
    report, _, _ = analyze(paths, [producer, weights, chunk, merge])
    assert report.failed
    assert any("space_path" in check.detail for check in report.checks if check.status == FAIL)
