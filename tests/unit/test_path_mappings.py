# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Path mappings must preserve command semantics across inputs and extra steps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mcitlib.config.schema import Command
from mcitlib.dataflow import find_gaps
from mcitlib.doctor import FAIL, OK, Report, check_inputs
from mcitlib.methods._keeplora.config_paths import resolve_fields
from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands


def _resolver(tmp_path: Path, mappings: dict) -> PathResolver:
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"model_root": "models", "data_root": "data", "ckpt_root": "ckpts",
                                      "path_mappings": mappings}), encoding="utf-8")
    return PathResolver.from_file(config, repo_root=tmp_path)


def test_longest_prefix_boundary_and_benchmark_scope(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MCITLIB_DATA_ROOT", str(tmp_path / "override"))
    mappings = {
        "ucit": [{"from": "${data_root}/UCIT", "to": str(tmp_path / "images")},
                 {"from": "/your_data_path/UCIT/ImageNet-R", "to": str(tmp_path / "annotations")}],
        "mllm_dcl": [{"from": "${data_root}/MLLM-CL", "to": str(tmp_path / "domain")}],
        "mllm_acl": [{"from": "/your_data_path/MLLM-CL", "to": str(tmp_path / "ability")}],
    }
    resolver = _resolver(tmp_path, mappings)
    assert resolver.resolve_templates("${data_root}/UCIT", "ucit") == str(tmp_path / "images")
    assert resolver.resolve_templates("${data_root}/UCIT/ImageNet-R/train.json", "ucit") == str(tmp_path / "annotations/train.json")
    assert resolver.resolve_templates("/your_data_path/UCIT/ImageNet-R-other", "ucit") == str(tmp_path / "images/ImageNet-R-other")
    assert resolver.resolve_templates("${data_root}/UCIT2", "ucit") == str(tmp_path / "override/UCIT2")
    assert resolver.resolve("/your_data_path/MLLM-CL/RS", benchmark="mllm_dcl") == tmp_path / "domain/RS"
    assert resolver.resolve("MLLM-CL/RS", benchmark="mllm_acl") == tmp_path / "ability/RS"
    assert resolver.resolve_templates("${data_root}/MLLM-CL", "mllm_acl") == str(tmp_path / "ability")


def test_conflicts_and_checkpoint_overlap_are_rejected(tmp_path: Path) -> None:
    source = "${data_root}/UCIT"
    with pytest.raises(ValueError, match="conflicting path mappings"):
        _resolver(tmp_path, {"common": [{"from": source, "to": "/a"}],
                             "ucit": [{"from": "/your_data_path/UCIT", "to": "/b"}]})
    with pytest.raises(ValueError, match="overlaps checkpoint/results root"):
        _resolver(tmp_path, {"ucit": [{"from": source, "to": "${ckpt_root}/input"}]})


def test_train_eval_and_keeplora_extra_use_the_same_mapping(repo_root: Path, tmp_path: Path) -> None:
    resolver = _resolver(tmp_path, {"ucit": [
        {"from": "${data_root}/UCIT", "to": str(tmp_path / "images")},
        {"from": "${data_root}/UCIT/ImageNet-R", "to": str(tmp_path / "annotations")},
    ]})
    commands = build_sequence_commands(repo_root / "configs", "llava", "keeplora", "ucit", paths=resolver, task_id=1)
    train = next(command for command in commands if command.kind == "train")
    evaluate = next(command for command in commands if "--question-file" in command.argv)
    assert train.argv[train.argv.index("--data_path") + 1] == str(tmp_path / "annotations/train.json")
    assert train.argv[train.argv.index("--image_folder") + 1] == str(tmp_path / "images")
    assert evaluate.argv[evaluate.argv.index("--question-file") + 1] == str(tmp_path / "annotations/test_3000.json")
    assert evaluate.argv[evaluate.argv.index("--image-folder") + 1] == str(tmp_path / "images")
    extra = next(command for command in commands if "mcitlib.methods._keeplora.extract_gradients" in command.argv)
    data = json.loads((repo_root / "configs/data_configs/UCIT/ImageNet-R.json").read_text())
    mapped = resolve_fields(data, ("train_path", "train_folder"), extra.env["MCITLIB_PATH_CONTEXT"])
    assert mapped["train_path"] == str(tmp_path / "annotations/train.json")
    assert mapped["train_folder"] == str(tmp_path / "images")
    assert mapped["test_path"] == data["test_path"]
    assert str(tmp_path / "ckpts") in train.argv[train.argv.index("--output_dir") + 1]


def test_mapping_is_applied_once_to_train_and_eval(repo_root: Path, tmp_path: Path) -> None:
    resolver = _resolver(tmp_path, {"ucit": [
        {"from": "${data_root}/UCIT", "to": "${data_root}/UCIT/mounted"},
        {"from": "${data_root}/UCIT/mounted", "to": str(tmp_path / "would_be_a_second_mapping")},
    ]})
    commands = build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "ucit", paths=resolver, task_id=1)
    train = next(command for command in commands if command.kind == "train")
    evaluate = next(command for command in commands if "--question-file" in command.argv)
    assert train.argv[train.argv.index("--data_path") + 1] == str(tmp_path / "data/UCIT/mounted/ImageNet-R/train.json")
    assert evaluate.argv[evaluate.argv.index("--question-file") + 1] == str(tmp_path / "data/UCIT/mounted/ImageNet-R/test_3000.json")


def test_doctor_checks_external_files_and_distinguishes_outputs(tmp_path: Path) -> None:
    model, data, output, external = (tmp_path / name for name in ("model", "data", "ckpt", "external"))
    external.mkdir()
    commands = [Command(argv=["--data_path", str(external / "missing.json"), "--model-path", str(output / "task1")])]
    report = Report()
    check_inputs(report, commands, model, data, (output,))
    failures = [check.detail for check in report.checks if check.status == FAIL]
    assert any("missing.json" in detail for detail in failures)
    assert all("task1" not in detail for detail in failures)
    commands = [Command(argv=["--model-base", str(external / "model"), "--model-path", str(output / "task1")])]
    gaps = find_gaps(commands, (str(model), str(data)), internal_roots=(str(output),))
    assert len(gaps) == 1 and gaps[0].path == str(output / "task1")


def test_doctor_checks_weight_shards_and_sample_images(tmp_path: Path) -> None:
    model, images, data, output = (tmp_path / name for name in ("model", "images", "data.json", "ckpt"))
    model.mkdir()
    images.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.model", "part1.bin"):
        (model / name).write_text("content")
    (model / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"a": "part1.bin", "b": "part2.bin"}}))
    data.write_text(json.dumps([{"image": "one.jpg"}]))
    command = Command(argv=["--model_name_or_path", str(model), "--data_path", str(data), "--image_folder", str(images)])
    report = Report()
    check_inputs(report, [command], model, tmp_path, (output,))
    assert any(check.name == "model contents" and check.status == FAIL and "part2.bin" in check.detail for check in report.checks)
    assert any(check.name == "sample images" and check.status == FAIL and "one.jpg" in check.detail for check in report.checks)
    (model / "part2.bin").write_text("content")
    (images / "one.jpg").write_text("image")
    report = Report()
    check_inputs(report, [command], model, tmp_path, (output,))
    assert all(check.status == OK for check in report.checks)


def test_dataflow_still_requires_producers_outside_configured_output_roots(tmp_path: Path) -> None:
    model, data, output, other = (tmp_path / name for name in ("model", "data", "ckpt", "other"))
    commands = [Command(argv=["--model-base", str(other / "provided-model"),
                              "--previous_task_model_path", str(other / "missing-checkpoint"),
                              "--result-file", str(other / "missing-results.jsonl")])]
    gaps = find_gaps(commands, (str(model), str(data)), internal_roots=(str(output),))
    assert {gap.flag for gap in gaps} == {"--previous_task_model_path", "--result-file"}
    report = Report()
    check_inputs(report, commands, model, data, (output,))
    assert any(check.status == FAIL and "missing-checkpoint" in check.detail for check in report.checks)


def test_mapping_cannot_reach_output_roots_through_symlinks(tmp_path: Path) -> None:
    (tmp_path / "ckpts").mkdir()
    (tmp_path / "link-to-ckpts").symlink_to(tmp_path / "ckpts", target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps checkpoint/results root"):
        _resolver(tmp_path, {"ucit": [{"from": "${data_root}/UCIT", "to": str(tmp_path / "link-to-ckpts/input")} ]})
