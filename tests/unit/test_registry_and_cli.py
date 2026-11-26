# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Registry, layering, paths and CLI smoke tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mcitlib.cli import main
from mcitlib.config.loader import deep_merge
from mcitlib.paths import PathResolver
from mcitlib.registry import BENCHMARKS, METHODS, VIDEO_BACKBONES, VIDEO_METHODS, get_method, matrix


def test_registry_aliases_and_matrix() -> None:
    assert get_method("RegLoRA").id == "sefe"
    assert get_method("O-LoRA").id == "olora"
    rows = list(matrix())
    assert len(rows) == len(METHODS) * (2 * len(BENCHMARKS) + len(VIDEO_BACKBONES))
    llava_rows = [row for row in rows if row["backbone"] == "llava"]
    assert len(llava_rows) == 51
    new_methods = {"pclr", "same", "sigma", "dgg", "octopus"}
    assert all(row["supported"] == (row["method"] not in new_methods or row["benchmark"] != "mllm_acl")
               for row in llava_rows)
    internvl = [row for row in rows if row["backbone"] == "internvl" and row["supported"]]
    assert {(row["method"], row["benchmark"]) for row in internvl} == {
        (method.id, benchmark) for method in METHODS if method.id not in new_methods
        for benchmark in ("ucit", "mllm_dcl")
    }
    videos = [row for row in rows if row["backbone"] in VIDEO_BACKBONES]
    assert len(videos) == 34 and {row["benchmark"] for row in videos} == {"cl_vista"}
    assert {row["method"] for row in videos if row["supported"]} == VIDEO_METHODS


def test_deep_merge_does_not_mutate_base() -> None:
    base = {"a": {"b": 1, "c": 2}, "x": 1}
    result = deep_merge(base, {"a": {"b": 3}, "y": 4})
    assert result == {"a": {"b": 3, "c": 2}, "x": 1, "y": 4}
    assert base == {"a": {"b": 1, "c": 2}, "x": 1}


def test_path_resolution_and_environment_override(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "paths.yaml"
    config.write_text("model_root: models\ndata_root: data\nckpt_root: ckpts\n", encoding="utf-8")
    monkeypatch.setenv("MCITLIB_MODEL_ROOT", str(tmp_path / "override-models"))
    resolver = PathResolver.from_file(config, repo_root=tmp_path)
    assert resolver.resolve("/your_model_path/llava", "model") == tmp_path / "override-models/llava"
    assert resolver.resolve("UCIT/train.json", "data") == tmp_path / "data/UCIT/train.json"
    assert resolver.run_dir("ucit", "hide", 3) == tmp_path / "logs/ucit/llava/hide/task3"


def test_cli_list(capsys) -> None:
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "method\tbackbone\tbenchmark" in output
    assert "hide\tllava\tucit\tTrue" in output


def test_config_expand_uses_selected_backbone(capsys) -> None:
    assert main(['config', 'expand', '--backbone', 'internvl', '--method', 'lora_ft',
                 '--benchmark', 'ucit', '--task', '2']) == 0
    config = json.loads(capsys.readouterr().out)
    assert config['backbone'] == 'internvl'
    assert config['model_config']['vision_tower'].endswith('/InternViT-6B-224px')
    assert config['train_config']['previous_model'].endswith('/Task1_internvl_lora')
    train = config['train_command']
    assert train['args']['vision_tower'] == config['model_config']['vision_tower']
    assert train['args']['model_name_or_path'] == config['model_config']['model_name']
    assert 'mm_projector_lr' not in train['args']
    assert main(['config', 'expand', '--backbone', 'videollava', '--method', 'lora_ft',
                 '--benchmark', 'ucit', '--task', '1']) == 2


def test_config_validate_covers_all_internvl_tasks_and_router(capsys) -> None:
    assert main(['config', 'validate', '--backbone', 'internvl', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result['methods'], result['benchmarks'], result['tasks']) == (12, 2, 143)
    assert result['recorded_commands'] > result['tasks']


def test_cli_train_and_eval_dry_run(capsys) -> None:
    assert main(["train", "--backbone", "llava", "--method", "lora_ft", "--benchmark", "ucit", "--task", "1", "--dry-run"]) == 0
    train = json.loads(capsys.readouterr().out)
    assert any(command["kind"] == "train" for command in train)
    assert any(command["kind"] == "eval" for command in train)
    assert all(command["kind"] != "install" for command in train)

    assert main(["eval", "--backbone", "llava", "--method", "lora_ft", "--benchmark", "ucit", "--after-task", "1", "--dry-run"]) == 0
    evaluate = json.loads(capsys.readouterr().out)
    assert evaluate
    assert all(command["kind"] == "eval" for command in evaluate)


def test_cli_train_executes_and_resumes(tmp_path: Path, capsys, monkeypatch) -> None:
    from unittest.mock import patch

    from mcitlib.config.schema import Command

    monkeypatch.setattr("mcitlib.paths.repo_root", lambda: tmp_path)
    output = tmp_path / "executed.txt"
    command = Command(argv=[sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"])
    paths = tmp_path / "paths.yaml"
    paths.write_text("model_root: models\ndata_root: data\nckpt_root: ckpts\n", encoding="utf-8")
    argv = [
        "train", "--method", "lora_ft", "--benchmark", "ucit",
        "--config-root", str(tmp_path / "configs"), "--paths", str(paths),
        "--run-dir", str(tmp_path / "logs/run"),
    ]
    with patch("mcitlib.runner.build_sequence_commands", return_value=[command]):
        assert main(argv) == 0
        first = json.loads(capsys.readouterr().out)
        assert first["steps"][0]["status"] == "completed"
        assert main(argv + ["--resume"]) == 0
        second = json.loads(capsys.readouterr().out)
        assert second["steps"][0]["status"] == "skipped"
    assert output.read_text() == "ok"
