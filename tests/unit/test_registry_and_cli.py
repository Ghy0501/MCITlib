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
from mcitlib.registry import BENCHMARKS, METHODS, get_method, matrix


def test_registry_aliases_and_matrix() -> None:
    assert get_method("RegLoRA").id == "sefe"
    assert get_method("O-LoRA").id == "olora"
    rows = list(matrix())
    assert len(rows) == len(METHODS) * 4 * len(BENCHMARKS)
    llava_rows = [row for row in rows if row["backbone"] == "llava"]
    assert len(llava_rows) == 36
    assert all(row["supported"] for row in llava_rows)
    assert all(not row["supported"] for row in rows if row["backbone"] != "llava")


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
    assert resolver.run_dir("ucit", "hide", 3) == tmp_path / "ckpts/ucit/llava/hide/task3"


def test_cli_list(capsys) -> None:
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "method\tbackbone\tbenchmark" in output
    assert "hide\tllava\tucit\tTrue" in output


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


def test_cli_train_executes_and_resumes(tmp_path: Path, capsys) -> None:
    from unittest.mock import patch

    from mcitlib.config.schema import Command

    output = tmp_path / "executed.txt"
    command = Command(argv=[sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"])
    paths = tmp_path / "paths.yaml"
    paths.write_text("model_root: models\ndata_root: data\nckpt_root: ckpts\n", encoding="utf-8")
    argv = [
        "train", "--method", "lora_ft", "--benchmark", "ucit",
        "--config-root", str(tmp_path / "configs"), "--paths", str(paths),
        "--run-dir", str(tmp_path / "run"),
    ]
    with patch("mcitlib.runner.build_sequence_commands", return_value=[command]):
        assert main(argv) == 0
        first = json.loads(capsys.readouterr().out)
        assert first["steps"][0]["status"] == "completed"
        assert main(argv + ["--resume"]) == 0
        second = json.loads(capsys.readouterr().out)
        assert second["steps"][0]["status"] == "skipped"
    assert output.read_text() == "ok"
