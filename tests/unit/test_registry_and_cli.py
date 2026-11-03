# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Registry, layering, paths and CLI smoke tests."""
from __future__ import annotations

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
