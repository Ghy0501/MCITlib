# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""The standalone project owns run records even when checkpoints are external."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import mcitlib
from mcitlib.config.schema import Command
from mcitlib.paths import PathResolver
from mcitlib.runner import execute_commands


def test_project_root_owns_package_config_and_vendored_code(repo_root: Path) -> None:
    assert Path(mcitlib.__file__).resolve() == repo_root / "mcitlib/__init__.py"
    assert (repo_root / "configs/methods/lora_ft.yaml").is_file()
    assert (repo_root / "third_party/LLaVA/llava").is_dir()
    assert (repo_root / "third_party/peft").is_dir()


def test_config_is_anchored_to_project_from_any_cwd(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs/paths.yaml").write_text("model_root: models\ndata_root: data\nckpt_root: ../weights\n")
    monkeypatch.setattr("mcitlib.paths.repo_root", lambda: project)
    for name in ("MODEL_ROOT", "DATA_ROOT", "CKPT_ROOT", "LOGS_ROOT", "RESULTS_ROOT"):
        monkeypatch.delenv(f"MCITLIB_{name}", raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for config in (None, Path("configs/paths.yaml")):
        paths = PathResolver.from_file(config)
        assert paths.paths.model_root == project / "models"
        assert paths.paths.data_root == project / "data"
        assert paths.paths.ckpt_root == tmp_path / "weights"
        assert paths.results_root == project / "results"
        assert paths.run_dir("ucit", "lora_ft") == project / "logs/ucit/llava/lora_ft"
    assert not (elsewhere / "logs").exists()


def test_managed_paths_reject_traversal_and_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "logs").mkdir()
    (project / "logs/escape").symlink_to(tmp_path, target_is_directory=True)
    paths = PathResolver.from_file(repo_root=project)
    for candidate in ("../outside", "logs/../../outside", "logs/escape/outside", tmp_path / "outside"):
        with pytest.raises(ValueError, match="inside"):
            paths.managed_run_dir(candidate)
    (project / "results").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="default results_root"):
        PathResolver.from_file(repo_root=project)


def test_execution_keeps_training_logs_inside_project_and_preserves_attempts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capture = project / "argv.json"
    external_weights = tmp_path / "external-checkpoints"
    command = Command(
        argv=[sys.executable, "-c", f"import json,sys; open({str(capture)!r},'w').write(json.dumps(sys.argv))",
              "mcitlib.backbones.llava.train_entry", "--output_dir", str(external_weights / "task")],
        kind="train",
    )
    run_dir = project / "logs/experiment"
    execute_commands([command], run_dir, project)
    executed = json.loads(capture.read_text())
    assert executed[executed.index("--output_dir") + 1] == str(external_weights / "task")
    logging_dir = Path(executed[executed.index("--logging_dir") + 1])
    assert run_dir in logging_dir.parents
    assert not external_weights.exists()
    steps = run_dir / ".mcitlib/steps"
    old_marker = next(steps.glob("*.json"))
    before = old_marker.read_bytes()
    execute_commands([command], run_dir, project)
    assert old_marker.read_bytes() == before
    assert len(list(steps.glob("*.json"))) == 2
    assert len(list((run_dir / ".mcitlib/logs").glob("*.log"))) == 2
    with pytest.raises(ValueError, match="inside"):
        execute_commands([], external_weights / "run", project)
