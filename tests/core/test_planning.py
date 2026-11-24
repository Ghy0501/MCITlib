# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Plan representative real experiments without loading a model or accessing data."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands


def _arg(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "paths.yaml"
    path.write_text(yaml.safe_dump({name: str(tmp_path / name) for name in (
        "model_root", "data_root", "ckpt_root", "results_root"
    )}), encoding="utf-8")
    return path


@pytest.mark.parametrize("method", ["lora_ft", "replay", "mrlora"])
@pytest.mark.parametrize("benchmark", ["ucit", "mllm_dcl", "mllm_acl"])
def test_real_plan_uses_resolved_paths_and_requested_launcher(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, benchmark: str,
) -> None:
    for key in tuple(os.environ):
        if key.startswith("MCITLIB_"):
            monkeypatch.delenv(key)
    paths = PathResolver.from_file(_profile(tmp_path), repo_root=repo_root)
    commands = build_sequence_commands(repo_root / "configs", "llava", method, benchmark, paths=paths,
                                       task_id=1, gpus="4,6", master_port=29513)
    train = [command for command in commands if command.kind == "train"]
    assert len(train) == 1
    argv = train[0].argv
    assert argv[0] == "deepspeed"
    assert _arg(argv, "--module") == "mcitlib.backbones.llava.train_entry"
    assert _arg(argv, "--include") == "localhost:4,6"
    assert _arg(argv, "--master_port") == "29513"
    assert Path(_arg(argv, "--data_path")).is_relative_to(tmp_path / "data_root")
    assert Path(_arg(argv, "--output_dir")).is_relative_to(tmp_path / "ckpt_root")
    assert not any(command.argv[:2] in (["pip", "install"], ["pip3", "install"])
                   or command.argv[:3] in (["python", "-m", "pip"], ["python3", "-m", "pip"])
                   for command in commands)
    for command in commands:
        if "--answers-file" in command.argv:
            assert Path(_arg(command.argv, "--answers-file")).is_relative_to(tmp_path / "results_root" / method)


def test_router_and_distinct_methods_keep_their_results_separate(repo_root: Path, tmp_path: Path) -> None:
    paths = PathResolver.from_file(_profile(tmp_path), repo_root=repo_root)
    outputs = []
    for method in ("lora_ft", "replay", "mrlora"):
        commands = build_sequence_commands(repo_root / "configs", "llava", method, "ucit", paths=paths,
                                           sequence="router" if method == "mrlora" else "train", task_id=1)
        answers = [Path(_arg(c.argv, "--answers-file")) for c in commands if "--answers-file" in c.argv]
        assert answers
        assert all(path.is_relative_to(tmp_path / "results_root" / method) for path in answers)
        outputs.extend(answers)
    assert len(set(outputs)) == len(outputs)


@pytest.mark.parametrize("working_directory", ["project", "outside"])
def test_cli_preview_uses_the_checkout(repo_root: Path, tmp_path: Path, working_directory: str) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH" and not key.startswith("MCITLIB_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    executable = str(Path(sys.executable).parent / "mcitlib")
    cwd = repo_root if working_directory == "project" else tmp_path
    listing = subprocess.run([executable, "list"], cwd=cwd, env=env, check=True, capture_output=True, text=True)
    assert "17 methods; 4 backbones; 3 benchmarks" in listing.stdout
    preview = subprocess.run(
        [executable, "train", "--method", "lora_ft", "--benchmark", "ucit", "--task", "1", "--dry-run",
         "--paths", str(_profile(tmp_path)), "--gpus", "4,6", "--master-port", "29513"],
        cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    train = next(command for command in json.loads(preview.stdout) if command["kind"] == "train")
    assert _arg(train["argv"], "--data_path") == str(tmp_path / "data_root/UCIT/ImageNet-R/train.json")
    assert train["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(repo_root)
    assert not (tmp_path / "ckpt_root").exists()
