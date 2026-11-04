# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""General-benchmark command generation and private-model parameter forwarding."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands


def test_general_eval_builds_four_ordered_inference_merge_score_groups(repo_root: Path) -> None:
    commands = build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "general", only_eval=True)
    assert len(commands) == 16
    assert [name for name in dict.fromkeys(c.metadata["general_benchmark"] for c in commands)] == [
        "pope", "mme", "mmbench", "seed",
    ]
    for name in ("pope", "mme", "mmbench", "seed"):
        group = [c for c in commands if c.metadata["general_benchmark"] == name]
        assert len(group) == 4
        assert group[-2].argv[3] == "merge"
        assert group[-1].argv[3:5] == ["score", "--benchmark"]
        assert group[-1].argv[5] == name
        output = group[-1].argv[group[-1].argv.index("--output") + 1]
        assert output.endswith("score.json")
    for command in commands:
        for index, value in enumerate(command.argv):
            if value == "-m":
                assert importlib.util.find_spec(command.argv[index + 1]) is not None
        if "eval_runners.entry" in command.argv:
            target = command.argv[command.argv.index("eval_runners.entry") + 1]
            assert importlib.util.find_spec(target) is not None


def test_general_eval_forwards_modalprompt_and_hide_parameters(repo_root: Path) -> None:
    modal = build_sequence_commands(repo_root / "configs", "llava", "modalprompt", "general", only_eval=True)
    inference = modal[0].argv
    for flag, value in (("--prefix-len", "20"), ("--cur-task", "5"), ("--num-tasks", "5")):
        assert inference[inference.index(flag) + 1] == value
    assert "--text-tower" in inference
    assert modal[0].env["MCITLIB_MODEL_OVERRIDE"] == "modalprompt"

    hide = build_sequence_commands(repo_root / "configs", "llava", "hide", "general", only_eval=True)
    assert hide[0].argv[hide[0].argv.index("--num-task") + 1] == "5"
    assert hide[0].env["MCITLIB_MODEL_OVERRIDE"] == "hide"


def test_general_model_override_chunks_and_path_resolution(repo_root: Path, tmp_path: Path) -> None:
    path_config = tmp_path / "paths.yaml"
    path_config.write_text("model_root: models\ndata_root: data\nckpt_root: ckpts\n", encoding="utf-8")
    paths = PathResolver.from_file(path_config, repo_root=repo_root)
    commands = build_sequence_commands(
        repo_root / "configs", "llava", "lora_ft", "general", only_eval=True,
        model_path="${ckpt_root}/custom", model_base="${model_root}/base", num_chunks=1, paths=paths,
    )
    assert len(commands) == 12
    inference = commands[0].argv
    assert inference[inference.index("--model-path") + 1] == str(repo_root / "ckpts/custom")
    assert inference[inference.index("--model-base") + 1] == str(repo_root / "models/base")
    assert "${" not in json.dumps([command.as_dict() for command in commands])
    with pytest.raises(ValueError, match="only by the eval command"):
        build_sequence_commands(repo_root / "configs", "llava", "lora_ft", "general")


def test_general_eval_cli_dry_run(repo_root: Path, capsys) -> None:
    from mcitlib.cli import main

    assert main(["eval", "--method", "lora_ft", "--benchmark", "general", "--after-task", "5", "--dry-run"]) == 0
    commands = json.loads(capsys.readouterr().out)
    assert len(commands) == 16
    assert all(command["sequence"] == "general" for command in commands)
