# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Public CL-VISTA plans and resource boundaries, without GPU dependencies."""
import json
from pathlib import Path

import pytest

from mcitlib.backbones.video.binding import template
from mcitlib.backbones.video.cli import selected_gpus
from mcitlib.backbones.video.workflow import TASKS, build_cl_vista_plan
from mcitlib.cli import main
from mcitlib.registry import VIDEO_BACKBONES, VIDEO_METHODS


@pytest.mark.parametrize("backbone", sorted(VIDEO_BACKBONES))
@pytest.mark.parametrize("method", sorted(VIDEO_METHODS))
def test_public_plans(repo_root, tmp_path, backbone, method):
    node = template(tmp_path)
    commands = build_cl_vista_plan(repo_root / "configs", node, backbone, method, "sample", "public",
                                   gpu_count=2 if method == "mrlora" else 4, judge_directory=tmp_path)
    scores = [x for x in commands if x.kind == "score"]
    assert len(scores) == (128 if method == "mrlora" else 36)
    assert {x.task_id for x in scores} == set(range(1, len(TASKS) + 1))
    for command in commands:
        for arg in command.argv:
            if arg.endswith(".py") or (arg.startswith(str(repo_root)) and arg.endswith(".json")):
                assert Path(arg).is_file(), arg


def test_release_gpu_assignment_and_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MCITLIB_VIDEO_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5")
    node = tmp_path / "node.json"
    node.write_text(json.dumps(template(tmp_path)))
    assert main(["train", "--backbone", "Video-LLaVA", "--benchmark", "CL-VISTA", "--method", "lora_ft",
                 "--video-node", str(node), "--gpus", "4,5", "--task", "1", "--dry-run"]) == 0
    train, generate, score = json.loads(capsys.readouterr().out)
    assert train["argv"][:3] == ["deepspeed", "--include", "localhost:4,5"]
    assert train["argv"].count("--include") == 1
    assert Path(train["env"]["PYTHONPATH"]) == tmp_path.resolve() / "sources/Video-LLaVA"
    assert generate["env"]["PYTHONPATH"] == train["env"]["PYTHONPATH"]
    assert "CUDA_VISIBLE_DEVICES" not in train["env"]
    assert generate["env"]["CUDA_VISIBLE_DEVICES"] == score["env"]["CUDA_VISIBLE_DEVICES"] == "4,5"
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", "4,5")
    with pytest.raises(ValueError, match="outside"):
        selected_gpus("3", "0")
    for invalid in ("4,4", "-1", "0, 1", "", "all"):
        with pytest.raises(ValueError):
            selected_gpus(invalid, "0")


def test_wrong_benchmark_fails_before_launch(capsys):
    assert main(["doctor", "--backbone", "videollama2", "--method", "lora_ft", "--benchmark", "ucit"]) == 2
    assert "CL-VISTA" in capsys.readouterr().err
