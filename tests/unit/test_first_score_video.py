# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
import json

import pytest

from tools import first_score_video as smoke


@pytest.mark.parametrize("backbone", ["videollava", "videollama2"])
def test_plan_is_bounded_and_separates_reload(tmp_path, backbone):
    run = tmp_path / "new"
    train, generate = smoke.commands(backbone, tmp_path, run)

    def value(argv, key):
        return argv[argv.index("--" + key) + 1]

    assert value(train, "max_steps") == value(train, "gradient_accumulation_steps") == "1"
    assert value(train, "warmup_ratio") == "0"
    assert value(train, "save_strategy") == "no"
    assert "--observe" in train and "--standalone" in train
    assert value(train, "output_dir") == value(generate, "checkpoint")
    assert "generate" in generate and "--merged" not in generate
    assert not run.exists()


@pytest.mark.parametrize("backbone", ["videollava", "videollama2"])
def test_private_model_view_leaves_snapshot_untouched(tmp_path, backbone):
    snapshot = tmp_path / ".hub-cache" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    source = tmp_path / (smoke.NATIVE[backbone] + "-7B")
    source.symlink_to(snapshot, target_is_directory=True)
    (source / "config.json").write_text('{"old": true}')
    (source / "tokenizer.model").touch()
    for name in smoke.TOWERS[backbone].values():
        target = tmp_path / ".hub-cache" / name / "rev"
        target.mkdir(parents=True)
        (tmp_path / name).symlink_to(target, target_is_directory=True)
    view = tmp_path / "view"
    smoke.model_view(backbone, tmp_path, view)
    conf = json.loads((view / "config.json").read_text())
    for key, name in smoke.TOWERS[backbone].items():
        assert conf[key] == str(tmp_path / name)
    assert (source / "config.json").read_text() == '{"old": true}'
    assert (view / "tokenizer.model").is_symlink()


def test_exact_match_requires_one_matching_id_and_real_text(tmp_path):
    path = tmp_path / "predictions.jsonl"

    def write(pred="Blue.", qid="eval-blue"):
        path.write_text(json.dumps({"question_id": qid, "pred": pred}) + "\n")

    answers = [{"question_id": "eval-blue", "answer": "blue"}]
    write()
    assert smoke.score(path, answers)["correct"] == 0  # punctuation counts
    write(" BLUE ")
    assert smoke.score(path, answers)["correct"] == 1
    write(qid="wrong")
    with pytest.raises(ValueError):
        smoke.score(path, answers)
    write("  ")
    with pytest.raises(ValueError):
        smoke.score(path, answers)
    write()
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError):
        smoke.score(path, answers)


@pytest.mark.parametrize("visible,lease", [("", "2,3"), ("0", "2,3"), ("2,3", "2,3")])
def test_refuses_ambiguous_or_unallocated_devices(monkeypatch, visible, lease):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", lease)
    with pytest.raises(ValueError):
        smoke.free_gpu()


@pytest.mark.parametrize("memory,utilization,pid", [(2, 0, "123"), (1024, 0, ""), (0, 3, "")])
def test_busy_gpu_fails_closed(monkeypatch, memory, utilization, pid):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", "2,3")

    def inventory(argv, text):
        assert argv[argv.index("-i") + 1] == "2"
        return pid if "compute-apps" in argv[-1] else f"GPU-uuid,{memory},{utilization}"

    monkeypatch.setattr(smoke.subprocess, "check_output", inventory)
    with pytest.raises(RuntimeError, match="not free"):
        smoke.free_gpu()
