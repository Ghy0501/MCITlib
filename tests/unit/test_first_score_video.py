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


@pytest.mark.parametrize("mask", ["2", "GPU-12345678-1234-1234-1234-123456789abc", None])
def test_worker_env_inherits_cuda_settings(tmp_path, monkeypatch, mask):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_CACHE_PATH", "/custom/cuda-cache")
    env = smoke.worker_env(tmp_path, tmp_path / "sources")
    assert env.get("CUDA_VISIBLE_DEVICES") == mask
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_CACHE_PATH"] == "/custom/cuda-cache"
