# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
import json
from types import SimpleNamespace

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


GPU = "GPU-12345678-1234-1234-1234-123456789abc"
OTHER = "GPU-aaaaaaaa-1234-1234-1234-123456789abc"


@pytest.mark.parametrize("visible", [None, "", "-1", "2,3", "GPU-1234", "MIG-dev", "2,"])
def test_refuses_ambiguous_or_missing_mask_before_driver(monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(smoke, "logical_gpu_uuid", lambda: pytest.fail("unexpected CUDA probe"))
    with pytest.raises(ValueError):
        smoke.free_gpu()


def mock_inventory(monkeypatch, *, index="5", uuid=GPU, memory=0, utilization=0, pid=""):
    monkeypatch.setattr(smoke, "logical_gpu_uuid", lambda: GPU)

    def inventory(argv, **kwargs):
        assert argv[argv.index("-i") + 1] == GPU
        return pid if "compute-apps" in argv[-1] else f"{index},{uuid},{memory},{utilization}"

    monkeypatch.setattr(smoke.subprocess, "check_output", inventory)


@pytest.mark.parametrize("visible", ["2", GPU])
def test_probe_joins_by_driver_uuid_and_preserves_child_mask(tmp_path, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", "5")  # physical index, not CUDA ordinal 2
    mock_inventory(monkeypatch)
    assert smoke.free_gpu() == GPU
    env = smoke.worker_env(tmp_path, tmp_path / "sources")
    assert env["CUDA_VISIBLE_DEVICES"] == visible
    assert env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_matching_ordinal_is_not_proof_of_physical_lease(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", "2,3")
    mock_inventory(monkeypatch, index="5")
    with pytest.raises(ValueError, match="lease"):
        smoke.free_gpu()


@pytest.mark.parametrize("mask,reported", [(OTHER, GPU), ("2", OTHER)])
def test_conflicting_identities_fail_closed(monkeypatch, mask, reported):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.delenv("MCITLIB_VIDEO_GPUS", raising=False)
    mock_inventory(monkeypatch, uuid=reported)
    with pytest.raises(RuntimeError, match="UUID"):
        smoke.free_gpu()


@pytest.mark.parametrize("count", [0, 2])
def test_driver_requires_exactly_one_device(monkeypatch, count):
    class Driver:
        def cuInit(self, flags):
            return 0

        def cuDeviceGetCount(self, ptr):
            ptr._obj.value = count
            return 0

    monkeypatch.setattr(smoke.ctypes, "CDLL", lambda lib: Driver())
    with pytest.raises(RuntimeError, match="exactly one visible CUDA device"):
        smoke.logical_gpu_uuid()


def test_driver_failure_never_falls_back_to_inventory(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(smoke.ctypes, "CDLL", lambda lib: SimpleNamespace(cuInit=lambda flags: 100))
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *a, **kw: pytest.fail("unexpected fallback"))
    with pytest.raises(RuntimeError, match="cuInit returned 100"):
        smoke.free_gpu()


@pytest.mark.parametrize("memory,utilization,pid", [(2, 0, "123"), (1024, 0, ""), (0, 3, "")])
def test_busy_gpu_fails_closed(monkeypatch, memory, utilization, pid):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", "5")
    mock_inventory(monkeypatch, memory=memory, utilization=utilization, pid=pid)
    with pytest.raises(RuntimeError, match="not free"):
        smoke.free_gpu()


@pytest.mark.parametrize("mask", ["2", GPU])
def test_actual_stage_launches_keep_original_allocation(tmp_path, monkeypatch, mask):
    run = tmp_path / "run"
    native = tmp_path / "sources/Video-LLaVA/videollava"
    native.mkdir(parents=True)
    (native / "__init__.py").touch()
    monkeypatch.setattr(smoke.sys, "argv", ["smoke", "videollava", "--models", str(tmp_path / "models"),
                                            "--sources", str(tmp_path / "sources"), "--run-dir", str(run)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("MCITLIB_VIDEO_GPUS", "5")
    mock_inventory(monkeypatch)
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=8 * 2**30))
    monkeypatch.setattr(smoke, "model_view", lambda *_: None)
    monkeypatch.setattr(smoke, "fixture", lambda _: smoke.dump(run / "answers.json",
                                                               [{"question_id": "id", "answer": "blue"}]))
    stages = []

    def launch(argv, *, env, **kwargs):
        assert env["CUDA_VISIBLE_DEVICES"] == mask
        if "train" in argv:
            stages.append("train")
            ckpt = run / "videollava-lora"
            ckpt.mkdir()
            for f in ("config.json", "adapter_config.json", "non_lora_trainables.bin", "adapter_model.bin"):
                (ckpt / f).write_bytes(b"x")
        else:
            assert "generate" in argv
            stages.append("generate")
            (run / "predictions.jsonl").write_text('{"question_id":"id","pred":"blue"}\n')

    monkeypatch.setattr(smoke.subprocess, "run", launch)
    smoke.main()
    assert stages == ["train", "generate"]
    receipt = json.loads((run / "plan.json").read_text())
    assert (receipt["cuda_visible_devices"], receipt["gpu_uuid"]) == (mask, GPU)
