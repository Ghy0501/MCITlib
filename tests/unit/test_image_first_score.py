# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mcitlib import smoke


@pytest.mark.parametrize("backbone", ["llava", "internvl"])
def test_one_device_first_score_plan(tmp_path, repo_root, monkeypatch, backbone):
    from mcitlib.smoke_images import score_plan

    (tmp_path / "configs").symlink_to(repo_root / "configs")
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(smoke, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    plan = score_plan(None, tmp_path / "logs/score", backbone, "1", 19021)
    train, infer = plan["training_command"], plan["inference_command"]
    assert train["env"]["CUDA_VISIBLE_DEVICES"] == infer["env"]["CUDA_VISIBLE_DEVICES"] == "6"
    assert not any("CUDA_DEVICE_ORDER" in cmd["env"] for cmd in (train, infer))
    assert "--nproc_per_node=1" in train["argv"]
    assert smoke._value(train["argv"], "--max_steps") == "1"
    assert smoke._value(train["argv"], "--warmup_ratio") == "0"
    assert smoke._value(infer["argv"], "--subset").endswith("/heldout-question.json")
    assert smoke._value(infer["argv"], "--adapter") == smoke._value(train["argv"], "--output_dir")
    assert plan["test_data"].endswith("/logs/score/fixture/test.json")
    assert plan["source_data"].endswith("/logs/score/fixture/train.json")
    assert plan["dataset_scope"] == "generated fixture; non-benchmark"
    assert smoke._value(train["argv"], "--image_folder") == smoke._value(infer["argv"], "--image-folder")


def test_gpu_allocation(monkeypatch):
    from mcitlib.smoke_images import selected_gpu

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert selected_gpu("7") == "7"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,GPU-abcd")
    assert selected_gpu("1") == "GPU-abcd"
    with pytest.raises(ValueError):
        selected_gpu("2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert selected_gpu("0") == "0"  # read-only planning
    with pytest.raises(ValueError):
        selected_gpu("0", require_mask=True)


@pytest.mark.parametrize("mask,index", [(None, "0"), ("", "0"), (" ", "0"), ("-1", "0"),
                         ("4,-1,7", "2"), ("4,,7", "2"), ("4,4", "0"), ("GPU-not-a-device", "0"),
                         ("4,7", "2"), ("4,7", "-1")])
def test_execution_requires_valid_allocation_before_any_side_effect(tmp_path, monkeypatch, mask, index):
    from mcitlib.smoke_images import execute

    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(smoke, "build_plan", lambda *a, **kw: pytest.fail("invalid allocation reached planner"))
    monkeypatch.setattr(smoke, "_run", lambda *a, **kw: pytest.fail("invalid allocation reached a worker"))
    with pytest.raises(ValueError, match="GPU|CUDA_VISIBLE_DEVICES"):
        execute(None, tmp_path / "score", "llava", index, 19021)
    assert not (tmp_path / "score").exists()


def test_local_tower_view_preserves_source_and_does_not_copy_weights(tmp_path):
    from mcitlib.smoke_images import local_model_view

    source, target = tmp_path / "snapshot", tmp_path / "view"
    source.mkdir()
    cfg = {"mm_vision_tower": "openai/or-absolute-author-path", "hidden_size": 4096}
    (source / "config.json").write_text(json.dumps(cfg))
    (source / "pytorch_model.bin").write_bytes(b"weights")
    tower = tmp_path / "clip-vit-large-patch14-336"
    local_model_view(source, target, tower)
    assert json.loads((target / "config.json").read_text()) == {**cfg, "mm_vision_tower": str(tower)}
    assert json.loads((source / "config.json").read_text()) == cfg
    assert (target / "pytorch_model.bin").samefile(source / "pytorch_model.bin")


def test_unavailable_device_fails_before_creating_run(tmp_path, monkeypatch):
    from mcitlib import smoke_images

    run = tmp_path / "run"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "999")
    monkeypatch.setattr(smoke_images, "score_plan", lambda *a, **kw: {"run_dir": str(run)})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0)))
    with pytest.raises(ValueError, match="CUDA cannot resolve"):
        smoke_images.execute(None, run, "llava", "0", 19021)
    assert not run.exists()


def test_real_native_metric_and_missing_score(tmp_path):
    from mcitlib.smoke_images import SCORER, read_score

    with pytest.raises((OSError, ValueError)):
        read_score(tmp_path)
    annotation = dict(question_id=17, text="What color?", image="a.png", answer="red")
    labels = tmp_path / "test.json"
    labels.write_text(json.dumps([annotation]))
    for prediction, expected in [("red", 100), ("blue", 0)]:
        predictions = tmp_path / "answers.jsonl"
        predictions.write_text(json.dumps(dict(question_id=17, prompt="What color?", text=prediction)) + "\n")
        subprocess.run([sys.executable, "-m", SCORER, "--annotation-file", str(labels),
                        "--result-file", str(predictions), "--output-dir", str(tmp_path)], check=True)
        assert read_score(tmp_path) == expected
    for malformed in ["Samples: 2\nAccuracy: 0.00%\n", "Samples: 1\nAccuracy: nan%\n",
                      "Samples: 1\nAccuracy: 1.00%\n", "Accuracy: 100.00%\n"]:
        (tmp_path / "Result.text").write_text(malformed)
        with pytest.raises(ValueError):
            read_score(tmp_path)


def test_generated_artifacts_and_heldout_path_protection(tmp_path):
    from PIL import Image

    from mcitlib.smoke_images import generate_fixture, heldout

    folder = tmp_path / "data"
    generate_fixture(folder)
    train = json.loads((folder / "train.json").read_text())[0]
    test = json.loads((folder / "test.json").read_text())[0]
    for name, color in [("train.png", (255, 0, 0)), ("test.png", (0, 0, 255))]:
        with Image.open(folder / name) as image:
            assert image.getpixel((64, 64)) == color
    question = heldout(train, test, folder)
    assert question["id"] == test["question_id"] and test["answer"] not in json.dumps(question["conversations"])
    assert len(question["conversations"]) == 1
    with pytest.raises(ValueError, match="overlap"):
        heldout(train, dict(test, image=train["image"]), folder)
    (tmp_path / "external.png").touch()
    (folder / "escape").symlink_to(tmp_path / "external.png")
    with pytest.raises(ValueError, match="inside"):
        heldout(train, dict(test, image="escape"), folder)


@pytest.mark.parametrize("field,bad", [("loaded", False), ("adapter_path", "some-other-checkpoint"),
                         ("sample_id", "training-id"), ("image", "unrelated.png"), ("question", "wrong"),
                         ("new_tokens", 9), ("new_tokens", 0), ("text", "  ")])
def test_inference_must_match_heldout_and_checkpoint(tmp_path, field, bad):
    from mcitlib.smoke_images import prediction

    test = dict(question_id="q", image="test.png", text="What color?")
    adapter = tmp_path / "Task1_llava_lora"
    inferred = dict(benchmark="mllm_dcl", loaded=True, adapter_path=str(adapter), sample_id="q",
                    image="test.png", question="What color?", new_tokens=2, text="blue")
    assert prediction(inferred, test, adapter)["text"] == "blue"
    inferred[field] = bad
    with pytest.raises(ValueError):
        prediction(inferred, test, adapter)


def test_incomplete_checkpoint_stops_before_reload_or_score(tmp_path, repo_root, monkeypatch):
    from mcitlib.smoke_images import execute

    (tmp_path / "configs").symlink_to(repo_root / "configs")
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(smoke, "repo_root", lambda: tmp_path)
    models = tmp_path / "models"
    monkeypatch.setenv("MCITLIB_MODEL_ROOT", str(models))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)))
    tower = models / "clip-vit-large-patch14-336"
    base = models / "llava-v1.5-7b"
    for directory in (tower, base):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({"mm_vision_tower": str(tower)}))
    (base / "mm_projector.bin").touch()
    run = tmp_path / "logs/first"

    def incomplete(command, root, log):
        assert command["kind"] == "train"
        adapter = run / "Task1_llava_lora"
        adapter.mkdir()
        (adapter / "trainer_state.json").write_text('{"global_step": 1}')

    monkeypatch.setattr(smoke, "_run", incomplete)
    with pytest.raises(RuntimeError, match="adapter files"):
        execute(None, run, "llava", "0", 19021, minimum=0)
    result = json.loads((run / "smoke-result.json").read_text())
    assert result["status"] == "failed" and "score" not in result
    assert not (run / "inference.json").exists() and not (run / "Result.text").exists()
    with pytest.raises(ValueError, match="already exists"):
        execute(None, run, "llava", "0", 19021)


def test_check_update_reads_actual_weight_file(tmp_path):
    from mcitlib.smoke_images import check_update

    torch = pytest.importorskip("torch")
    for key, value in [("lora_A", 1.0), ("lora_B", 0.0), ("lora_B", float("nan"))]:
        torch.save({f"layer.{key}.weight": torch.tensor([value])}, tmp_path / "adapter_model.bin")
        with pytest.raises(ValueError, match="LoRA B update"):
            check_update(tmp_path)
    torch.save({"layer.lora_B.weight": torch.tensor([0.01])}, tmp_path / "adapter_model.bin")
    assert check_update(tmp_path) == 1
