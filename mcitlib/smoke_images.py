# SPDX-License-Identifier: Apache-2.0
"""One updated LoRA checkpoint, one held-out MLLM-DCL RS answer, one native score."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from . import smoke
from .paths import PathResolver

SCORER = "mcitlib.backbones.llava.eval_runners.coin.eval_ai2d"


def selected_gpu(index):
    """Translate a logical index without escaping the caller's visible set."""
    if not re.fullmatch(r"[0-9]+", index):
        raise ValueError("GPU must be a nonnegative logical index")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return index
    devices = visible.split(",")
    # CUDA accepts a GPU UUID prefix as well as ordinals and MIG identifiers.
    uuid = r"[0-9a-fA-F]{1,8}(?:-[0-9a-fA-F]{1,4}){0,3}(?:-[0-9a-fA-F]{1,12})?"
    device = rf"(?:0|[1-9][0-9]*|(?:GPU|MIG)-{uuid}|MIG-GPU-{uuid}/[0-9]+/[0-9]+)"
    if (any(not re.fullmatch(device, d) for d in devices) or len(set(devices)) != len(devices)
            or int(index) >= len(devices)):
        raise ValueError("invalid CUDA_VISIBLE_DEVICES or logical GPU index outside its allocation")
    return devices[int(index)]


def score_plan(paths, directory, backbone, gpu, port, dataset="fixture"):
    gpu = selected_gpu(gpu)
    plan = smoke.build_plan(paths, directory, 1, 1, master_port=port, benchmark="mllm_dcl", backbone=backbone)
    run = Path(plan["run_dir"])
    train, infer = plan["training_command"], plan["inference_command"]
    argv = train["argv"]
    # Keep the shared DeepSpeed trainer/optimizer. Its normal multi-rank launcher
    # rewrites CUDA_VISIBLE_DEVICES; torchrun leaves the physical pin intact.
    train["argv"] = ["python", "-m", "torch.distributed.run", "--nproc_per_node=1",
                     "--master_addr=127.0.0.1", f"--master_port={port}", *argv[argv.index("--module"):]]
    smoke._set_value(train["argv"], "--warmup_ratio", "0")
    train["metadata"] = {"role": "one_step_first_score", "world_size": 1, "global_batch_size": 1}
    for command in train, infer:
        command["env"].update(CUDA_VISIBLE_DEVICES=gpu,
                              HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    smoke._set_value(infer["argv"], "--subset", str(run / "heldout-question.json"))
    resolver = PathResolver.from_file(paths, repo_root=smoke.repo_root())
    plan["test_data"] = resolver.resolve_templates("${data_root}/MLLM-CL/RS/test.json", "mllm_dcl")
    if dataset == "fixture":
        folder = run / "fixture"
        plan.update(image_folder=str(folder), source_data=str(folder / "train.json"),
                    test_data=str(folder / "test.json"), dataset_scope="generated fixture; non-benchmark")
        smoke._set_value(train["argv"], "--image_folder", str(folder))
        smoke._set_value(infer["argv"], "--image-folder", str(folder))
    elif dataset == "rs":
        plan["dataset_scope"] = "user-supplied RS split; first test row only"
    else:
        raise ValueError("dataset must be fixture or rs")
    plan["score_command"] = {"kind": "native-rs-score", "cwd": train["cwd"], "env": {"CUDA_VISIBLE_DEVICES": ""},
                             "argv": ["python", "-m", SCORER, "--annotation-file", str(run / "heldout-label.json"),
                                      "--result-file", str(run / "answers.jsonl"), "--output-dir", str(run)]}
    return plan


def check_space(path, minimum):
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum:
        raise RuntimeError(f"only {free:.2f} GiB free on {path}; need {minimum} GiB")
    return round(free, 2)


def generate_fixture(folder):
    """Original flat-color PNGs and prompts; distributable under Apache-2.0."""
    from PIL import Image, ImageDraw

    folder.mkdir()
    prompt = "What color is the square? Answer with one word."
    for split, color in [("train", "red"), ("test", "blue")]:
        image = Image.new("RGB", (128, 128), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 107, 107), fill=color)
        image.save(folder / f"{split}.png")
    smoke._write_json(folder / "train.json", [{"id": "fixture-train-1", "image": "train.png", "conversations": [
        {"from": "human", "value": "<image>\n" + prompt}, {"from": "gpt", "value": "red"}]}])
    smoke._write_json(folder / "test.json", [{"question_id": "fixture-test-1", "image": "test.png",
                                             "text": prompt, "answer": "blue"}])


def local_model_view(source, target, tower):
    # Public configs can name a Hub ID or the original training host's tower.
    # Localize only that field; reuse weight shards, tokenizer and projector.
    config = json.loads((source / "config.json").read_text())
    target.mkdir()
    smoke._write_json(target / "config.json", {**config, "mm_vision_tower": str(tower)})
    for file in source.iterdir():
        if file.name != "config.json" and file.is_file():
            (target / file.name).symlink_to(file)


def read_score(directory):
    raw = (directory / "Result.text").read_text().strip()
    match = re.fullmatch(r"Samples: 1\nAccuracy: (0\.00|100\.00)%", raw)
    if not match:
        raise ValueError(f"invalid native one-row accuracy: {raw!r}")
    return float(match[1])


def heldout(train, test, folder):
    if (type(test.get("question_id")) not in {str, int} or not str(test["question_id"]).strip()
            or not all(isinstance(test.get(k), str) and test[k].strip() for k in ("image", "text", "answer"))):
        raise ValueError("invalid RS held-out question, image or answer")
    question = {"id": test["question_id"], "image": test["image"],
                "conversations": [{"from": "human", "value": test["text"]}]}
    smoke._validate_records([train], folder)
    smoke._validate_records([question], folder)
    if ((folder / train["image"]).samefile(folder / question["image"])
            or str(test["question_id"]) in {str(train.get(k)) for k in ("id", "question_id")}):
        raise ValueError("train/test image or question overlap")
    if not any(m.get("from") == "gpt" and str(m.get("value") or "").strip() for m in train["conversations"]):
        raise ValueError("training row lacks a supervised answer")
    return question


def prediction(inference, test, adapter):
    if (inference.get("loaded") is not True or inference.get("adapter_path") != str(adapter)
            or inference.get("benchmark") != "mllm_dcl" or inference.get("sample_id") != test["question_id"]
            or inference.get("image") != test["image"] or inference.get("question") != test["text"].strip()
            or type(inference.get("new_tokens")) is not int or not 1 <= inference["new_tokens"] <= smoke.MAX_NEW_TOKENS
            or not isinstance(inference.get("text"), str) or not inference["text"].strip()):
        raise ValueError("reload prediction does not match checkpoint, held-out row or token budget")
    return {"question_id": test["question_id"], "prompt": inference["question"], "text": inference["text"]}


def check_update(adapter):
    import torch

    weights = torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    # Standard LoRA-FT initializes every B to zero. A global_step alone does
    # not distinguish an actual optimizer update from a zero-LR warmup step.
    bs = [v for k, v in weights.items() if ".lora_B." in k]
    if not bs or not all(t.isfinite().all() for t in bs) or not any(t.count_nonzero() for t in bs):
        raise ValueError("no finite trained LoRA B update in saved adapter")
    return len(bs)


def execute(paths, directory, backbone, gpu, port, dataset="fixture", minimum=1):
    root = smoke.repo_root()
    plan = score_plan(paths, directory, backbone, gpu, port, dataset)
    run = Path(plan["run_dir"])
    run.mkdir(mode=0o700)
    try:
        plan["free_gib"] = {"start": check_space(run, minimum)}
        if dataset == "fixture":
            generate_fixture(Path(plan["image_folder"]))
        train = smoke._first_records(Path(plan["source_data"]), 1)[0]
        test = smoke._first_records(Path(plan["test_data"]), 1)[0]
        folder = Path(plan["image_folder"])
        question = heldout(train, test, folder)
        smoke._write_json(run / "train-subset.json", [train])
        smoke._write_json(run / "heldout-question.json", [question])
        smoke._write_json(run / "heldout-label.json", [test])
        for flag in ("--model_name_or_path", "--vision_tower", "--pretrain_mm_mlp_adapter"):
            path = Path(smoke._value(plan["training_command"]["argv"], flag))
            if not (path.is_file() if flag == "--pretrain_mm_mlp_adapter" else (path / "config.json").is_file()):
                raise ValueError(f"missing local {flag}: {path}")
        train_argv = plan["training_command"]["argv"]
        base = Path(smoke._value(train_argv, "--model_name_or_path"))
        tower = Path(smoke._value(train_argv, "--vision_tower"))
        plan["source_model"] = str(base)
        if json.loads((base / "config.json").read_text()).get("mm_vision_tower") != str(tower):
            view = run / "base-model"
            local_model_view(base, view, tower)
            smoke._set_value(train_argv, "--model_name_or_path", str(view))
            smoke._set_value(plan["inference_command"]["argv"], "--model-base", str(view))
        print(f"Training {backbone} on logical GPU {gpu}; logs: {run}", flush=True)
        smoke._run(plan["training_command"], root, run / "train.log")
        adapter = smoke._check_checkpoint(plan)
        plan["lora_b_tensors_checked"] = check_update(adapter)
        plan["free_gib"]["after_train"] = check_space(run, minimum)
        print("Reloading saved adapter for held-out inference", flush=True)
        smoke._run(plan["inference_command"], root, run / "inference.log")
        inference = json.loads((run / "inference.json").read_text())
        answer = prediction(inference, test, adapter)
        (run / "answers.jsonl").write_text(json.dumps(answer) + "\n")
        smoke._run(plan["score_command"], root, run / "score.log")
        plan["score"] = {"metric": "RS accuracy (%)", "value": read_score(run), "samples": 1, "scorer": SCORER}
        plan["free_gib"]["end"] = check_space(run, minimum)
        plan["status"] = "completed"
    except Exception as exc:
        plan.update(status="failed", error=str(exc))
        raise
    finally:
        smoke._write_json(run / "smoke-result.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("llava", "internvl"))
    parser.add_argument("--gpu", default="0", help="logical GPU index (default: 0; respects CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--dataset", choices=("fixture", "rs"), default="fixture")
    parser.add_argument("--min-free-gib", type=int, default=1, help="minimum free disk before each stage")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="new directory under checkout logs/")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.min_free_gib < 1:
        parser.error("--min-free-gib must be positive")
    result = execute(args.paths, args.run_dir, args.backbone, args.gpu, args.port, args.dataset, args.min_free_gib)
    print(json.dumps(result["score"]))


if __name__ == "__main__":
    main()
