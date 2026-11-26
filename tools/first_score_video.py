# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Run one observed native video LoRA update and a separate offline fixture score."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcitlib.backbones.video.binding import TOWERS  # noqa: E402
from mcitlib.backbones.video.commands import NATIVE, lora_train_command, video_command  # noqa: E402


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def towers(backbone, models):
    # Keep the public symlink names: native builders can dispatch by path name.
    return {key: str(models / name) for key, name in TOWERS[backbone].items()}


def model_view(backbone, models, view):
    base = models / (NATIVE[backbone] + "-7B")
    config = json.loads((base / "config.json").read_text())
    local = towers(backbone, models)
    for path in local.values():
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    view.mkdir(mode=0o700)
    dump(view / "config.json", {**config, **local})
    for f in base.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.name != "config.json":
            (view / f.name).symlink_to(f)


def fixture(run):
    import cv2
    import numpy as np

    question = "What color is the background? Answer with one color word."
    for name, bgr in [("train-red", (0, 0, 220)), ("eval-blue", (220, 0, 0))]:
        writer = cv2.VideoWriter(str(run / (name + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 8, (128, 128))
        if not writer.isOpened():
            raise RuntimeError("OpenCV needs its FFmpeg MPEG-4 video writer")
        try:
            for i in range(16):
                frame = np.full((128, 128, 3), bgr, dtype=np.uint8)
                frame[55:73, 6*i:6*i+18] = 255
                writer.write(frame)
        finally:
            writer.release()
    dump(run / "train.json", [{"id": "train-red", "video": "train-red.mp4", "conversations": [
        {"from": "human", "value": "<video>\n" + question}, {"from": "gpt", "value": "Red"}]}])
    dump(run / "questions.json", [{"video_name": "eval-blue", "question_id": "eval-blue", "question": question}])
    dump(run / "answers.json", [{"question_id": "eval-blue", "answer": "blue"}])


def commands(backbone, models, run):
    base, ckpt = run / (NATIVE[backbone] + "-7B"), run / (backbone + "-lora")
    node = {"media": run, "models": {backbone: {"model_view": base, "towers": towers(backbone, models)}}}
    budgets = dict(rank=4, lr=1e-4, epoch=1, batch_size=1, grad_acc=1)
    train = lora_train_command(ROOT, node, backbone, "lora_ft", 1, run / "train.json",
                               ckpt, budgets, port=None, steps=1).argv
    generate = video_command(ROOT, backbone, "generate", dict(base=base, checkpoint=ckpt, media=run,
                             questions=run / "questions.json", answers=run / "answers.json",
                             predictions=run / "predictions.jsonl")).argv
    for argv in (train, generate):
        argv[3:3] = ["--worker-seed", "42"]
    train[:2] = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
    generate[0] = sys.executable
    return train, generate


def score(path, answers):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1 or len(answers) != 1 or rows[0]["question_id"] != answers[0]["question_id"]:
        raise ValueError("Expected one prediction for the fixture answer ID")
    prediction = rows[0]["pred"]
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("No generated text to score")
    correct = int(prediction.strip().casefold() == answers[0]["answer"].strip().casefold())
    return {"metric": "fixture_exact_match", "correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def worker_env(run, native):
    env = dict(os.environ, HOME=str(run), PYTHONPATH=str(native),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    for key in ("HF_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TMPDIR"):
        path = run / key.lower()
        path.mkdir()
        env[key] = str(path)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=NATIVE)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="new directory under private scratch")
    parser.add_argument("--plan", action="store_true", help="print worker commands without touching disk/GPU")
    args = parser.parse_args()
    run, models = args.run_dir.resolve(), args.models.resolve()
    train, infer = commands(args.backbone, models, run)
    native = args.sources.resolve() / NATIVE[args.backbone]
    plan = {"train": train, "generate": infer, "source": str(native), "metric": "fixture_exact_match"}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not (native / args.backbone / "__init__.py").is_file():
        parser.error(f"Prepare the pinned native sources first: {native}")
    if any(run.is_relative_to(p) for p in (ROOT, models, args.sources.resolve())):
        parser.error("Place the run directory outside the checkout, snapshots and prepared sources")
    if shutil.disk_usage(run.parent).free < 4 * 2**30:
        parser.error("Need at least 4 GiB free in the private scratch filesystem")
    run.mkdir(mode=0o700)
    dump(run / "plan.json", plan)
    model_view(args.backbone, models, run / (NATIVE[args.backbone] + "-7B"))
    fixture(run)
    env = worker_env(run, native)
    for stage, argv in (("train", train), ("generate", infer)):
        print(f"{stage}: {run / (stage + '.log')}", flush=True)
        with (run / (stage + ".log")).open("x") as log:
            subprocess.run(argv, env=env, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
        if stage == "train":
            ckpt = run / (args.backbone + "-lora")
            for name in ("config.json", "adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json"):
                if not (ckpt / name).stat().st_size:
                    raise ValueError(f"Empty checkpoint file: {ckpt / name}")
    result = score(run / "predictions.jsonl", json.loads((run / "answers.json").read_text()))
    dump(run / "fixture-score.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
