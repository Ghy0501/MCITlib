# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Print an editable CL-VISTA node binding without opening models or datasets."""
import argparse
import json
from pathlib import Path

from .commands import NATIVE
from .workflow import TASKS

TOWERS = {
    "videollava": {"mm_image_tower": "LanguageBind_Image", "mm_video_tower": "LanguageBind_Video_merge"},
    "videollama2": {"mm_vision_tower": "clip-vit-large-patch14-336"},
}


def template(root: Path) -> dict:
    root = root.expanduser().resolve()
    models = root / "models"
    inputs = {"train_path": "train", "test_q_path": "test_questions", "test_a_path": "test_answers",
              "router_path": "router", "replay_path": "replay"}
    return {
        "benchmark": "cl_vista", "runtime_root": str(root), "checkpoints": str(root / "checkpoints"),
        "results": str(root / "results"), "media": str(root / "media"), "native_sources": str(root / "sources"),
        "auxiliary_clip": str(models / "clip-vit-large-patch14-336"),
        "models": {name: {"model_view": str(models / (directory + "-7B")),
                          "conda_prefix": str(root / "envs" / name),
                          "towers": {k: str(models / v) for k, v in TOWERS[name].items()}}
                   for name, directory in NATIVE.items()},
        "tasks": {task: {mode: {key: str(root / "data" / mode / task / (value + ".json"))
                                for key, value in inputs.items()} for mode in ("sample", "full")}
                  for task in TASKS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(template(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
