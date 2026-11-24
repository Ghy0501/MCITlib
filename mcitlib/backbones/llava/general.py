# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build LLaVA general-purpose benchmark inference and scoring sequences."""
from __future__ import annotations

import json
from pathlib import Path

from ...config.loader import load_layered_config, read_yaml
from ...config.schema import Command
from ...registry import get_method

EVAL_ENTRY = "mcitlib.backbones.llava.eval_runners.entry"
GENERAL_VQA = "mcitlib.backbones.llava.eval_runners.general_model_vqa"
GENERAL_MMBENCH = "mcitlib.backbones.llava.eval_runners.general_model_vqa_mmbench"
GENERAL_SCORER = "mcitlib.eval.general"


def build_general_commands(
    config_root: str | Path,
    method_id: str,
    after_task: int | None = None,
    model_path: str | None = None,
    model_base: str | None = None,
    num_chunks: int | None = None,
) -> list[Command]:
    config_root = Path(config_root)
    repo_root = config_root.resolve().parent
    after_task = after_task or 5
    method = get_method(method_id)
    values = load_layered_config(config_root, "llava", "mllm_dcl", method.id, task_id=after_task)
    eval_config = values.get("eval_config", {})
    if not eval_config or "model_path" not in eval_config:
        raise ValueError(f"no MLLM-DCL eval config for {method.id} task {after_task}")
    model_values = values["model_config"]
    model_config = model_values if isinstance(model_values, dict) else json.loads(
        (repo_root / model_values).read_text(encoding="utf-8")
    )
    selected_model = model_path or eval_config["model_path"]
    selected_base = model_base or model_config["model_name"]
    chunks = num_chunks or int(eval_config.get("gpu_num", 1))
    if chunks < 1:
        raise ValueError("num_chunks must be positive")
    benchmark = read_yaml(config_root / "benchmarks/general.yaml")
    commands: list[Command] = []
    order = 0

    def add(argv: list[str], name: str, chunk: int | None = None) -> None:
        nonlocal order
        env = {"MCITLIB_METHOD": method.id}
        if method.id == "same":
            env["MCITLIB_SAME_CLIP_TOWER"] = model_config["vision_tower"]
        if method.model_override:
            env["MCITLIB_MODEL_OVERRIDE"] = method.model_override
        if chunk is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(chunk)
        commands.append(Command(
            argv=argv, env=env, kind="eval", task_id=after_task, sequence="general",
            metadata={"order": order, "general_benchmark": name},
        ))
        order += 1

    for task in benchmark["tasks"]:
        name = task["id"]
        data = json.loads((config_root / "data_configs/General_benchmark" / task["data_config"]).read_text())
        result_dir = Path(data["result_path"]) / task["output_name"] / eval_config["stage"]
        merged = result_dir / "merge.jsonl"
        target = GENERAL_MMBENCH if name == "mmbench" else GENERAL_VQA
        inputs = []
        for chunk in range(chunks):
            answers = result_dir / f"{chunks}_{chunk}.jsonl"
            inputs.append(str(answers))
            argv = [
                "python", "-m", EVAL_ENTRY, target,
                "--model-path", str(selected_model), "--model-base", str(selected_base),
                "--question-file", data["test_path"], "--answers-file", str(answers),
                "--num-chunks", str(chunks), "--chunk-idx", str(chunk),
                "--temperature", "0", "--conv-mode", "vicuna_v1",
            ]
            if name != "mmbench":
                argv.extend(("--image-folder", data["test_folder"]))
            else:
                argv.append("--single-pred-prompt")
            for key, flag in (
                ("num_task", "--num-task"), ("text_tower", "--text-tower"),
                ("prefix_len", "--prefix-len"), ("cur_task", "--cur-task"), ("num_tasks", "--num-tasks"),
            ):
                if eval_config.get(key) is not None:
                    argv.extend((flag, str(eval_config[key])))
            add(argv, name, chunk)
        add(["python", "-m", GENERAL_SCORER, "merge", "--inputs", *inputs, "--output", str(merged)], name)
        score = [
            "python", "-m", GENERAL_SCORER, "score", "--benchmark", name,
            "--result-file", str(merged), "--output", str(result_dir / "score.json"),
        ]
        if name == "pope":
            score.extend(("--question-file", data["test_path"], "--annotation-dir", data["anno_path"]))
        elif name == "mme":
            score.extend(("--question-file", data["test_path"], "--annotation-dir", data["test_folder"]))
        else:
            score.extend(("--annotation-file", data["anno_path"] if name == "seed" else data["test_path"]))
        add(score, name)
    return commands
