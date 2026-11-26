# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build effective native CL-VISTA commands from a node binding."""
import json
import os
from pathlib import Path

from ...config.schema import Command

NATIVE = {"videollava": "Video-LLaVA", "videollama2": "VideoLLaMA2"}


def bind_native_sources(commands, node, backbone, *, required=False):
    """Use the prepared runtime only, independently of the image PEFT checkout."""
    default = Path(node.get("runtime_root", ".")) / "sources"
    base = Path(os.environ.get("MCITLIB_VIDEO_SOURCES") or node.get("native_sources", default)).expanduser()
    native = base.resolve() / NATIVE[backbone]
    if required and not (native / backbone / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing {native / backbone}; prepare it with tools/prepare_video_sources.py")
    for command in commands:
        if command.metadata.get("native_video"):
            command.env.update(PYTHONPATH=str(native), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return commands


def video_command(source, backbone, stage, flags, *, task=None, train_port=None, observe=False, method="lora_ft", method_options=None):
    native = source / "third_party" / NATIVE[backbone]
    worker = source / "mcitlib/backbones/video/worker.py"
    if train_port is None:
        launcher = ["python", "-u"]
    else:
        launcher = ["deepspeed"]
        launcher.extend(("--master_port", str(train_port)))
    argv = [*launcher, str(worker), "--backbone", backbone, "--method", method]
    if method_options:
        argv.extend(("--method-options", json.dumps(method_options, separators=(",", ":"))))
    argv.append(stage)
    if observe:
        argv.append("--observe")
    for name, value in flags.items():
        argv.extend((f"--{name}", str(value)))
    return Command(
        argv, cwd=str(source), kind=stage, task_id=task,
        env={"PYTHONPATH": str(native), "PYTHONNOUSERSITE": "1"},
        metadata={"native_video": True},
    )


def lora_train_command(source, node, backbone, method, task, dataset, output, budgets, port, *, previous=None, steps=None, accumulation=1, runtime_options=None):
    paths = node["models"][backbone]
    rank = budgets["rank"]
    flags = {
        "lora_enable": True, "lora_r": rank, "lora_alpha": budgets.get("lora_alpha", rank * 2),
        "mm_projector_lr": 1e-5, "learning_rate": budgets["lr"],
        "num_train_epochs": budgets["epoch"], "per_device_train_batch_size": budgets["batch_size"],
        "gradient_accumulation_steps": budgets["grad_acc"], "image_aspect_ratio": "pad",
        "mm_vision_select_layer": -2, "bf16": True, "tf32": True, "data_path": dataset,
        "output_dir": output, "save_strategy": "steps", "save_steps": 50000, "weight_decay": 0,
        "warmup_ratio": 0.03, "lr_scheduler_type": "cosine", "logging_steps": 1,
        "model_max_length": 2048, "gradient_checkpointing": True,
        "dataloader_num_workers": 4, "report_to": "tensorboard",
    }
    if backbone == "videollava":
        flags.update(
            model_name_or_path=paths["model_view"], video_tower=paths["towers"]["mm_video_tower"],
            video_folder=node["media"], mm_projector_type="mlp2x_gelu",
            version="v1", mm_use_im_start_end=False, mm_use_im_patch_token=False,
            per_device_eval_batch_size=16, evaluation_strategy="no", group_by_modality_length=True,
            tokenizer_model_max_length=3072, lazy_preprocess=True,
            deepspeed=source / "configs/deepspeed/video_llava_zero2_offload.json",
        )
    else:
        flags.update(
            model_path=paths["model_view"], model_type="videollama2_mistral",
            vision_tower=paths["towers"]["mm_vision_tower"], data_folder=node["media"],
            mm_projector_type="stc_connector", num_frames=8, per_device_eval_batch_size=4,
            fp16=False, deepspeed=source / "configs/deepspeed/zero2.json",
        )
    if method in {"pclr", "modalprompt"}:
        flags["lora_enable"] = False
    if method == "modalprompt":
        flags["deepspeed"] = source / "configs/deepspeed/zero3_offload.json"
    if method == "dgg" and runtime_options:
        flags["save_steps"] = runtime_options.get("save_steps", 50000)
        if runtime_options.get("keep_existing_checkpoint"):
            flags["save_strategy"] = "no"
    if previous is not None:
        flags["previous_task_model_path"] = previous
    if steps is not None:
        flags.update(
            max_steps=steps, dataloader_num_workers=0, report_to="none",
            per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        )
        if steps == 1 and method == "lora_ft":
            # One warmup step would leave the only optimizer update at LR zero.
            flags.update(warmup_ratio=0, save_strategy="no")
    method_options = runtime_options
    if method == "modalprompt" and method_options is None:
        method_options = dict(task=task, tasks=budgets["num_tasks"], prefix_length=budgets["prefix_len"],
                              text_tower=node["auxiliary_clip"])
    if method in {"moelora", "clmoe", "olora", "smolora", "sigma", "disco", "hide", "same"}:
        # These pure plan inputs do not import the heavy tuner implementation.
        task_slot = int(budgets.get("cur_task", task - 1))
        method_options = {"expert_num": int(budgets["expert_num"]), "task_id": task, "cur_task": task_slot}
        if method in {"disco", "hide"}:
            method_options["text_tower"] = node["auxiliary_clip"]
        if method == "same":
            method_options["clip_path"] = node["auxiliary_clip"]
        if method == "smolora":
            asset = source / "preparations/other_backbones/assets" / NATIVE[backbone] / "smolora/ins_emb_multi.pkl"
            method_options.update(
                ins_type=int(budgets["ins_type"]), ins_emb=str(asset),
                ins_emb_sha256="c8a769183da3b2b41e5b08d09e928fd539e14717d490d2a61ed819ed17ff82ef",
            )
    command = video_command(
        source, backbone, "train", flags, task=task, train_port=port, observe=steps is not None,
        method=method, method_options=method_options,
    )
    command.metadata["method"] = method
    return command
