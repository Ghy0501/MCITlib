# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Dispatch CL-VISTA through the same command runner as the bounded worker."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from ...paths import repo_root
from ...registry import VIDEO_BACKBONES, compatibility, get_backbone, get_method
from ...runner import execute_commands
from .commands import bind_native_sources, video_command
from .workflow import build_cl_vista_plan, load_binding


def selected_gpus(requested, default):
    lease = os.environ.get("MCITLIB_VIDEO_GPUS")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = next((value for value in (requested, lease, visible) if value is not None), default)
    ids = selected.split(",")
    if (len(ids) not in {1, 2, 4} or len(set(ids)) != len(ids)
            or not all(re.fullmatch(r"0|[1-9][0-9]*", i) for i in ids)):
        raise ValueError("CL-VISTA requires 1, 2 or 4 distinct numeric physical GPU IDs")
    for limit in (lease, visible):
        if limit is not None and not set(ids).issubset(limit.split(",")):
            raise ValueError("Selected GPUs are outside the current lease or CUDA_VISIBLE_DEVICES")
    return selected


def run(args):
    backbone = get_backbone(args.backbone).id
    method = get_method(args.method).id
    if backbone not in VIDEO_BACKBONES:
        raise ValueError("CL-VISTA execution requires one of the registered video backbones")
    supported, reason = compatibility(method, backbone)
    if not supported:
        raise ValueError(f"{backbone}/{method}: {reason}")
    if args.video_node is None:
        raise ValueError("CL-VISTA requires --video-node with the node-local model and dataset binding")
    for option in ("paths", "execution_profile", "validation_fixes", "expected_eval_samples",
                   "expected_eval_hashes", "dgg_sample_profile", "dgg_sample_manifest", "dgg_resume_checkpoint",
                   "model_base", "skip_gpu", "skip_entrypoints"):
        if getattr(args, option, None):
            raise ValueError(f"--{option.replace('_', '-')} is not a CL-VISTA option")
    node_path = args.video_node.resolve(strict=True)
    node = load_binding(node_path)
    source = repo_root()
    runtime = Path(node["runtime_root"]).resolve(strict=not args.dry_run)
    port = args.master_port or 29640
    mode = args.video_mode
    dry = args.dry_run
    gpu_ids = selected_gpus(args.gpus, "0,1" if mode == "full" or method == "mrlora" else "0,1,2,3")
    gpu_set = gpu_ids.split(",")
    model = node["models"][backbone]
    if not dry:
        if Path(sys.prefix).resolve() != Path(model["conda_prefix"]).resolve():
            raise ValueError(f"Select the pinned conda environment {model['conda_prefix']}")
        bind_native_sources([], node, backbone, required=True)
    if args.command == "doctor":
        dataset = node["tasks"]["counting"][mode]
        rows = json.loads(Path(dataset["train_path"]).read_text())
        if not rows:
            raise ValueError("The counting train split has no example for the video doctor")
        video = Path(node["media"]) / rows[0]["video"]
        commands = [video_command(source, backbone, "doctor", {"base": model["model_view"], "video": video},
                                  train_port=port)]
        bind_native_sources(commands, node, backbone)
    else:
        if args.sequence != "train" or getattr(args, "model_path", None) or getattr(args, "num_chunks", None):
            raise ValueError("CL-VISTA binds its checkpoint pool and assessment cells through the node plan")
        config_root = args.config_root or source / "configs"
        task = args.task if args.command == "train" else args.after_task
        commands = build_cl_vista_plan(
            config_root, node, backbone, method, mode, args.video_run_id, task=task,
            only_eval=args.command == "eval", master_port=port, gpu_count=len(gpu_set),
            judge_directory=runtime / "control",
        )
    for command in commands:
        if command.argv[0] == "deepspeed":
            command.argv[1:1] = ["--include", "localhost:" + gpu_ids]
        else:
            command.env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if dry:
        print(json.dumps([item.as_dict() for item in commands], indent=2))
        return 0
    logs = runtime / "logs/runner"
    run_dir = args.run_dir if getattr(args, "run_dir", None) else logs / f"cli-{backbone}-{method}-{args.command}-{args.video_run_id}"
    result = execute_commands(commands, run_dir, runtime, logs_root=logs, resume=getattr(args, "resume", False))
    print(json.dumps(result, indent=2))
    return 0
