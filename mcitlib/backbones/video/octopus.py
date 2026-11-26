# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Plan the CL-VISTA Octopus stages and the mode-specific auxiliary datasets."""
from ...config.schema import Command
from ...registry import CL_VISTA
from .commands import NATIVE, lora_train_command


def octopus_steps(config, node, backbone, mode, task, data, checkpoint, port, gpus):
    source = config.parent
    ckpt_root = checkpoint.parent
    stage1 = ckpt_root / "stage1" / checkpoint.name
    previous_stage1 = stage1.with_name(f"Task{task - 1}_{backbone}_lora") if task > 1 else None
    main_budget = dict(rank=48, lora_alpha=96, lr=1e-4, batch_size=1 if mode == "sample" else 4,
                       grad_acc=64 // (gpus * (1 if mode == "sample" else 4)), epoch=1)
    first = lora_train_command(
        source, node, backbone, "octopus", task, data, stage1, main_budget, port,
        previous=previous_stage1, runtime_options={"stage": "stage1"},
    )
    first.argv[first.argv.index("--warmup_ratio") + 1] = "0.05"
    first.sequence = "octopus_stage1"
    yield first
    if task == 1:
        yield Command(
            ["python", "-m", "mcitlib.methods._octopus.workflow", "copy-first", "--input", str(stage1),
             "--output_dir", str(checkpoint)], cwd=str(source), kind="octopus_first_copy", task_id=task,
        )
        return

    auxiliary = ckpt_root / "auxiliary" / f"task{task}"
    train_paths = node["tasks"][CL_VISTA.tasks[task - 1]]
    stage2_rows = auxiliary / "stage2.json"
    gpwc_rows = auxiliary / "gpwc.json"
    for output, count, batch in ((stage2_rows, 9600, 64), (gpwc_rows, 256, 16)):
        argv = ["python", "-m", "mcitlib.backbones.video.octopus_budget", "--full",
                train_paths["full"]["train_path"], "--budget", str(count), "--batch", str(batch),
                "--output", str(output)]
        if mode == "sample":
            argv.extend(("--sample", train_paths["sample"]["train_path"]))
        yield Command(argv, cwd=str(source), kind="octopus_budget", task_id=task)
    gpwc = auxiliary / f"task{task}_gpwc.pt"
    history = [str(ckpt_root / f"Task{older}_{backbone}_lora") for older in range(1, task)]
    argv = [
        "python", str(source / "mcitlib/backbones/video/octopus_gpwc.py"), "--backbone", backbone,
        "--base", node["models"][backbone]["model_view"], "--train", str(gpwc_rows), "--media", node["media"],
        "--output", str(gpwc), "--offload-activations",
    ]
    for path in history:
        argv.extend(("--history", path))
    yield Command(
        argv, cwd=str(source), kind="octopus_gpwc", task_id=task,
        env={"PYTHONPATH": str(source / "third_party" / NATIVE[backbone]), "PYTHONNOUSERSITE": "1"},
        metadata={"native_video": True},
    )
    stage2 = lora_train_command(
        source, node, backbone, "octopus", task, stage2_rows, checkpoint, main_budget, port,
        previous=stage1, runtime_options=dict(stage="stage2", history=history, gpwc_path=str(gpwc),
                                              lambda_1=.02, lambda_2=.01),
    )
    stage2.argv[stage2.argv.index("--warmup_ratio") + 1] = "0.05"
    stage2.sequence = "octopus_stage2"
    yield stage2
