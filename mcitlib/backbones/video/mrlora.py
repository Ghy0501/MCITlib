# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Historical CVU expert/router training with domain-scoped question-ID selection."""
from __future__ import annotations

import json
from pathlib import Path

from ...config.schema import Command
from .commands import NATIVE, lora_train_command, video_command
from .workflow import TASKS, score_command


def build_mrlora_plan(config, node, backbone, mode, run_id, master_port, gpu_count, judge_directory):
    if gpu_count != 2:
        raise ValueError("CL-VISTA MR-LoRA's native router global batch requires the two-GPU profile")
    assert mode in {"full", "sample"}
    source = config.parent
    reference = config / "train_configs/MR-LoRA" / NATIVE[backbone] / "CL-VISTA"
    suffix = Path(mode) / backbone / "mrlora" / run_id
    checkpoints = Path(node["checkpoints"]) / suffix
    results = Path(node["results"]) / suffix
    base = node["models"][backbone]["model_view"]
    expert_dir = {domain: checkpoints / f"expert-{i}-{backbone}-lora" for i, domain in enumerate(TASKS, 1)}
    router_dir = {i: checkpoints / f"router-{i}-{backbone}-lora" for i in range(1, len(TASKS) + 1)}
    expert_eval = {}
    commands = []

    def generate(checkpoint, domain, output, task, route=False):
        data = node["tasks"][domain][mode]
        command = video_command(source, backbone, "generate", {
            "base": base, "checkpoint": checkpoint, "media": node["media"],
            "questions": data["test_q_path"], "answers": data["test_a_path"], "predictions": output,
        }, task=task)
        if route:
            command.argv.append("--router")
        command.metadata.update(eval_domain=domain, router=route, expert_pool_scope="retrospective-eight")
        commands.append(command)

    for task, expert in enumerate(TASKS, 1):
        data = node["tasks"][expert][mode]
        expert_budget = json.loads((reference / "train" / f"task{task}.json").read_text())
        router_budget = json.loads((reference / "train_router" / f"task{task}.json").read_text())
        if mode == "sample":
            expert_budget.update(lr=2e-4, batch_size=1, grad_acc=32, epoch=1)
        commands.append(lora_train_command(
            source, node, backbone, "mrlora", task, data["train_path"], expert_dir[expert], expert_budget, master_port,
        ))
        router = lora_train_command(
            source, node, backbone, "mrlora", task, data["router_path"],
            router_dir[task], router_budget, master_port, previous=router_dir.get(task - 1),
        )
        router.sequence = "router"
        commands.append(router)

    for task, expert in enumerate(TASKS, 1):
        for domain in TASKS:
            predicted = results / "expert" / expert / domain / "predictions.jsonl"
            generate(expert_dir[expert], domain, predicted, task)
            expert_eval[expert, domain] = predicted
            commands.append(score_command(source, predicted, judge_directory, task=task))

    for task in range(1, len(TASKS) + 1):
        for domain in TASKS:
            predicted = results / "router" / f"task{task}" / domain / "predictions.jsonl"
            generate(router_dir[task], domain, predicted, task, route=True)
            routed = results / "routed" / f"task{task}" / domain / "predictions.jsonl"
            experts = {name: str(expert_eval[name, domain]) for name in TASKS}
            commands.append(Command([
                "python", "-m", "mcitlib.backbones.video.router_join",
                "--questions", node["tasks"][domain][mode]["test_q_path"],
                "--router-predictions", str(predicted), "--experts", json.dumps(experts), "--output", str(routed),
            ], cwd=str(source), kind="route", task_id=task, sequence="router"))
            commands.append(score_command(source, routed, judge_directory, task=task))
    return commands
