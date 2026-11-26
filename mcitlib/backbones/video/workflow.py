# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Plan native CL-VISTA train / held-out generation / score cells."""
from __future__ import annotations

import json
from pathlib import Path

from ...config.schema import Command
from ...registry import CL_VISTA
from .commands import NATIVE, bind_native_sources, lora_train_command, video_command

TASKS = CL_VISTA.tasks
METHOD_DIRECTORY = {
    "lora_ft": "LoRA-FT", "replay": "Replay",
    "moelora": "MoELoRA", "clmoe": "CL-MoE", "olora": "OLoRA", "smolora": "SMoLoRA", "sigma": "SiGMA",
    "disco": "DISCO", "hide": "HiDe", "same": "SAME",
    "sefe": "RegLoRA", "keeplora": "KeepLoRA", "modalprompt": "ModalPrompt",
}


def load_binding(node: str | Path) -> dict:
    data = json.loads(Path(node).read_text())
    if data.get("benchmark") != "cl_vista" or set(data["tasks"]) != set(TASKS):
        raise ValueError("node profile does not bind the eight CL-VISTA tasks")
    return data


def score_command(source, predictions, judge_directory, *, task=None):
    return Command(
        ["python", str(source / "mcitlib/backbones/video/score_client.py"),
         str(judge_directory), str(predictions)],
        cwd=str(source), kind="score", task_id=task,
    )


def pclr_steps(config, node, backbone, mode, current, train_data, checkpoint, previous, port, gpus):
    import yaml

    settings = yaml.safe_load((config / "methods/pclr.yaml").read_text())["pclr"]
    rank = settings["new_rank_by_task"].get(current, settings["new_rank"])
    skip = settings["skip_interval_by_task"].get(current, settings["skip_interval"])
    pool = dict(new_rank=rank, top_rank=settings["top_rank"], alpha=settings["alpha"],
                share_rank=settings["shared_rank"], skip_interval=skip)
    if mode == "full" and gpus != 2:
        raise ValueError("PCLR's release profile uses two GPUs")
    budgets = dict(rank=rank, lr=2e-4, batch_size=1 if mode == "sample" else 16,
                   grad_acc=64 // gpus if mode == "sample" else 2, epoch=1)
    compress = current >= settings["first_compression_after_task"]
    learning = checkpoint / "learning" if compress else checkpoint
    source = config.parent
    yield lora_train_command(
        source, node, backbone, "pclr", current, train_data, learning, budgets, port, previous=previous,
        runtime_options=dict(stage="learn", task_id=current, pool=pool),
    )
    if compress:
        budgets.update(lr=settings["integration_learning_rate"], epoch=settings["integration_epochs"])
        integration = lora_train_command(
            source, node, backbone, "pclr", current, train_data, checkpoint, budgets, port, previous=learning,
            runtime_options=dict(stage="integrate", task_id=current, pool=pool),
        )
        integration.sequence = "pclr_integration"
        yield integration


def replace_native_base(command, backbone, base):
    flag = "--model_name_or_path" if backbone == "videollava" else "--model_path"
    command.argv[command.argv.index(flag) + 1] = str(base)


def merge_command(source, backbone, base, adapter, output, *, task):
    return Command(
        ["python", str(source / "mcitlib/backbones/video/merge.py"), "--backbone", backbone,
         "--base", str(base), "--adapter", str(adapter), "--output", str(output)],
        cwd=str(source), kind="merge", task_id=task,
        env={"PYTHONPATH": str(source / "third_party" / NATIVE[backbone]), "PYTHONNOUSERSITE": "1"},
        metadata={"native_video": True},
    )


def sefe_steps(config, node, backbone, current, train_data, checkpoint_root, budgets, port):
    source = config.parent
    base = node["models"][backbone]["model_view"]
    if current > 1:
        base = checkpoint_root / f"Task{current - 1}-{backbone}-merged"
    adapter = checkpoint_root / f"Task{current}_{backbone}_lora"
    output = checkpoint_root / f"Task{current}-{backbone}-merged"
    options = {"previous_keys": str(checkpoint_root / f"Task{current - 1}_{backbone}_lora_topP.pth")} if current > 1 else None
    train = lora_train_command(source, node, backbone, "sefe", current, train_data, adapter, budgets, port,
                               runtime_options=options)
    replace_native_base(train, backbone, base)
    yield train
    yield Command(
        ["python", "-m", "mcitlib.methods._sefe.key_elements", str(current - 1), str(checkpoint_root),
         backbone, "--device", "cuda:0"],
        cwd=str(source), kind="sefe_keys", task_id=current,
    )
    yield merge_command(source, backbone, base, adapter, output, task=current)


def dgg_steps(config, node, backbone, mode, task, data, checkpoint, previous, port, gpus):
    source = config.parent
    options = {"guidance_probability": .1}
    if previous is not None:
        replay = checkpoint / "replay.json"
        paths = [data, *[node["tasks"][older][mode]["train_path"] for older in TASKS[:task - 1]]]
        yield video_command(source, backbone, "dgg_replay", {
            "media": node["media"], "output": replay, "sources": json.dumps(paths),
        }, task=task, method="dgg")
        options.update(reference_adapter=str(previous), replay_manifest=str(replay.with_suffix(".manifest.json")))
        data = replay
    options.update(data_path=str(data), save_steps=500)
    budget = dict(rank=48, lr=2e-4, batch_size=1, grad_acc=64 // gpus, epoch=1)
    yield lora_train_command(source, node, backbone, "dgg", task, data, checkpoint, budget, port, runtime_options=options)


def keeplora_steps(config, node, backbone, current, data, root, budgets, port):
    source = config.parent
    base = node["models"][backbone]["model_view"]
    if current > 1:
        base = root / f"Task{current - 1}-{backbone}-merged"
    adapter = root / f"Task{current}_{backbone}_lora"
    output = root / f"Task{current}-{backbone}-merged"
    pools = root / "svd"
    previous = pools / f"task{current - 1}-principal.pt"
    initial = pools / f"task{current}-initial.pt"
    principal = pools / f"task{current}-principal.pt"
    env = {"PYTHONPATH": str(source / "third_party" / NATIVE[backbone]),
           "MCITLIB_NO_VENDORED_PATH": "1", "PYTHONNOUSERSITE": "1"}

    def extract(source_model, phase, target, threshold, prior=None):
        argv = [
            "python", "-m", "mcitlib.backbones.video.keep_subspace", "--backbone", backbone,
            "--base", str(source_model), "--train", str(data), "--media", node["media"],
            "--mode", phase, "--output", str(target), "--threshold", str(threshold),
            "--rank", str(budgets["rank"]),
        ]
        if prior is not None:
            argv.extend(("--previous", str(prior)))
        return Command(argv, cwd=str(source), kind="keeplora_" + phase, task_id=current,
                       env=env, metadata={"native_video": True})

    if current == 1:
        yield extract(base, "weights", previous, .6)
    yield extract(base, "fixed", initial, .99, previous)
    train = lora_train_command(
        source, node, backbone, "keeplora", current, data, adapter, budgets, port,
        runtime_options={"gradient_subspace": str(initial)},
    )
    replace_native_base(train, backbone, base)
    yield train
    yield merge_command(source, backbone, base, adapter, output, task=current)
    yield extract(output, "energy", principal, .99, previous)


def build_cl_vista_plan(
    config_path: str | Path,
    node: dict,
    backbone: str,
    method: str,
    mode: str,
    run_id: str,
    *,
    task: int | None = None,
    only_eval=False,
    master_port=29640,
    gpu_count=4,
    judge_directory: str | Path,
) -> list[Command]:
    if method not in METHOD_DIRECTORY and method not in {"mrlora", "pclr", "dgg", "octopus"}:
        raise ValueError(f"{method} does not yet have a CL-VISTA runtime")
    if mode not in {"sample", "full"}:
        raise ValueError("select the sample or full CL-VISTA annotations explicitly")
    if task is not None and not 1 <= task <= len(TASKS):
        raise ValueError("CL-VISTA task index must be 1–8")
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run ID must be one path component")
    if gpu_count not in {1, 2, 4}:
        raise ValueError("video worker expects 1, 2 or 4 leased training GPUs")
    config = Path(config_path).resolve()
    if method == "mrlora":
        if task is not None or only_eval:
            raise ValueError("MR-LoRA's full plan requires the separately bound expert and router pools")
        from .mrlora import build_mrlora_plan
        plan = build_mrlora_plan(config, node, backbone, mode, run_id, master_port, gpu_count, judge_directory)
        return bind_native_sources(plan, node, backbone)
    source = config.parent
    path = config / "train_configs" / METHOD_DIRECTORY.get(method, method.upper()) / NATIVE[backbone] / "CL-VISTA/train"
    if method == "keeplora" and backbone == "videollama2":
        path = config / "train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train"
    namespace = Path(mode) / backbone / method / run_id
    ckpt_root = Path(node["checkpoints"]) / namespace
    result_root = Path(node["results"]) / namespace
    base = node["models"][backbone]["model_view"]
    schedule = range(1, len(TASKS) + 1) if task is None else [task]
    plan = []
    for current in schedule:
        train_domain = TASKS[current - 1]
        current_ckpt = ckpt_root / f"task{current}-{backbone}-lora"
        if method in {"sefe", "keeplora"}:
            current_ckpt = ckpt_root / f"Task{current}-{backbone}-merged"
        elif method == "octopus":
            current_ckpt = ckpt_root / f"Task{current}_{backbone}_lora"
        if not only_eval:
            train_data = node["tasks"][train_domain][mode]["replay_path" if method == "replay" else "train_path"]
            prior = ckpt_root / f"task{current - 1}-{backbone}-lora" if current > 1 else None
            if method == "pclr":
                plan.extend(pclr_steps(config, node, backbone, mode, current, train_data, current_ckpt, prior,
                                       master_port, gpu_count))
            elif method == "dgg":
                plan.extend(dgg_steps(config, node, backbone, mode, current, train_data, current_ckpt, prior,
                                      master_port, gpu_count))
            elif method == "octopus":
                from .octopus import octopus_steps

                plan.extend(octopus_steps(config, node, backbone, mode, current, train_data, current_ckpt,
                                          master_port, gpu_count))
            else:
                reference = json.loads((path / f"task{current}.json").read_text())
                if mode == "sample":
                    reference.update(lr=2e-4, batch_size=1, grad_acc=64 // gpu_count, epoch=1)
                elif reference["gpu_num"] != gpu_count:
                    raise ValueError(f"{train_domain} full schedule expects {reference['gpu_num']} GPUs")
                if method == "sefe":
                    plan.extend(sefe_steps(config, node, backbone, current, train_data, ckpt_root, reference, master_port))
                elif method == "keeplora":
                    plan.extend(keeplora_steps(config, node, backbone, current, train_data, ckpt_root, reference, master_port))
                else:
                    plan.append(lora_train_command(
                        source, node, backbone, method, current, train_data, current_ckpt, reference, master_port,
                        previous=prior,
                    ))
        for held_out in TASKS[:current]:
            eval_data = node["tasks"][held_out][mode]
            predicted = result_root / f"task{current}" / held_out / "predictions.jsonl"
            generation = video_command(source, backbone, "generate", {
                "base": base, "checkpoint": current_ckpt, "media": node["media"],
                "questions": eval_data["test_q_path"], "answers": eval_data["test_a_path"], "predictions": predicted,
            }, task=current, method=method)
            generation.metadata["eval_domain"] = held_out
            if method in {"sefe", "keeplora"}:
                generation.argv.append("--merged")
            plan.append(generation)
            score = score_command(source, predicted, judge_directory, task=current)
            score.metadata["eval_domain"] = held_out
            plan.append(score)
        if method == "octopus" and current > 1 and not only_eval:
            gpwc = ckpt_root / "auxiliary" / f"task{current}" / f"task{current}_gpwc.pt"
            plan.append(Command(
                ["python", "-m", "mcitlib.methods._octopus.workflow", "cleanup", "--input", str(gpwc),
                 "--checkpoint", str(current_ckpt), "--receipt", str(gpwc.with_suffix(".receipt.json"))],
                cwd=str(source), kind="octopus_cleanup", task_id=current,
            ))
    return bind_native_sources(plan, node, backbone)
