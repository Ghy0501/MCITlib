# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Audit the 24 UCIT/DCL four-GPU plans without starting any training or GPU job.

Plan construction may import method runtime packages. Compare each training
argv against its two-GPU plan, inspect inference pins/merge inputs, and report
any remaining discovery dependencies rather than treating them as success.
"""
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "four_gpu_validation"
CHANGED = {"--include", "--per_device_train_batch_size", "--gradient_accumulation_steps"}


def option(command, flag):
    return command.argv[command.argv.index(flag) + 1]


def effective(command):
    world = len(option(command, "--include").split(":", 1)[1].split(","))
    return world * int(option(command, "--per_device_train_batch_size")) * int(option(command, "--gradient_accumulation_steps"))


def normalized_train(command):
    result = list(command.argv)
    for flag in CHANGED:
        result[result.index(flag) + 1] = "<resource>"
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan(method, benchmark, four, sequence="train"):
    kwargs = {"gpus": "0,1,2,3" if four else "0,1", "sequence": sequence, "validation_fixes": True}
    if four:
        kwargs["execution_profile"] = PROFILE
    commands = build_sequence_commands(ROOT / "configs", "llava", method.id, benchmark.id, **kwargs)
    if method.separate_train_and_infer and sequence == "train":
        for task_id in range(1, len(benchmark.tasks) + 1):
            commands += build_sequence_commands(ROOT / "configs", "llava", method.id, benchmark.id,
                                                task_id=task_id, only_eval=True, **kwargs)
    return commands


def inspect(method, benchmark, sequence="train"):
    case = f"{method.id}/{benchmark.id}/{sequence}"
    old, new = plan(method, benchmark, False, sequence), plan(method, benchmark, True, sequence)
    before = {c.task_id: c for c in old if c.kind == "train"}
    after = {c.task_id: c for c in new if c.kind == "train"}
    require(set(before) == set(after) == set(range(1, len(benchmark.tasks) + 1)), f"{case}: task coverage")
    for task, command in after.items():
        require(effective(before[task]) == effective(command), f"{case}/{task}: effective batch changed")
        require(normalized_train(before[task]) == normalized_train(command), f"{case}/{task}: training argv changed")
        require(option(command, "--include") == "localhost:0,1,2,3", f"{case}/{task}: invalid training GPUs")
        require(command.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3", f"{case}/{task}: invalid training visibility")
    chunks = [c for c in new if c.kind == "eval" and "--answers-file" in c.argv]
    groups = {}
    for command in chunks:
        require(option(command, "--num-chunks") == "4", f"{case}: inference chunk count")
        idx = int(option(command, "--chunk-idx"))
        require(command.env["CUDA_VISIBLE_DEVICES"] == str(idx), f"{case}: inference GPU pin")
        group = groups.setdefault((command.task_id, str(Path(option(command, "--answers-file")).parent)), {})
        require(idx not in group, f"{case}: duplicate inference chunk {idx}")
        group[idx] = option(command, "--answers-file")
    require(all(set(group) == set(range(4)) for group in groups.values()), f"{case}: incomplete inference group")
    merges = [c for c in new if c.metadata.get("merge_chunks")]
    discovery = 0
    for command in merges:
        if "--discover" in command.argv:
            discovery += 1
            require("merge-expected-chunks" in command.argv, f"{case}: unchecked discovery merge")
        else:
            key = (command.task_id, str(Path(option(command, "--output")).parent))
            group = groups[key]
            require(command.argv[command.argv.index("--inputs") + 1:] == [group[i] for i in range(4)],
                    f"{case}: incorrect merge inputs")
    require(len(merges) == len(groups) + discovery, f"{case}: merge coverage")
    old_keep = [c for c in old if c.kind == "extra" and "mcitlib.methods._keeplora.extract_gradients" in c.argv]
    new_keep = [c for c in new if c.kind == "extra" and "mcitlib.methods._keeplora.extract_gradients" in c.argv]
    require([c.argv for c in old_keep] == [c.argv for c in new_keep], f"{case}: gradient preparation argv changed")
    require([c.env for c in old_keep] == [c.env for c in new_keep], f"{case}: gradient preparation environment changed")
    require(all(c.metadata.get("gradient_preparation_shards") == 2 for c in new_keep), f"{case}: gradient shard metadata")
    require(all(c.metadata.get("execution_profile") == PROFILE for c in new), f"{case}: profile metadata")
    return {"method": method.id, "benchmark": benchmark.id, "sequence": sequence, "training_tasks": len(after),
            "inference_groups": len(groups), "merge_groups": len(merges), "discovery_dependencies": discovery,
            "keeplora_gradient_commands_unchanged": len(new_keep)}


def main():
    benchmarks = [b for b in BENCHMARKS if b.id in {"ucit", "mllm_dcl"}]
    with redirect_stdout(sys.stderr):
        cases = [inspect(m, b) for m in METHODS for b in benchmarks]
        router = next(m for m in METHODS if m.id == "mrlora")
        cases.extend(inspect(router, b, "router") for b in benchmarks)
    print(json.dumps({"cases": len(cases), "primary_combinations": 24, "router_sequences": 2,
                      "training_tasks": sum(c["training_tasks"] for c in cases),
                      "inference_groups": sum(c["inference_groups"] for c in cases),
                      "discovery_dependencies": sum(c["discovery_dependencies"] for c in cases),
                      "details": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
