# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""GPU and port assignment (U1/U2).

v3 hard-coded ``localhost:0,1`` and port 9001 for training and, for evaluation, pinned chunk
``i`` of ``N`` to GPU ``i`` (``CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}``) so the chunks of one
dataset ran in parallel, one per card.  The recorded commands lost both properties:

* training always used GPUs 0,1 and one fixed port (two sequences on one node collided);
* evaluation chunks were not pinned, so each chunk spread over every visible GPU
  (``device_map="auto"``) and, run one after the other, was slower and could run out of memory.

Only GPU *ids* and the port are replaced here.  The *number* of GPUs is part of the experiment
(it fixes the effective batch size) and must match the configuration.
"""
from __future__ import annotations

from collections import defaultdict

from .config.schema import Command

DEFAULT_PORT = "9001"


def parse_gpus(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    ids = [item.strip() for item in spec.split(",") if item.strip()]
    if not ids or any(not item.isdigit() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"--gpus must be distinct comma-separated GPU ids, got {spec!r}")
    return ids


def apply_gpu_assignment(commands: list[Command], gpus: str | None, master_port: int | None) -> None:
    """Mutate ``commands``: swap GPU ids/port, pin every evaluation chunk to one GPU."""
    ids = parse_gpus(gpus)
    for command in commands:
        argv = command.argv
        if command.kind == "train" and "--include" in argv:
            index = argv.index("--include") + 1
            host, listed = argv[index].split(":", 1)
            needed = len(listed.split(","))
            if ids is not None:
                if len(ids) != needed:
                    raise ValueError(
                        f"this configuration trains on {needed} GPUs (it fixes the effective batch size); "
                        f"--gpus lists {len(ids)}: {','.join(ids)}"
                    )
                argv[index] = f"{host}:{','.join(ids)}"
            if master_port is not None and "--master_port" in argv:
                argv[argv.index("--master_port") + 1] = str(master_port)
    _pin_evaluation_chunks(commands, ids)


def _pin_evaluation_chunks(commands: list[Command], ids: list[str] | None) -> None:
    chunk_count: dict[tuple, int] = defaultdict(int)
    for command in commands:
        if "--chunk-idx" in command.argv and "--num-chunks" in command.argv:
            chunk_count[(command.task_id, command.sequence)] = max(
                chunk_count[(command.task_id, command.sequence)],
                int(command.argv[command.argv.index("--num-chunks") + 1]),
            )
    for command in commands:
        argv = command.argv
        if "--chunk-idx" not in argv or "--num-chunks" not in argv:
            continue
        index = int(argv[argv.index("--chunk-idx") + 1])
        total = int(argv[argv.index("--num-chunks") + 1])
        if ids is not None and len(ids) < total:
            raise ValueError(f"evaluation uses {total} chunks (one GPU each) but --gpus lists {len(ids)}")
        recorded = command.env.get("CUDA_VISIBLE_DEVICES")
        if ids is not None:
            command.env["CUDA_VISIBLE_DEVICES"] = ids[index]
        elif recorded is None:
            command.env["CUDA_VISIBLE_DEVICES"] = str(index)  # v3: GPULIST[IDX] with the default 0..N-1
        command.metadata["parallel_group"] = (
            f"{command.sequence}:{command.task_id}:{command.argv[command.argv.index('--answers-file') + 1].rsplit('/', 1)[0]}"
            if "--answers-file" in argv else f"{command.sequence}:{command.task_id}"
        )
        command.metadata["parallel_total"] = total
