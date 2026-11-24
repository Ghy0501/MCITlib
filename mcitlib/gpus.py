# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""GPU and port assignment (U1/U2).

v3 hard-coded ``localhost:0,1`` and port 9001 for training and, for evaluation, pinned chunk
``i`` of ``N`` to GPU ``i`` (``CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}``) so the chunks of one
dataset ran in parallel, one per card.  The recorded commands lost both properties:

* training always used GPUs 0,1 and one fixed port (two sequences on one node collided);
* evaluation chunks were not pinned, so each chunk spread over every visible GPU
  (``device_map="auto"``) and, run one after the other, was slower and could run out of memory.

By default only GPU *ids* and the port are replaced, and the GPU count must match the
recorded configuration. The explicit four-GPU validation profile adjusts per-device batch
or accumulation to preserve the effective batch and expands inference chunk groups.
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
        serial_dgg = command.env.get("MCITLIB_METHOD") == "dgg" and ids is not None and len(ids) == 1
        if ids is not None and len(ids) < total and not serial_dgg:
            raise ValueError(f"evaluation uses {total} chunks (one GPU each) but --gpus lists {len(ids)}")
        recorded = command.env.get("CUDA_VISIBLE_DEVICES")
        if ids is not None:
            command.env["CUDA_VISIBLE_DEVICES"] = ids[0] if serial_dgg else ids[index]
        elif recorded is None:
            command.env["CUDA_VISIBLE_DEVICES"] = str(index)  # v3: GPULIST[IDX] with the default 0..N-1
        if not serial_dgg:
            command.metadata["parallel_group"] = (
                f"{command.sequence}:{command.task_id}:{command.argv[command.argv.index('--answers-file') + 1].rsplit('/', 1)[0]}"
                if "--answers-file" in argv else f"{command.sequence}:{command.task_id}"
            )
            command.metadata["parallel_total"] = total


FOUR_GPU_VALIDATION = "four_gpu_validation"


def execution_profile(config_root, name: str | None, gpus: str | None) -> dict | None:
    """Load the explicit validation profile; the unprofiled path remains untouched."""
    if name is None:
        return None
    if name != FOUR_GPU_VALIDATION:
        raise ValueError(f"unknown execution profile: {name}")
    from pathlib import Path

    from .config.loader import read_yaml

    profile = read_yaml(Path(config_root) / "execution" / f"{name}.yaml")
    expected_ids = ["0", "1", "2", "3"]
    if (profile.get("id") != name or profile.get("gpu_ids") != expected_ids or
            profile.get("train_world_size") != 4 or profile.get("evaluation_chunks") != 4 or
            profile.get("batch_policy") != "preserve_effective_batch"):
        raise ValueError(f"invalid execution profile {name}: expected four GPUs 0,1,2,3 and preserved effective batch")
    selected = parse_gpus(gpus)
    if selected is not None and selected != expected_ids:
        raise ValueError(f"execution profile {name} requires --gpus 0,1,2,3")
    return profile


def _flag_value(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"execution profile requires exactly one {flag}: {argv}")
    return argv[argv.index(flag) + 1]


def _set_flag(argv: list[str], flag: str, value: str) -> None:
    _flag_value(argv, flag)
    argv[argv.index(flag) + 1] = value


def _resize_training(command: Command, profile: dict) -> None:
    argv = command.argv
    include = _flag_value(argv, "--include")
    try:
        host, original = include.split(":", 1)
        old_world = len(original.split(","))
        batch = int(_flag_value(argv, "--per_device_train_batch_size"))
        accumulation = int(_flag_value(argv, "--gradient_accumulation_steps"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot determine training world size and effective batch: {include}") from error
    world = profile["train_world_size"]
    if min(old_world, batch, accumulation) < 1:
        raise ValueError("training world size, batch size and gradient accumulation must be positive")
    effective = old_world * batch * accumulation
    if effective % (world * batch) == 0:
        new_batch, new_accumulation = batch, effective // (world * batch)
    elif effective % (world * accumulation) == 0:
        new_batch, new_accumulation = effective // (world * accumulation), accumulation
    else:
        raise ValueError(f"cannot preserve effective batch {effective} with {world} GPUs: "
                         f"per-device batch {batch}, gradient accumulation {accumulation}")
    if new_batch < 1 or new_accumulation < 1:
        raise ValueError(f"cannot preserve effective batch {effective} with {world} GPUs")
    _set_flag(argv, "--include", f"{host}:{','.join(profile['gpu_ids'])}")
    _set_flag(argv, "--per_device_train_batch_size", str(new_batch))
    _set_flag(argv, "--gradient_accumulation_steps", str(new_accumulation))
    command.env["CUDA_VISIBLE_DEVICES"] = ",".join(profile["gpu_ids"])
    command.metadata["execution_profile"] = profile["id"]
    command.metadata["effective_batch"] = effective


def _chunk_signature(command: Command) -> tuple:
    argv = list(command.argv)
    for flag in ("--chunk-idx", "--num-chunks", "--answers-file"):
        if flag in argv:
            argv[argv.index(flag) + 1] = "<chunk>"
    env = tuple(sorted((key, value) for key, value in command.env.items() if key != "CUDA_VISIBLE_DEVICES"))
    return command.kind, command.task_id, command.sequence, tuple(argv), env


def _chunk_file(path: str, old_total: int, old_index: int, new_total: int, new_index: int) -> str:
    from pathlib import Path

    old = Path(path)
    if old.name != f"{old_total}_{old_index}.jsonl":
        raise ValueError(f"cannot expand non-standard chunk output: {path}")
    return str(old.with_name(f"{new_total}_{new_index}.jsonl"))


def apply_execution_profile(commands: list[Command], profile: dict) -> list[Command]:
    """Expand ordered 2-way records to 4-way plans before GPU pinning and merging."""
    from copy import deepcopy

    chunks = profile["evaluation_chunks"]
    result: list[Command] = []
    commands = sorted(commands, key=lambda c: (c.task_id or 0, c.metadata.get("order", -1)))
    position = 0
    while position < len(commands):
        command = commands[position]
        argv = command.argv
        if command.kind == "train":
            _resize_training(command, profile)
        if command.kind == "extra" and "mcitlib.methods._keeplora.extract_gradients" in argv:
            # Keep the existing two logical gradient shards and their equal-weight
            # merge: changing the partition alters KeepLoRA initialization.
            command.metadata["execution_profile"] = profile["id"]
            command.metadata["gradient_preparation_shards"] = 2
            result.append(command)
            position += 1
            continue
        if "--num-chunks" in argv and "--chunk-idx" in argv:
            old_total = int(_flag_value(argv, "--num-chunks"))
            group = commands[position:position + old_total]
            if (old_total != 2 or len(group) != old_total or
                    any(_chunk_signature(item) != _chunk_signature(command) for item in group) or
                    {int(_flag_value(item.argv, "--chunk-idx")) for item in group} != set(range(old_total))):
                raise ValueError(f"cannot safely expand incomplete or inconsistent chunk group for task {command.task_id}")
            if "--answers-file" not in argv:
                raise ValueError(f"unknown chunked command in four-GPU profile: {argv}")
            indexed = {int(_flag_value(item.argv, "--chunk-idx")): item for item in group}
            first_order = command.metadata.get("order", -1)
            last_order = group[-1].metadata.get("order", first_order + 1)
            if last_order <= first_order:
                raise ValueError("chunk records must have increasing order")
            for index in range(chunks):
                source_index = index % old_total
                new = deepcopy(indexed[source_index])
                old_answer = _flag_value(new.argv, "--answers-file") if "--answers-file" in new.argv else None
                _set_flag(new.argv, "--num-chunks", str(chunks))
                _set_flag(new.argv, "--chunk-idx", str(index))
                if old_answer is not None:
                    _set_flag(new.argv, "--answers-file", _chunk_file(old_answer, old_total, source_index, chunks, index))
                new.metadata["order"] = first_order + (last_order - first_order) * index / chunks
                new.metadata["execution_profile"] = profile["id"]
                result.append(new)
            position += old_total
            continue
        result.append(command)
        position += 1
    return result


def require_four_discovered_chunks(commands: list[Command]) -> None:
    """A dependency-only merge must not accept stale two-way outputs."""
    for command in commands:
        if command.metadata.get("merge_chunks") and "--discover" in command.argv:
            command.argv[2:4] = ["mcitlib.gpus", "merge-expected-chunks"]
            command.metadata["execution_profile"] = FOUR_GPU_VALIDATION


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Helpers for explicit multi-GPU validation plans")
    sub = parser.add_subparsers(dest="operation", required=True)
    discover = sub.add_parser("merge-expected-chunks", help="merge an existing complete four-chunk set")
    discover.add_argument("--discover", required=True)
    discover.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "merge-expected-chunks":
            from .eval.chunks import discover_chunks, merge_chunks

            inputs = discover_chunks(args.discover)
            if len(inputs) != 4:
                raise ValueError(f"expected four chunks in {args.discover}, found {len(inputs)}")
            merge_chunks(args.output, inputs)
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"mcitlib.gpus: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
