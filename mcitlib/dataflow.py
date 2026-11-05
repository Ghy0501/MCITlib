# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Static producer/consumer check over a command sequence.

A sequence only works end to end if every intermediate file a step *reads* (a previous
task's checkpoint, the merged chunk file a scorer reads, a router result, ...) is *written*
by an earlier step.  Inputs that the user supplies (models, datasets, replay/router data)
live under ``model_root`` / ``data_root`` and are checked by ``mcitlib doctor`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config.schema import Command

# flag -> role.  Values are paths; only the flags below carry intermediate artefacts.
PRODUCES = {"--output_dir", "--save-model-path", "--answers-file", "--output", "--output-dir", "--output_xlsx", "--output_file"}
CONSUMES = {"--previous_task_model_path", "--model-path", "--model-base", "--result-file", "--regularization_info_path"}


@dataclass(frozen=True)
class Gap:
    step: int
    task_id: int | None
    kind: str
    flag: str
    path: str

    def __str__(self) -> str:
        return f"step {self.step} (task {self.task_id}, {self.kind}) reads {self.flag} {self.path}"


def _norm(path: str) -> str:
    return str(Path(path))


def _under(path: str, produced: set[str]) -> bool:
    return path in produced or any(path.startswith(item + "/") for item in produced)


def _positional_outputs(argv: list[str]) -> list[str]:
    """Artefacts written by steps that take their destination positionally.

    ``key_elements.py ID KEY_DIR BASE`` (SEFE) writes ``KEY_DIR/Task{ID+1}_{BASE}_lora_topP.pth``
    (see ``identify_key_elements`` in ``methods/_sefe/key_elements.py``).
    """
    for position, token in enumerate(argv):
        if token.endswith("key_elements.py") and len(argv) > position + 3:
            step, directory, base = argv[position + 1], argv[position + 2], argv[position + 3]
            return [_norm(f"{directory}/Task{int(step) + 1}_{base}_lora_topP.pth")]
    return []


def find_gaps(
    commands: list[Command], user_roots: tuple[str, ...], already_produced: tuple[str, ...] = (),
) -> list[Gap]:
    """Reads of artefacts (outside ``user_roots``) that no earlier step writes.

    ``already_produced`` lists outputs of sequences that must run first (the MR-LoRA router
    sequence consumes the expert checkpoints of the ordinary sequence).
    """
    roots = tuple(_norm(root) for root in user_roots)
    produced: set[str] = {_norm(item) for item in already_produced}
    gaps: list[Gap] = []
    for index, command in enumerate(commands):
        argv = command.argv
        for position, token in enumerate(argv[:-1]):
            value = argv[position + 1]
            if token in CONSUMES and not value.startswith("-"):
                path = _norm(value)
                if any(path == root or path.startswith(root + "/") for root in roots):
                    continue
                if not _under(path, produced):
                    gaps.append(Gap(index, command.task_id, command.kind, token, path))
        if "--inputs" in argv:  # merge step: its inputs are chunk files written by inference steps
            for item in argv[argv.index("--inputs") + 1:]:
                if not _under(_norm(item), produced):
                    gaps.append(Gap(index, command.task_id, command.kind, "--inputs", _norm(item)))
        if "--discover" in argv:  # merge of chunks that no step of this sequence writes
            directory = _norm(argv[argv.index("--discover") + 1])
            if not any(item.startswith(directory + "/") for item in produced):
                gaps.append(Gap(index, command.task_id, command.kind, "--discover", directory))
        for position, token in enumerate(argv[:-1]):
            if token in PRODUCES and not argv[position + 1].startswith("-"):
                produced.add(_norm(argv[position + 1]))
        produced.update(_positional_outputs(argv))
    return gaps


def produced_paths(commands: list[Command]) -> set[str]:
    """Everything a sequence writes (for sequences that must run before another one)."""
    produced: set[str] = set()
    for command in commands:
        argv = command.argv
        for position, token in enumerate(argv[:-1]):
            if token in PRODUCES and not argv[position + 1].startswith("-"):
                produced.add(_norm(argv[position + 1]))
        produced.update(_positional_outputs(argv))
    return produced
