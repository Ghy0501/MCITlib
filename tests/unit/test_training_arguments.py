# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Parse every v3 training argv with the selected shared MCITlib entry."""
from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest
import transformers

from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands


@pytest.mark.parametrize("method", METHODS, ids=lambda item: item.id)
@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_all_training_argv_parse(
    repo_root: Path, method, benchmark, monkeypatch
) -> None:
    deepspeed_stub = ModuleType("deepspeed")
    deepspeed_stub.__spec__ = ModuleSpec("deepspeed", loader=None)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_stub)
    if method.train_entry == "moe":
        from mcitlib.backbones.llava import moe_train as module
    else:
        from mcitlib.backbones.llava import standard_train as module
    parser = transformers.HfArgumentParser((module.ModelArguments, module.DataArguments, module.TrainingArguments))
    commands = build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark.id)
    for command in commands:
        if command.kind != "train":
            continue
        argv = list(command.argv)
        argv = argv[argv.index("-m") + 2:]
        deepspeed_index = argv.index("--deepspeed")
        del argv[deepspeed_index:deepspeed_index + 2]
        # Deepspeed path existence/contents and launcher integration are covered
        # separately; omit it here so parsing does not require its installation.
        # Parser-only CPU test: hardware capabilities are not under test.
        for flag in ("--bf16", "--tf32"):
            if flag in argv:
                argv[argv.index(flag) + 1] = "False"
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(argv)
        assert training_args.output_dir
        assert data_args.data_path
        assert model_args.model_name_or_path
