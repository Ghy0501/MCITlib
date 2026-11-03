# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Deepspeed entry point preserving v3's two training paths."""
from __future__ import annotations

import os

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


def main() -> None:
    replace_llama_attn_with_flash_attn()
    entry = os.environ.get("MCITLIB_TRAIN_ENTRY", "standard")
    if entry == "standard":
        from .standard_train import train
    elif entry == "moe":
        from .moe_train import train
    else:
        raise ValueError(f"unknown MCITLIB_TRAIN_ENTRY: {entry}")
    train()


if __name__ == "__main__":
    main()
