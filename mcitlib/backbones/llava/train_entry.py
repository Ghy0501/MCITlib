# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Deepspeed entry point preserving v3's two training paths.

Run through the deepspeed launcher as ``deepspeed ... --module mcitlib.backbones.llava.train_entry``.
Setting ``MCITLIB_SELFCHECK=1`` imports everything the real run needs (without applying the
flash-attention patch, building a model, or training) and exits; ``mcitlib doctor`` uses it.
"""
from __future__ import annotations

import json
import os


def main() -> None:
    from .overrides import import_llava, install_model_override

    import_llava()
    selfcheck = os.environ.get("MCITLIB_SELFCHECK") == "1"
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    if not selfcheck:
        replace_llama_attn_with_flash_attn()
    install_model_override(os.environ.get("MCITLIB_MODEL_OVERRIDE"))
    entry = os.environ.get("MCITLIB_TRAIN_ENTRY", "standard")
    if entry == "standard":
        from .standard_train import train
    elif entry == "moe":
        from .moe_train import train
    else:
        raise ValueError(f"unknown MCITLIB_TRAIN_ENTRY: {entry}")
    if selfcheck:
        print(json.dumps({"selfcheck": "ok", "entry": entry, "method": os.environ.get("MCITLIB_METHOD")}))
        return
    train()


if __name__ == "__main__":
    main()
