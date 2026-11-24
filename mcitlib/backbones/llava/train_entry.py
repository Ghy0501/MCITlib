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
    if os.environ.get('MCITLIB_METHOD') == 'same' and os.environ.get('MCITLIB_BACKBONE', 'llava') != 'llava':
        raise ValueError('SAME prior integration supports only the LLaVA backbone')
    from .overrides import import_llava, install_model_override

    import_llava()
    if os.environ.get("MCITLIB_BACKBONE") == "internvl":
        from ..internvl.overrides import install_vision_factory

        install_vision_factory()
    selfcheck = os.environ.get("MCITLIB_SELFCHECK") == "1"
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    if not selfcheck:
        replace_llama_attn_with_flash_attn()
    install_model_override(os.environ.get("MCITLIB_MODEL_OVERRIDE"))
    entry = os.environ.get("MCITLIB_TRAIN_ENTRY", "standard")
    if entry == "standard":
        from .standard_train import train
    elif entry == "moe":
        # Install type mappings and save/load compatibility before the
        # training module binds any peft helpers in this fresh rank process.
        from ...tuners._peft_registry import install_method_adapter

        install_method_adapter(os.environ.get("MCITLIB_METHOD", "moelora"))
        from .moe_train import train
    else:
        raise ValueError(f"unknown MCITLIB_TRAIN_ENTRY: {entry}")
    if selfcheck:
        if os.environ.get("MCITLIB_METHOD") == "pclr":
            from ...methods._pclr import llava  # noqa: F401
        if os.environ.get("MCITLIB_METHOD") == "dgg":
            from ...methods._dgg import runtime  # noqa: F401
        if os.environ.get("MCITLIB_METHOD") == "same":
            from ...methods._same import prior  # noqa: F401
        print(json.dumps({"selfcheck": "ok", "entry": entry, "method": os.environ.get("MCITLIB_METHOD")}))
        return
    train()


if __name__ == "__main__":
    main()
