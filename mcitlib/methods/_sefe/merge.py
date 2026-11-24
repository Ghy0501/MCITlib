# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2025 Jinpeng Chen
# Copyright (c) 2025-2026 MCITlib authors
"""SEFE merge step using its private newer-snapshot model builder."""
from __future__ import annotations

import argparse


def merge_lora(args) -> None:
    from mcitlib.backbones.llava.overrides import allow_t437_llava_registration
    with allow_t437_llava_registration():
        from mcitlib.methods._sefe.llava_overrides.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="cpu"
    )
    generation_config = getattr(model, "generation_config", None)
    if getattr(generation_config, "do_sample", None) is False:
        # Sampling parameters are ignored by greedy generation but must be neutral for saving.
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--save-model-path", required=True)
    merge_lora(parser.parse_args())


if __name__ == "__main__":
    main()
