# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2025 Jinpeng Chen
# Copyright (c) 2025-2026 MCITlib authors
"""SEFE merge step using its private newer-snapshot model builder."""
from __future__ import annotations

import argparse

from .llava_overrides.builder import load_pretrained_model


def merge_lora(args) -> None:
    from llava.eval.CoIN.coin_utils import get_model_name_from_path

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="cpu"
    )
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
