# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Reload one smoke adapter and generate from the first training example only.

This module is entered through the regular eval_runners.entry, which verifies vendored
imports and installs the selected method hooks before importing this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...smoke import MAX_NEW_TOKENS, MAX_SAMPLES, SUPPORTED_BENCHMARKS


def generate_one(torch, model, input_ids, image_tensor, instruction_ids, stopping_criteria=None):
    """Perform exactly one generation with a fixed token limit."""
    options = {"do_sample": False, "temperature": 0, "top_p": None, "num_beams": 1,
               "max_new_tokens": MAX_NEW_TOKENS, "use_cache": True}
    if stopping_criteria is not None:
        options["stopping_criteria"] = stopping_criteria
    with torch.inference_mode():
        return model.generate(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), instruction_ids=instruction_ids, **options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from PIL import Image

    from ...backbones.llava.eval_runners.coin.coin_utils import get_model_name_from_path

    records = json.loads(args.subset.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_SAMPLES:
        raise ValueError("smoke subset must contain 1..8 records")
    sample = records[0]
    human = next(item["value"] for item in sample["conversations"] if item.get("from") == "human")
    question = human.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    adapter_path = str(args.adapter.resolve())
    model_name = get_model_name_from_path(adapter_path)
    if "llava" not in model_name.lower() or "lora" not in model_name.lower():
        raise ValueError("adapter directory name must select the shared LLaVA LoRA loader")
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(adapter_path, args.model_base, model_name)
    if model.config.mm_use_im_start_end:
        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        text = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image = Image.open(args.image_folder / sample["image"])
    stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if args.benchmark == "mllm_dcl":
        image_tensor = process_images([image.convert("RGB")], image_processor, model.config)[0]
        stopping_criteria = [KeywordsStoppingCriteria([stop], tokenizer, input_ids)]
    else:
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        stopping_criteria = None
    instruction_ids = tokenizer(question, return_tensors="pt").input_ids.cuda()
    output_ids = generate_one(torch, model, input_ids, image_tensor, instruction_ids, stopping_criteria)
    tokens = output_ids[:, input_ids.shape[1]:]
    if tokens.shape[1] > MAX_NEW_TOKENS:
        raise RuntimeError("model returned more tokens than the smoke limit")
    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    if args.benchmark == "ucit" and text.endswith(stop):
        text = text[:-len(stop)].strip()
    result = {"benchmark": args.benchmark, "loaded": True, "adapter_path": adapter_path, "sample_id": sample.get("id"),
              "image": sample["image"], "question": question, "new_tokens": tokens.shape[1], "text": text}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
