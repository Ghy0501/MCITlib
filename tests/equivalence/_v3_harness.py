# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Isolated v3/v4 data-pipeline harness; never downloads model weights."""
from __future__ import annotations

import argparse
import builtins
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


class ToyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    model_max_length = 2048

    def __call__(self, text, **kwargs):
        ids = [ord(char) % 251 + 2 for char in text]
        if kwargs.get("return_tensors") == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)


class ToyProcessor:
    image_mean = [0.5, 0.5, 0.5]
    crop_size = {"height": 4, "width": 4}

    def preprocess(self, image, return_tensors="pt"):
        image = image.convert("RGB").resize((4, 4))
        values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(4, 4, 3)
        values = values.permute(2, 0, 1) / 255.0
        return {"pixel_values": values.unsqueeze(0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--entry", choices=["standard", "moe"], default="standard")
    parser.add_argument("--method", default="")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    random.seed(1234)
    torch.manual_seed(1234)
    if args.implementation == "v3":
        # O-LoRA's v3 train_MOE.py uses sys.path.append without importing sys.
        builtins.sys = sys
        module_name = "llava.train.train" if args.entry == "standard" else "llava.train.train_MOE"
    else:
        module_name = "mcitlib.backbones.llava.standard_train" if args.entry == "standard" else "mcitlib.backbones.llava.moe_train"
    if args.method in {"RegLoRA", "KeepLoRA"}:
        from mcitlib.backbones.llava.overrides import allow_t437_llava_registration
        with allow_t437_llava_registration():
            module = __import__(module_name, fromlist=["make_supervised_data_module"])
    else:
        module = __import__(module_name, fromlist=["make_supervised_data_module"])
    data_args = module.DataArguments(
        data_path=str(args.data),
        image_folder=str(args.image_folder),
        is_multimodal=True,
        image_aspect_ratio="pad",
    )
    data_args.mm_use_im_start_end = False
    data_args.mm_use_im_patch_token = False
    data_args.image_processor = ToyProcessor()
    tokenizer = ToyTokenizer()
    extra = {"prefix_len": 2, "cur_task": 1, "transfer_num": 3} if args.method == "ModalPrompt" else {}
    module = module.make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, **extra)
    dataset = module["train_dataset"]
    collator = module["data_collator"]
    examples = [dataset[0], dataset[1] if len(dataset) > 1 else dataset[0]]
    batch = collator(examples)
    output = {key: value for key, value in batch.items() if key in {"input_ids", "labels", "attention_mask", "images"}}
    if args.method == "ModalPrompt":
        output["modality_lengths"] = torch.tensor(dataset.modality_lengths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)


if __name__ == "__main__":
    main()
