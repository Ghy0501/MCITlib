# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Collect full-model Octopus gradients through a native video data/model path."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from types import MethodType

import peft
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))
from mcitlib.methods._octopus.gpwc import _hash_tensor, _target_weight, collect_mean, reduce_history  # noqa: E402
from mcitlib.methods._octopus.state import digest, load_manifest  # noqa: E402


def move_media(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.bfloat16)
    if isinstance(value, list):
        return [move_media(sample, device) for sample in value]
    if isinstance(value, tuple):
        return tuple(move_media(item, device) for item in value)
    return value


def move_batch(batch, device):
    return {key: move_media(value, device) if key == "images" else value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def native_supervision(model, batch):
    """Measure the actual supervised labels after this model's video token insertion."""
    part = move_batch(batch, model.device)
    prepare = model.prepare_inputs_labels_for_multimodal
    inputs = {
        "input_ids": part["input_ids"], "attention_mask": part.get("attention_mask"),
        "position_ids": None, "past_key_values": None, "labels": part["labels"], "images": part["images"],
    }
    bound = {key: value for key, value in inputs.items() if key in inspect.signature(prepare).parameters}
    with torch.no_grad():
        labels = prepare(**bound)[-1]
    if labels is None or labels.ndim != 2:
        raise ValueError("GPWC cannot determine supervised labels after native video expansion")
    return labels[:, 1:].ne(-100).sum(dim=1).tolist()


def build_runtime(backbone, args, device, *, differentiable_tower=True):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, use_fast=backbone == "videollama2", model_max_length=2048, padding_side="right",
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    common = dict(data_path=[args.train], lazy_preprocess=True, is_multimodal=True, image_aspect_ratio="pad")
    if backbone == "videollava":
        from videollava import conversation
        from videollava.model import LlavaLlamaForCausalLM
        from videollava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()
        trainer = importlib.import_module("videollava.train.train")
        base_cls = LlavaLlamaForCausalLM
        model = base_cls.from_pretrained(args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        tower = model.get_video_tower()
        tower.load_model()
        dataset = trainer.DataArguments(video_folder=args.media, **common)
        dataset.video_processor = tower.video_processor
        dataset.num_frames = tower.config.num_frames
        dataset.mm_use_im_start_end = model.config.mm_use_im_start_end
        conversation.default_conversation = conversation.conv_templates["v1"]
        model.config.tokenizer_model_max_length = 3072
    else:
        from videollama2.model import Videollama2MistralForCausalLM

        trainer = importlib.import_module("videollama2.train")
        model = Videollama2MistralForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
        )
        tower = model.get_vision_tower()
        dataset = trainer.DataArguments(data_folder=args.media, num_frames=8, **common)
        dataset.image_processor = tower.image_processor
        dataset.video_processor = tower.image_processor
        dataset.image_size = tower.image_size
        dataset.is_pretraining = False
        model.config.num_frames = 8
        model.config.tokenizer_model_max_length = 2048
    model.config.use_cache = False
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.to(device=device, dtype=torch.bfloat16)
    if differentiable_tower:
        method = getattr(tower.forward, "__wrapped__", None)
        if method is None:
            raise RuntimeError("The native video tower lacks its expected no-grad wrapper")
        tower.forward = MethodType(method, tower)
    data = trainer.make_supervised_data_module(tokenizer=tokenizer, data_args=dataset)
    loader = DataLoader(data["train_dataset"], batch_size=args.logical_batch, shuffle=False,
                        collate_fn=data["data_collator"], num_workers=0)
    return model, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("videollava", "videollama2"))
    for name in ("base", "train", "media"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--history", action="append", required=True)
    parser.add_argument("--logical-batch", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--offload-activations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if not 0 < args.microbatch <= args.logical_batch or args.output.exists():
        parser.error("invalid batch geometry or duplicate GPWC output")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if Path(sys.prefix) not in Path(peft.__file__).resolve().parents:
        raise RuntimeError("GPWC requires the pinned native PEFT environment")
    model, loader = build_runtime(args.backbone, args, device)
    history = [Path(path).resolve(strict=True) for path in args.history]
    histories, audits = [], []
    for index, path in enumerate(history):
        if load_manifest(path) != history[:index]:
            raise ValueError(f"Octopus history does not match the preceding tasks at {path}")
        model = peft.PeftModel.from_pretrained(model, path).merge_and_unload()
        means, batches = collect_mean(
            model, loader, device, args.microbatch,
            count_tokens=native_supervision, move_batch=move_batch, offload_activations=args.offload_activations,
        )
        keys = {name for name in means if _target_weight(name)}
        visual = {name for name in means if ".video_tower." in name or ".vision_tower." in name}
        if len(keys) != 224 or not visual or not any(means[name].count_nonzero().item() for name in sorted(visual)):
            raise RuntimeError("GPWC did not produce differentiable video tower and language gradients")
        if histories and keys != set(histories[0]):
            raise RuntimeError("GPWC language target keys changed across history")
        audits.append({"history": str(path), "batches": batches, "video_gradient_tensors": len(visual),
                       "full_gradient_tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                                        "sha256": _hash_tensor(value)}
                                                 for name, value in means.items()}})
        histories.append({name: means[name] for name in keys})
        del means
        model.zero_grad(set_to_none=True)
    result = reduce_history(histories, keys, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.pt")
    identity = [str(path) for path in history]
    common = {"version": 1, "history": identity, "samples": len(loader.dataset), "logical_batch": args.logical_batch,
              "microbatch": args.microbatch}
    with partial.open("xb") as handle:
        torch.save({**common, "gradients": result}, handle)
    partial.rename(args.output)
    audit = {**common, "gpwc_sha256": digest(args.output), "reduction": "cuda_stack_sum_original_dtype",
             "full_gradient_audits": audits, "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    audit_path = args.audit or args.output.with_suffix(args.output.suffix + ".audit.json")
    with audit_path.open("x") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps({"event": "video_gpwc_complete", "samples": len(loader.dataset), "keys": len(keys),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)


if __name__ == "__main__":
    main()
