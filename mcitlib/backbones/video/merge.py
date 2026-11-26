# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Save a native video LoRA merge as the complete next-task base."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("videollava", "videollama2"))
    for name in ("base", "adapter", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (args.adapter / "non_lora_trainables.bin").is_file():
        raise FileNotFoundError(f"Non-LoRA video state is missing at {args.adapter}")
    if args.backbone == "videollava":
        from videollava.model.builder import load_pretrained_model
        name = "videollava-lora"
    else:
        from videollama2.model import load_pretrained_model
        name = "videollama2-lora"
    tokenizer, model, _, _ = load_pretrained_model(
        str(args.adapter), str(args.base), name, device_map={"": "cuda:0"},
    )
    if hasattr(model, "peft_config"):
        raise RuntimeError("Merge did not return the native base model")
    model.eval()
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output)
    print(json.dumps({"event": "merged_video_model", "output": str(args.output),
                      "dtype": str(model.dtype), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)


if __name__ == "__main__":
    main()
