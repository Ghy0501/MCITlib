# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2025 Jinpeng Chen
# Copyright (c) 2025-2026 MCITlib authors
"""SEFE key-element identification, migrated from v3 key_elements_identification.py."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import tqdm


def identify_key_elements(step: int, model_dir: str | Path, base_model: str, device: str | None = None) -> Path:
    model_dir = Path(model_dir)
    model_path_list = [
        model_dir / f"Task{index}_{base_model}_lora" for index in range(1, 10)
    ]
    if step > 0:
        legacy_topk_path = Path(f"{model_path_list[step - 1]}_topP.pth")
        legacy_topk = torch.load(legacy_topk_path, map_location="cpu" if device else None)
    model_path_list = model_path_list[step:step + 1]
    for model_path in model_path_list:
        lora_params = torch.load(model_path / "adapter_model.bin", map_location="cpu" if device else None)
        original_names = {
            key.replace(".lora_A", "").replace(".lora_B", "") for key in lora_params
        }
        lora_pairs = {key: [] for key in original_names}
        for key in lora_params:
            original = key.replace(".lora_A", "").replace(".lora_B", "")
            lora_pairs[original].append(key)
            if len(lora_pairs[original]) == 2:
                lora_pairs[original].sort()
        stats = {}
        for _, (name, pair) in tqdm.tqdm(enumerate(sorted(lora_pairs.items()))):
            lora_a = lora_params[pair[0]]
            lora_b = lora_params[pair[1]]
            if device:
                lora_a, lora_b = lora_a.to(device), lora_b.to(device)
            lora_result = (lora_b @ lora_a).to(torch.float32)
            importance = torch.abs(lora_result)
            count = int(0.02 * importance.shape[0] * importance.shape[1])
            _, flat_indices = torch.topk(importance.view(-1), count)
            indices = torch.stack([
                torch.div(flat_indices, importance.size(1), rounding_mode="floor"),
                flat_indices % importance.size(1),
            ]).T
            if device:
                indices = indices.cpu()
            stats[name] = {"top2": indices}
            if step > 0:
                stats[name]["top2"] = torch.cat((stats[name]["top2"], legacy_topk[name]["top2"]), dim=0)
        save_path = model_dir / (os.path.basename(model_path).strip("_prev") + "_topP.pth")
        torch.save(stats, save_path)
        return save_path
    raise ValueError(f"step {step} does not select a model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=int)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("base_model")
    parser.add_argument("--device")
    args = parser.parse_args()
    identify_key_elements(args.step, args.model_dir, args.base_model, args.device)


if __name__ == "__main__":
    main()
