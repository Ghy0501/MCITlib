#!/usr/bin/env python3
"""Print keys and coarse stats for a RegLoRA regularization_info .pth file."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pth_path", type=Path, help="Path to *.pth (dict of param_name -> {top2: ...})")
    p.add_argument("--max-print", type=int, default=0, help="Print at most N keys (0 = all)")
    args = p.parse_args()

    path = args.pth_path.expanduser()
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        print("Top-level type:", type(obj))
        print(repr(obj)[:2000])
        return

    keys = list(obj.keys())
    print("num_keys:", len(keys))
    v0 = obj[keys[0]]
    print("sample_value_type:", type(v0))
    if isinstance(v0, dict):
        print("sample_value_keys:", list(v0.keys()))

    kinds = Counter()
    for k in keys:
        if "vision_tower" in k:
            kinds["vision_tower"] += 1
        elif ".model.layers." in k and ".encoder.layers." not in k:
            kinds["llm_model.layers"] += 1
        elif "mm_projector" in k:
            kinds["mm_projector"] += 1
        else:
            kinds["other"] += 1
    print("key_prefix_guess:", dict(kinds))

    limit = args.max_print if args.max_print > 0 else len(keys)
    print("\n--- keys ---")
    for k in keys[:limit]:
        print(k)
    if limit < len(keys):
        print(f"... ({len(keys) - limit} more)")


if __name__ == "__main__":
    main()
