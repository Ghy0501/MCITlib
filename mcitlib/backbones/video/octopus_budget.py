# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Materialize bound Stage2 and GPWC datasets and retain the selected row indices."""
import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

from .protocol import scaled_auxiliary_sample


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.batch < 1 or args.epochs < 1 or args.budget < 1:
        parser.error("require a positive original budget and batch schedule")
    full = json.loads(args.full.read_text())
    if args.sample:
        sample = json.loads(args.sample.read_text())
        by_id = {row["id"]: row for row in full}
        if len(by_id) != len(full) or any(by_id.get(row["id"]) != row for row in sample):
            raise ValueError("Octopus sample is not a byte-equivalent QA subset of the bound full task")
        selected, audit = scaled_auxiliary_sample(sample, len(full), args.budget, seed=args.seed)
    else:
        from ...methods._octopus.core import sample_indices

        indices = sample_indices(len(full), args.budget, seed=args.seed)
        selected = [full[index] for index in indices]
        audit = {"full_train": len(full), "full_budget": args.budget, "indices": indices, "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.update(
        full_sha256=hashlib.sha256(args.full.read_bytes()).hexdigest(),
        sample_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest() if args.sample else None,
        selected_count=len(selected), effective_batch=args.batch, epochs=args.epochs,
        full_updates_per_epoch=len(selected) // args.batch,
    )
    with args.output.open("x") as handle:
        json.dump(selected, handle, ensure_ascii=False)
    with args.output.with_suffix(".budget.json").open("x") as handle:
        json.dump(audit, handle, indent=2)


if __name__ == "__main__":
    main()
