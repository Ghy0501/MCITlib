# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build KeepLoRA input subspaces from native video-model weights or gradients."""
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .expert_hooks import LLM_TARGETS


def directions(value, mode, *, threshold, rank, previous=None):
    activation = value.float().T
    if previous is not None:
        previous = previous.to(activation)
        residual = activation - previous.T @ (previous @ activation)
    else:
        residual = activation
    if mode == "fixed":
        vectors, _, _ = torch.svd_lowrank(residual, q=rank)
        return vectors.T
    vectors, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    energy = singular.square().cumsum(dim=0)
    total = activation.square().sum()
    if total <= 0:
        raise ValueError("KeepLoRA cannot update a principal subspace from a zero matrix")
    if mode == "weights":
        retained = max(1, int((energy < threshold * total).sum()))
    else:
        represented = total - energy[-1] if previous is not None else 0
        retained = min(len(energy), int((energy + represented < threshold * total).sum()) + 1)
        if previous is not None and represented >= threshold * total:
            retained = 0
    basis = vectors[:, :retained].T
    if previous is not None:
        basis = torch.cat((previous, basis))[:activation.shape[0]]
    return basis


def video_gradients(model, loader, targets, device, ratio):
    from .octopus_gpwc import move_batch

    original = loader.dataset
    indices = torch.linspace(0, len(original) - 1, max(1, int(len(original) * ratio))).long().tolist()
    data = DataLoader(Subset(original, indices), batch_size=1, collate_fn=loader.collate_fn)
    means = {}
    model.requires_grad_(False).train()
    for parameter in targets.values():
        parameter.requires_grad_(True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for count, batch in enumerate(data, 1):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            loss = model(**move_batch(batch, device)).loss
            if not torch.isfinite(loss) or loss.item() <= 0:
                raise RuntimeError("KeepLoRA received an invalid native video loss")
            loss.backward()
        for name, parameter in targets.items():
            if parameter.grad is None:
                raise RuntimeError(f"No native video gradient for {name}")
            current = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            if count == 1:
                means[name] = current
            else:
                means[name].mul_((count - 1) / count).add_(current, alpha=1 / count)
        model.zero_grad(set_to_none=True)
    return means, indices


def main():
    from .octopus_gpwc import build_runtime

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("videollava", "videollama2"))
    for field in ("base", "train", "media"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--mode", required=True, choices=("weights", "fixed", "energy"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=.99)
    parser.add_argument("--ratio", type=float, default=.1)
    parser.set_defaults(logical_batch=1)
    args = parser.parse_args()
    if not (0 < args.threshold <= 1 and 0 < args.ratio <= 1 and args.rank > 0):
        parser.error("invalid SVD or extraction budget")
    if (args.mode != "weights") != bool(args.previous):
        parser.error("gradient SVD requires a previous principal subspace")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    model, loader = build_runtime(args.backbone, args, device, differentiable_tower=False)
    targets = {key: parameter for key, parameter in model.named_parameters()
               if key.endswith(".weight") and re.fullmatch(LLM_TARGETS, key.removesuffix(".weight"))}
    if len(targets) != 224:
        raise ValueError(f"Expected 224 native language weights, got {len(targets)}")
    previous = torch.load(args.previous, map_location="cpu")["gradients"] if args.previous else None
    if previous is not None and previous.keys() != targets.keys():
        raise ValueError("KeepLoRA previous subspace has different language targets")
    values, indices = (targets, []) if args.mode == "weights" else video_gradients(model, loader, targets, device, args.ratio)
    result = {}
    for index, (name, tensor) in enumerate(values.items(), 1):
        with torch.no_grad():
            result[name] = directions(
                tensor.detach().to(device), args.mode, threshold=args.threshold, rank=args.rank,
                previous=previous[name] if previous is not None else None,
            ).to(device="cpu", dtype=torch.bfloat16).contiguous()
        if index % 16 == 0:
            print(json.dumps({"event": "video_keeplora_svd_progress", "phase": args.mode,
                              "completed": index, "total": len(values)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save({"gradients": result, "source": args.mode, "rank_info": {k: len(v) for k, v in result.items()},
                    "data_indices": indices, "energy_threshold": args.threshold}, handle)
    print(json.dumps({"event": "video_keeplora_subspace", "mode": args.mode, "samples": len(indices),
                      "layers": len(result), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
