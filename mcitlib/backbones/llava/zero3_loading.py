# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Load selected checkpoint tensors into a model containing ZeRO-3 parameters."""
from __future__ import annotations

from typing import Any, Mapping


def load_zero3_state_dict(module: Any, state_dict: Mapping[str, Any], *, strict: bool = False) -> Any:
    """Load the matching state without interpreting ZeRO placeholders as real shapes.

    All ranks in the data-parallel group must call this with the same state
    mapping. Only partitioned parameters named in that mapping are gathered,
    avoiding an all-at-once gather of an entire model for a sparse checkpoint.
    The ordinary PyTorch path, including its return value, is unchanged when
    the checkpoint does not target any partitioned parameter.
    """
    named_parameters = list(module.named_parameters(remove_duplicate=False))
    targets = [parameter for name, parameter in named_parameters if name in state_dict]
    partitioned = []
    seen: set[int] = set()
    for parameter in targets:
        if hasattr(parameter, "ds_id") and id(parameter) not in seen:
            partitioned.append(parameter)
            seen.add(id(parameter))
    if not partitioned:
        return module.load_state_dict(state_dict, strict=strict)

    import deepspeed
    import torch

    if not deepspeed.comm.is_initialized():
        raise RuntimeError("loading ZeRO-3 parameters requires initialized DeepSpeed communication on all ranks")
    group = partitioned[0].ds_process_group
    if any(parameter.ds_process_group is not group for parameter in partitioned):
        raise ValueError("checkpoint parameters belong to different ZeRO process groups")
    # DeepSpeed broadcasts gathered parameters, not ordinary parameters or
    # buffers. A mixed checkpoint therefore has to load on every rank.
    replicated = (any(not hasattr(parameter, "ds_id") for parameter in targets) or
                  any(name in state_dict for name, _ in module.named_buffers(remove_duplicate=False)))
    local_error: Exception | None = None
    result = None
    with deepspeed.zero.GatheredParameters(partitioned, modifier_rank=0):
        if replicated or deepspeed.comm.get_rank(group=group) == 0:
            try:
                result = module.load_state_dict(state_dict, strict=strict)
            except Exception as error:
                local_error = error
        # Every rank exits the gather and then raises if any rank could not
        # load, rather than letting the other ranks continue into training.
        failed = torch.tensor([int(local_error is not None)], dtype=torch.int32, device=partitioned[0].device)
        deepspeed.comm.all_reduce(failed, group=group)
    if failed.item():
        if local_error is not None:
            raise local_error
        raise RuntimeError("checkpoint failed to load on another ZeRO-3 rank")
    return result
