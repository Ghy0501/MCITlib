# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU tests for checkpoint loading with partitioned parameter placeholders."""
from __future__ import annotations

import json
import multiprocessing
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from mcitlib.backbones.llava.zero3_loading import load_zero3_state_dict


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.full_like(value, index + 1) for index, (name, value) in enumerate(module.state_dict().items())}


def test_unpartitioned_module_uses_native_loading_and_reports_shape_errors(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "deepspeed", None)
    module = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 4))
    weights = _state(module)
    partial = {"0.weight": weights["0.weight"], "unused": torch.ones(1)}
    result = load_zero3_state_dict(module, partial, strict=False)
    assert torch.equal(module[0].weight, partial["0.weight"])
    assert set(result.missing_keys) == {"0.bias", "2.weight", "2.bias"}
    assert result.unexpected_keys == ["unused"]
    with pytest.raises(RuntimeError, match="size mismatch for 0.weight"):
        load_zero3_state_dict(module, {"0.weight": torch.zeros(5, 3)}, strict=False)


def test_partitioned_checkpoint_requires_initialized_deepspeed(monkeypatch) -> None:
    module = nn.Linear(2, 2)
    weights = _state(module)
    module.weight.ds_id = 1
    fake = ModuleType("deepspeed")
    fake.comm = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "deepspeed", fake)
    with pytest.raises(RuntimeError, match="requires initialized DeepSpeed communication"):
        load_zero3_state_dict(module, weights)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 4))
        self.embedding = nn.Embedding(7, 4)
        self.prompt = nn.Linear(4, 2)
        self.unused = nn.Linear(4, 2)
        self.register_buffer("scale", torch.tensor([0.0]))


def _gloo_rank(rank: int, rendezvous: str, output: str, case: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        model = _TinyModel()
        all_weights = _state(model)
        if case in {"projector", "shape_error"}:
            model = model.projector
            weights = _state(model)
        else:
            weights = {name: all_weights[name] for name in ("projector.0.weight", "embedding.weight",
                                                             "prompt.weight", "scale")}
        records = {"rank": rank, "loaded": 0, "reductions": 0, "gathered": [], "captured": {}}
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if name.startswith("prompt."):
                continue  # Simulate prompt parameters created outside zero.Init.
            parameter.ds_id = index
            parameter.ds_shape = parameter.shape
            parameter.ds_process_group = dist.group.WORLD
            parameter.data = torch.empty(0, dtype=parameter.dtype)

        class FakeGather:
            def __init__(self, parameters, modifier_rank):
                assert modifier_rank == 0
                self.parameters = parameters
                records["gathered"] = [item.ds_id for item in parameters]

            def __enter__(self):
                for parameter in self.parameters:
                    parameter.data = torch.zeros(parameter.ds_shape, dtype=parameter.dtype)

            def __exit__(self, *_):
                for parameter in self.parameters:
                    dist.broadcast(parameter.data, src=0, group=dist.group.WORLD)
                    records["captured"][str(parameter.ds_id)] = parameter.data.tolist()
                    parameter.data = torch.empty(0, dtype=parameter.dtype)

        def reduce(flag, group):
            records["reductions"] += 1
            assert group is dist.group.WORLD
            dist.all_reduce(flag, group=group)

        fake = ModuleType("deepspeed")
        fake.zero = SimpleNamespace(GatheredParameters=FakeGather)
        fake.comm = SimpleNamespace(is_initialized=dist.is_initialized, get_rank=dist.get_rank, all_reduce=reduce)
        sys.modules["deepspeed"] = fake
        original = model.load_state_dict

        def load(state_dict, strict):
            records["loaded"] += 1
            if case == "remote_error" and rank == 1:
                raise RuntimeError("synthetic rank-one load error")
            return original(state_dict, strict=strict)

        model.load_state_dict = load
        if case == "shape_error":
            weights["0.weight"] = torch.zeros(5, 3)
        try:
            load_zero3_state_dict(model, weights, strict=False)
            records["status"] = "ok"
        except RuntimeError as error:
            records["status"] = "error"
            records["error"] = str(error)
        records["partitioned_sizes"] = [parameter.numel() for parameter in model.parameters()
                                         if hasattr(parameter, "ds_id")]
        if case in {"sparse", "remote_error"}:
            records["ordinary_prompt"] = model.prompt.weight.detach().tolist()
            records["buffer"] = model.scale.tolist()
        Path(output, f"{case}-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["projector", "sparse", "shape_error", "remote_error"])
def test_two_cpu_ranks_gather_only_targeted_parameters_and_propagate_errors(tmp_path: Path, case: str) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_gloo_rank, args=(rank, str(tmp_path / "rendezvous"), str(tmp_path), case))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=30)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    ranks = [json.loads((tmp_path / f"{case}-{rank}.json").read_text()) for rank in range(2)]
    assert all(item["reductions"] == 1 and not any(item["partitioned_sizes"]) for item in ranks)
    assert ranks[0]["gathered"] == ranks[1]["gathered"]
    if case == "projector":
        assert [item["loaded"] for item in ranks] == [1, 0]
        assert all(item["status"] == "ok" for item in ranks)
        assert len(ranks[0]["gathered"]) == 4
        assert ranks[0]["captured"] == ranks[1]["captured"]
        assert ranks[0]["captured"]["0"] == [[1.0] * 3] * 4
        assert ranks[0]["captured"]["3"] == [4.0] * 4
    elif case == "sparse":
        assert [item["loaded"] for item in ranks] == [1, 1]
        assert all(item["status"] == "ok" for item in ranks)
        assert len(ranks[0]["gathered"]) == 2
        assert ranks[0]["captured"] == ranks[1]["captured"]
        assert ranks[0]["ordinary_prompt"] == ranks[1]["ordinary_prompt"]
        assert ranks[0]["buffer"] == ranks[1]["buffer"] == [1.0]
    elif case == "shape_error":
        assert [item["loaded"] for item in ranks] == [1, 0]
        assert all(item["status"] == "error" for item in ranks)
        assert "size mismatch for 0.weight" in ranks[0]["error"]
        assert "another ZeRO-3 rank" in ranks[1]["error"]
    else:
        assert [item["loaded"] for item in ranks] == [1, 1]
        assert all(item["status"] == "error" for item in ranks)
        assert "synthetic rank-one" in ranks[1]["error"]
        assert "another ZeRO-3 rank" in ranks[0]["error"]
