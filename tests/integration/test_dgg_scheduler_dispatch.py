# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Exercise scheduler registration in the shared LLaVA training entry."""
from types import SimpleNamespace

import pytest
import torch

from mcitlib.backbones.llava import standard_train
from mcitlib.methods._dgg import runtime
from mcitlib.methods._dgg.scheduler_checkpoint import DGGSchedulerCheckpoint


@pytest.mark.parametrize("history", ["none", "prior-task"])
def test_dgg_entry_registers_scheduler_and_closes_guidance_on_failure(tmp_path, monkeypatch, history):
    trace = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

    class Guidance:
        def __init__(self, trainer, value):
            self.value = value
            trace.append("guidance-created")

        def close(self):
            trace.append("guidance-closed")

    class StopTraining(Exception):
        pass

    class Trainer:
        def __init__(self, **kwargs):
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint == "checkpoint-to-resume"
            assert isinstance(self.callbacks[0], DGGSchedulerCheckpoint)
            assert self.callbacks[0].trainer is self
            assert self.callbacks[0].resume_from_checkpoint == resume_from_checkpoint
            assert [type(item) for item in self.callbacks[1:]] == ([Guidance] if history != "none" else [])
            trace.append("train")
            raise StopTraining

    model_args = standard_train.ModelArguments(model_name_or_path="tiny", version="v0.5")
    data_args = SimpleNamespace(data_path="data.json")
    training = SimpleNamespace(local_rank=0, fp16=False, bf16=False, bits=16, gradient_checkpointing=False,
                               lora_enable=False, pt_enable=False, cache_dir=str(tmp_path), model_max_length=32,
                               output_dir=str(tmp_path), dgg_history_model_path=history, dgg_probability=.1,
                               dgg_resume_from_checkpoint="checkpoint-to-resume", dgg_expected_data_sha256=None,
                               dgg_replay_manifest=None)
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (model_args, data_args, training))
    monkeypatch.setattr(standard_train.transformers, "HfArgumentParser", lambda *_: parser)
    monkeypatch.setattr(standard_train.transformers.LlamaForCausalLM, "from_pretrained", lambda *a, **k: Model())
    monkeypatch.setattr(standard_train.transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(unk_token="<unk>"))
    monkeypatch.setattr(standard_train, "LLaVATrainer", Trainer)
    monkeypatch.setattr(standard_train, "make_supervised_data_module", lambda **kwargs: {})
    monkeypatch.setattr(runtime, "guard_reference", lambda *args: trace.append("guard"))
    monkeypatch.setattr(runtime, "load_guidance", lambda model, path, probability: "guidance")
    monkeypatch.setattr(runtime, "DGGCallback", Guidance)
    monkeypatch.setenv("MCITLIB_METHOD", "dgg")
    monkeypatch.delenv("MCITLIB_MODEL_OVERRIDE", raising=False)
    with pytest.raises(StopTraining):
        standard_train.train()
    assert trace == (["guard", "guidance-created", "train", "guidance-closed"] if history != "none"
                     else ["guard", "train"])
