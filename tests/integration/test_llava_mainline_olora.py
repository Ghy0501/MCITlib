# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU tests against the pinned HF Trainer and real OLoRA adapter."""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint as checkpoint_module
import transformers
from torch.utils.data import Dataset

from mcitlib.backbones.llava import olora_checkpoint as olora_checkpoint
from mcitlib.backbones.llava.trainer import LLaVATrainer
from mcitlib.methods.olora import OLoRAMethod
from mcitlib.tuners._peft_registry import install_method_adapter


def tiny_model():
    torch.manual_seed(1223)
    install_method_adapter("olora")
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32, use_cache=False,
    )
    base = transformers.LlamaForCausalLM(config)
    base.train()
    base.enable_input_require_grads()
    method = OLoRAMethod({"cur_task": 1, "expert_num": 2, "r": 4, "lora_alpha": 8,
                          "target_modules": ["q_proj"], "lora_dropout": 0.0})
    model = method.build_trainable(base, None)
    model.train()
    for name, parameter in model.named_parameters():
        if "lora_B.default.loraB.1.weight" in name:
            with torch.no_grad():
                parameter.normal_(0.0, 0.1)
    return model, method


def selected(model, needle):
    matches = [(name, parameter) for name, parameter in model.named_parameters() if needle in name]
    assert len(matches) == 1
    return matches[0][1]


def run_backward(mode):
    model, method = tiny_model()
    current = selected(model, "lora_A.default.loraA.1.weight")
    previous = selected(model, "lora_A.default.loraA.0.weight")
    previous_b = selected(model, "lora_B.default.loraB.0.weight")
    before_previous = (previous.detach().clone(), previous_b.detach().clone())
    hits = []
    current.register_hook(lambda gradient: hits.append(gradient.detach().clone()))
    args = SimpleNamespace(gradient_checkpointing=True, per_device_train_batch_size=4,
                           gradient_accumulation_steps=4, world_size=4)
    initial_checkpoint = checkpoint_module.checkpoint
    ids = torch.tensor([[1, 2, 3, 4, 5]])

    def backward():
        output = model(input_ids=ids, labels=ids)
        extra = method.extra_loss(model, None, None, output.loss)
        expected = 0.05 * torch.abs(current @ previous.detach().T).sum()
        torch.testing.assert_close(extra, expected)
        assert extra.item() > 0
        (output.loss + extra).backward()
        return output.loss.detach(), extra.detach()

    if mode == "uncheckpointed":
        loss, extra = backward()
    elif mode == "reentrant":
        model.gradient_checkpointing_enable()
        loss, extra = backward()
    else:
        with olora_checkpoint.olora_checkpoint_scope(method, args) as observation:
            # This is the same no-argument enable that HF 4.31 Trainer repeats.
            model.gradient_checkpointing_enable()
            loss, extra = backward()
            assert observation.calls == 1
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
        assert args.gradient_checkpointing is True
        assert args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size == 64
    assert checkpoint_module.checkpoint is initial_checkpoint
    assert previous.grad is None and previous_b.grad is None
    gradient = current.grad.detach().clone()
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1).step()
    assert torch.equal(previous, before_previous[0]) and torch.equal(previous_b, before_previous[1])
    return loss, extra, gradient, hits


def test_real_checkpoint_combines_olora_gradients_once():
    base = run_backward("uncheckpointed")
    old = run_backward("reentrant")
    fixed = run_backward("nonreentrant")
    assert len(old[3]) == 2 and len(fixed[3]) == 1
    for index in (0, 1, 2):
        torch.testing.assert_close(fixed[index], base[index], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(old[index], base[index], rtol=1e-5, atol=1e-7)


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        ids = torch.tensor([1, 2, 3, 4, (index % 8) + 5])
        return {"input_ids": ids, "labels": ids.clone()}


def test_real_hf_trainer_reenables_checkpoint_without_reentrant(tmp_path):
    model, method = tiny_model()
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), no_cuda=True, per_device_train_batch_size=1,
        gradient_accumulation_steps=2, gradient_checkpointing=True, max_steps=2,
        save_strategy="no", report_to=[], disable_tqdm=True, learning_rate=0.001,
    )
    args.group_by_modality_length = False
    args.is_SAT = False
    args.mm_projector_lr = None
    trainer = LLaVATrainer(model=model, args=args, train_dataset=TinyDataset())
    trainer.mcit_method = method
    calls = []
    old_enable = model.gradient_checkpointing_enable

    def enabled(*args, **kwargs):
        calls.append((args, kwargs))
        return old_enable(*args, **kwargs)

    model.gradient_checkpointing_enable = enabled
    current = selected(model, "lora_A.default.loraA.1.weight")
    previous = selected(model, "lora_A.default.loraA.0.weight")
    before_previous = previous.detach().clone()
    hooks = []
    current.register_hook(lambda grad: hooks.append(grad.detach().clone()))
    original = checkpoint_module.checkpoint
    with olora_checkpoint.olora_checkpoint_scope(method, args) as observation:
        result = trainer.train()
        assert observation.calls == 4
    assert result.global_step == 2 and len(hooks) == 4
    assert calls == [((), {})]
    assert model.is_gradient_checkpointing and args.gradient_checkpointing
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert torch.equal(previous, before_previous)
    assert checkpoint_module.checkpoint is original


def test_training_entry_uses_scope_with_real_hf_trainer(tmp_path, monkeypatch):
    from mcitlib.backbones.llava.overrides import import_llava

    import_llava()
    from mcitlib.backbones.llava import moe_train as entry

    root = Path(__file__).resolve().parents[2]
    assert Path(entry.__file__).resolve() == root / "mcitlib/backbones/llava/moe_train.py"
    assert Path(olora_checkpoint.__file__).resolve() == root / "mcitlib/backbones/llava/olora_checkpoint.py"
    install_method_adapter("olora")
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32, use_cache=False,
    )
    base = transformers.LlamaForCausalLM(config)
    model_args = entry.ModelArguments(model_name_or_path="tiny-local", version="v1", cur_task=1, expert_num=2)
    data_args = entry.DataArguments(data_path="tiny-local")
    args = entry.TrainingArguments(
        output_dir=str(tmp_path), no_cuda=True, per_device_train_batch_size=1,
        gradient_accumulation_steps=2, gradient_checkpointing=True, max_steps=2,
        save_strategy="no", report_to=[], disable_tqdm=True, learning_rate=0.001,
        lora_enable=True, lora_r=4, lora_alpha=8, lora_dropout=0.0,
    )
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (model_args, data_args, args))
    monkeypatch.setattr(entry.transformers, "HfArgumentParser", lambda *_: parser)
    monkeypatch.setattr(entry.transformers.LlamaForCausalLM, "from_pretrained", lambda *_, **__: base)
    monkeypatch.setattr(entry.transformers.AutoTokenizer, "from_pretrained",
                        lambda *_, **__: SimpleNamespace(unk_token="<unk>"))
    monkeypatch.setattr(entry, "make_supervised_data_module", lambda **_: {
        "train_dataset": TinyDataset(), "data_collator": transformers.default_data_collator,
    })
    monkeypatch.setenv("MCITLIB_METHOD", "olora")
    monkeypatch.delenv("MCITLIB_MODEL_OVERRIDE", raising=False)
    calls, observed = [], {}
    original_checkpoint = checkpoint_module.checkpoint

    @wraps(original_checkpoint)
    def observe_checkpoint(*values, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return original_checkpoint(*values, **kwargs)

    monkeypatch.setattr(checkpoint_module, "checkpoint", observe_checkpoint)

    class StopAfterTraining(Exception):
        pass

    class ObservedTrainer(LLaVATrainer):
        def train(self, *values, **kwargs):
            previous_a = selected(self.model, "q_proj.lora_A.default.loraA.0.weight")
            previous_b = selected(self.model, "q_proj.lora_B.default.loraB.0.weight")
            before_previous = (previous_a.detach().clone(), previous_b.detach().clone())
            current_a = selected(self.model, "q_proj.lora_A.default.loraA.1.weight")
            gradients = []
            current_a.register_hook(lambda grad: gradients.append(grad.detach().clone()))
            result = super().train(*values, **kwargs)
            observed["global_step"] = result.global_step
            observed["previous_unchanged"] = (
                torch.equal(previous_a, before_previous[0]) and torch.equal(previous_b, before_previous[1])
            )
            observed["checkpoint_enabled"] = self.model.is_gradient_checkpointing
            observed["current_gradients"] = (
                len(gradients) == 4 and all(torch.isfinite(grad).all() for grad in gradients)
                and any(torch.count_nonzero(grad) > 0 for grad in gradients)
            )
            raise StopAfterTraining

    monkeypatch.setattr(entry, "LLaVATrainer", ObservedTrainer)
    with pytest.raises(StopAfterTraining):
        entry.train()
    assert observed == {"global_step": 2, "previous_unchanged": True, "checkpoint_enabled": True,
                        "current_gradients": True}
    assert calls == [False] * 4
    assert args.gradient_checkpointing and args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert checkpoint_module.checkpoint is observe_checkpoint


def test_other_methods_and_disabled_checkpoint_are_unchanged():
    original = checkpoint_module.checkpoint
    method = SimpleNamespace(id="moelora", training_overrides=lambda: pytest.fail("unexpected override"))
    args = SimpleNamespace(gradient_checkpointing=True)
    with olora_checkpoint.olora_checkpoint_scope(method, args) as observation:
        assert observation is None and checkpoint_module.checkpoint is original
    assert not hasattr(args, "gradient_checkpointing_kwargs")
    args.gradient_checkpointing = False
    with olora_checkpoint.olora_checkpoint_scope(OLoRAMethod({}), args) as observation:
        assert observation is None and checkpoint_module.checkpoint is original


def test_scope_rejects_conflicts_and_restores_on_errors(monkeypatch):
    method = OLoRAMethod({"cur_task": 1})
    args = SimpleNamespace(gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": True})
    original = checkpoint_module.checkpoint
    with pytest.raises(RuntimeError, match="conflicting gradient_checkpointing_kwargs"):
        with olora_checkpoint.olora_checkpoint_scope(method, args):
            pass
    args.gradient_checkpointing_kwargs = None
    with pytest.raises(RuntimeError, match="conflicting reentrant checkpoint call"):
        with olora_checkpoint.olora_checkpoint_scope(method, args):
            checkpoint_module.checkpoint(lambda x: x, torch.ones(1), use_reentrant=True)
    assert checkpoint_module.checkpoint is original
    with pytest.raises(RuntimeError, match="did not invoke"):
        with olora_checkpoint.olora_checkpoint_scope(method, args):
            pass
    assert checkpoint_module.checkpoint is original

    @wraps(original)
    def unsupported_at_call(*values, **kwargs):
        assert kwargs["use_reentrant"] is False
        raise TypeError("non-reentrant execution unavailable")

    monkeypatch.setattr(checkpoint_module, "checkpoint", unsupported_at_call)
    with pytest.raises(TypeError, match="non-reentrant execution unavailable"):
        with olora_checkpoint.olora_checkpoint_scope(method, args):
            checkpoint_module.checkpoint(lambda x: x, torch.ones(1))
    assert checkpoint_module.checkpoint is unsupported_at_call
    monkeypatch.setattr(checkpoint_module, "checkpoint", lambda function, *values: function(*values))
    with pytest.raises(RuntimeError, match="does not support use_reentrant"):
        with olora_checkpoint.olora_checkpoint_scope(method, args):
            pass
