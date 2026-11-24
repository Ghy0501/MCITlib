# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""CPU algorithm, PEFT checkpoint, continuation and inference tests."""
from __future__ import annotations

import copy

import pytest
import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from mcitlib.methods.same import SAMEMethod
from mcitlib.methods.sigma import SiGMAMethod
from mcitlib.tuners._peft_registry import install_method_adapter
from mcitlib.tuners.variants.same import SAMELinear
from mcitlib.tuners.variants.sigma import SiGMALinear, inference_delta


def tiny():
    return LlamaForCausalLM(LlamaConfig(vocab_size=40, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32))


def setup(method, task=0):
    return method({'r': 4, 'lora_alpha': 8, 'target_modules': ['gate_proj'], 'expert_num': 2, 'cur_task': task})


def test_sigma_masks_scale_and_merge():
    anchor = torch.tensor([[2., -2., 0., 1.]])
    other = torch.tensor([[3., 1., 2., -.1]])
    # Same signs add 3; the opposing 1 survives (> .8*2 is false),
    # so it and the last small conflict are dropped; zero anchor retains 2.
    expected = torch.tensor([[5., -2., 2., 1.]])
    assert torch.allclose(inference_delta([anchor, other]), expected)
    anchor = torch.tensor([[2., -2., 1.]])
    other = torch.tensor([[3., 2., -1.]])
    # The selected conflicts exactly oppose the anchor: cosine distance = 2.
    assert torch.allclose(inference_delta([anchor, other]), torch.tensor([[5., 2., -1.]]))


@pytest.mark.parametrize('method_cls,method_name,linear', [
    (SAMEMethod, 'same', SAMELinear), (SiGMAMethod, 'sigma', SiGMALinear),
])
def test_save_reload_continuation_and_inference(tmp_path, method_cls, method_name, linear):
    torch.manual_seed(20)
    base = tiny()
    base_state = copy.deepcopy(base.state_dict())
    install_method_adapter(method_name)
    method = setup(method_cls)
    model = method.build_trainable(base, None)
    model.train()
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    model(input_ids=inputs, labels=inputs).loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-.1)
    if method_name == 'same':
        method.finish_task(model)
        for layer in model.modules():
            if isinstance(layer, SAMELinear):
                layer.set_global_prior(torch.ones(layer.expert_num))
    model.eval()
    expected = model(input_ids=inputs).logits.detach()
    saved = {name: value.detach().clone() for name, value in model.named_parameters() if 'lora_' in name}
    model.save_pretrained(tmp_path, state_dict=saved, safe_serialization=False)

    rebase = tiny()
    rebase.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(rebase, tmp_path)
    loaded.eval()
    if method_name == 'same':
        for layer in loaded.modules():
            if isinstance(layer, SAMELinear):
                layer.set_global_prior(torch.ones(layer.expert_num))
    actual = loaded(input_ids=inputs).logits.detach()
    assert torch.allclose(actual, expected, atol=1e-6)
    if method_name == 'same':
        assert all(m.state['valid'].item() == 1 for m in loaded.modules() if isinstance(m, linear))
        for layer in loaded.modules():
            if isinstance(layer, linear):
                layer.train()  # Newly injected children can still be in training mode.
        merged = loaded.merge_and_unload()
        assert all(not m.training for m in merged.modules() if isinstance(m, linear))
        assert torch.allclose(merged(input_ids=inputs).logits, expected, atol=1e-6)
    else:
        merged = loaded.merge_and_unload()
        merged.eval()
        assert torch.allclose(merged(input_ids=inputs).logits, expected, atol=1e-6)

    # Fresh task 2 model loads every historical adapter and SAME state using the
    # same PEFT state mapper used by the training entry, then takes a backward step.
    from peft.utils import set_peft_model_state_dict
    continued_base = tiny()
    continued_base.load_state_dict(base_state)
    next_method = setup(method_cls, task=1)
    continued = next_method.build_trainable(continued_base, None)
    weights = torch.load(tmp_path / 'adapter_model.bin', map_location='cpu')
    set_peft_model_state_dict(continued, weights)
    next_method.load_previous(continued, tmp_path)
    continued.train()
    continued(input_ids=inputs, labels=inputs).loss.backward()
    layer = next(m for m in continued.modules() if isinstance(m, linear))
    if method_name == 'same':
        assert layer.state['valid'].item() == 1
        assert layer.lora_A['default'].loraA[0].weight.grad is not None
    else:
        assert layer.lora_A['default'].loraA[0].weight.grad is None
        assert layer.lora_A['default'].loraA[1].weight.grad is not None
        assert layer.cur_task == 1
        assert layer.lora_A['default'].loraA[1].weight.shape[0] == 4


def test_same_curvature_and_router_projection():
    install_method_adapter('same')
    method = setup(SAMEMethod, task=1)
    model = method.build_trainable(tiny(), None)
    layer = next(m for m in model.modules() if isinstance(m, SAMELinear))
    layer.train()
    with torch.no_grad():
        layer.state['valid'].fill_(1)
        layer.state['u_prev'].zero_()
        layer.state['u_prev'][0, 0] = 1
        layer.state['s_prev'].zero_()
        layer.state['s_prev'][0] = 3
        layer.state['u'].copy_(layer.state['u_prev'])
        layer.state['s'].copy_(layer.state['s_prev'])
        layer.state['step'].fill_(10)
    grad = torch.ones_like(layer.lora_A['default'].loraA[0].weight)
    scaled = layer._expert_grad(grad)
    assert torch.allclose(scaled[:, 0], torch.full_like(scaled[:, 0], 1 / 3.9))
    assert torch.allclose(scaled[:, 1:], torch.full_like(scaled[:, 1:], 1 / .9))
    assert layer._router_grad(torch.ones_like(layer.lora_router['default'].weight)).shape == layer.lora_router['default'].weight.shape


def test_same_checkpoint_recompute_does_not_repeat_statistics():
    from torch.utils.checkpoint import checkpoint
    install_method_adapter('same')
    model = setup(SAMEMethod).build_trainable(tiny(), None)
    layer = next(m for m in model.modules() if isinstance(m, SAMELinear))
    layer.train()
    x = torch.randn(2, 4, layer.in_features, requires_grad=True)
    checkpoint(layer, x, use_reentrant=False).sum().backward()
    assert layer.state['step'].item() == 1
    assert layer.state['count'].item() == 8
    assert x.grad is not None


def test_same_production_checkpoint_scope_preserves_single_statistics_update():
    from types import SimpleNamespace

    from torch.utils import checkpoint as checkpoint_module

    from mcitlib.backbones.llava.olora_checkpoint import olora_checkpoint_scope

    install_method_adapter('same')
    method = setup(SAMEMethod)
    model = method.build_trainable(tiny(), None)
    layer = next(m for m in model.modules() if isinstance(m, SAMELinear))
    layer.train()
    x = torch.randn(2, 4, layer.in_features, requires_grad=True)
    original = checkpoint_module.checkpoint
    with olora_checkpoint_scope(method, SimpleNamespace(gradient_checkpointing=True)) as observation:
        checkpoint_module.checkpoint(layer, x).sum().backward()
        assert observation is not None and observation.calls == 1
    assert checkpoint_module.checkpoint is original
    assert layer.state['step'].item() == 1 and layer.state['count'].item() == 8
    assert x.grad is not None


def test_sigma_inference_merge_rounds_each_term_in_base_dtype():
    install_method_adapter('sigma')
    layer = SiGMALinear('default', 1, 1, r=1, lora_alpha=1, expert_num=3, cur_task=2)
    layer.weight.data = torch.tensor([[1000.]], dtype=torch.float16)
    with torch.no_grad():
        for i, value in enumerate((0.1, 0.3, 0.3)):
            layer.lora_A['default'].loraA[i].weight.fill_(1)
            layer.lora_B['default'].loraB[i].weight.fill_(value)
    layer.eval()
    before = layer._add_merged_to(layer.weight.detach().clone())
    layer.merge()
    assert layer.weight.item() == 1001.0
    assert torch.equal(layer.weight, before)
    combined = torch.tensor([[1000.]], dtype=torch.float16)
    combined.add_(inference_delta(layer._deltas()))
    assert combined.item() == 1000.5
    with pytest.raises(RuntimeError, match='cannot be reversed'):
        layer.unmerge()
