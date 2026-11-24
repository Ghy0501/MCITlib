# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Octopus CPU checks against independent mathematical expectations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from mcitlib.methods._octopus.core import regularization, sample_indices
from mcitlib.methods._octopus.gpwc import (
    _enable_flash_attention,
    collect_mean,
    reduce_history,
    supervised_token_counts,
)
from mcitlib.methods._octopus.history import attach_history
from mcitlib.methods._octopus.state import load_manifest, write_manifest
from mcitlib.methods._octopus.workflow import sample


def tiny():
    config = LlamaConfig(vocab_size=24, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=16)
    return LlamaForCausalLM(config)


def adapter(model):
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'],
                                           lora_dropout=0, task_type='CAUSAL_LM'))


def test_repeat_sampling_matches_release_and_does_not_cap(tmp_path):
    src = tmp_path / 'in.json'
    out = tmp_path / 'out.json'
    src.write_text(json.dumps([{'id': i} for i in range(3)]))
    sample(src, out, 8)
    assert [row['id'] for row in json.loads(out.read_text())] == sample_indices(3, 8)
    assert len(json.loads(out.read_text())) == 8


def test_stage_two_regularizer_only_changes_b_and_preserves_accumulation():
    a = torch.tensor([[1., 2.]], requires_grad=True)
    b = torch.tensor([[1.], [3.]], requires_grad=True)
    g = torch.tensor([[1., 2.], [3., 1.]])
    reg, _, _ = regularization([(a, b, g)], .2, .1)
    grads = torch.autograd.grad(reg, (a, b), allow_unused=True, retain_graph=True)
    assert grads[0] is None and grads[1] is not None and grads[1].abs().sum() > 0
    task = (a.square().sum() + b.square().sum())
    mixed = (task + 8 * reg) / 8
    mixed_grads = torch.autograd.grad(mixed, (a, b), retain_graph=True)
    explicit = task / 8 + regularization([(a, b, g)], .2, .1)[0]
    direct_grads = torch.autograd.grad(explicit, (a, b))
    assert all(torch.allclose(x, y) for x, y in zip(mixed_grads, direct_grads))


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 5)
        self.head = nn.Linear(5, 12)

    def forward(self, input_ids, labels):
        logits = self.head(self.embedding(input_ids))[:, :-1]
        loss = F.cross_entropy(logits.float().reshape(-1, 12), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


def test_full_model_gpwc_microbatch_preserves_logical_mean():
    torch.manual_seed(3)
    left = _TinyLM()
    right = copy.deepcopy(left)
    samples = [
        {'input_ids': torch.tensor([1, 2, 3, 4]), 'labels': torch.tensor([-100, 2, 3, 4])},
        {'input_ids': torch.tensor([3, 4, 5, 6]), 'labels': torch.tensor([-100, -100, 5, 6])},
        {'input_ids': torch.tensor([2, 4, 6, 8]), 'labels': torch.tensor([-100, 4, -100, 8])},
        {'input_ids': torch.tensor([1, 3, 5, 7]), 'labels': torch.tensor([-100, 3, 5, 7])},
    ]
    loader = DataLoader(samples, batch_size=2)
    a, n = collect_mean(left, loader, torch.device('cpu'), 1)
    b, m = collect_mean(right, loader, torch.device('cpu'), 2)
    assert n == m == 2 and set(a) == set(b) == {name for name, _ in left.named_parameters()}
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-7), name


def test_gpwc_rejects_unmasked_first_label():
    samples = [{'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([1, 2, 3])}]
    with pytest.raises(ValueError, match='masked first label'):
        collect_mean(_TinyLM(), DataLoader(samples, batch_size=1), torch.device('cpu'), 1)


def test_gpwc_counts_tokens_after_image_expansion_and_truncation():
    tower = SimpleNamespace(num_patches=3, select_feature='patch')
    model = SimpleNamespace(get_vision_tower=lambda: tower, config=SimpleNamespace(tokenizer_model_max_length=7))
    batch = {'input_ids': torch.tensor([[1, -200, 2, 3, 4, 5], [1, 2, 3, 4, 0, 0]]),
             'labels': torch.tensor([[-100, -100, 2, 3, 4, 5], [-100, 2, 3, 4, -100, -100]]),
             'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
             'images': torch.zeros(2, 3, 4, 4)}
    # Image patches replace the placeholder; the final supervised 5 is cut.
    assert supervised_token_counts(model, batch) == [3, 3]


def test_gpwc_flash_patch_matches_eager_full_model_gradients_on_cpu(monkeypatch):
    """Exercise the real V4 attention wrapper with an equivalent CPU SDPA kernel.

    The real CUDA FlashAttention kernel still needs hardware validation; this
    checks the V4 packing, causal masks, padding, loss and all model gradients.
    """
    from llava.train import llama_flash_attn_monkey_patch as flash
    from transformers.models.llama import modeling_llama

    def cpu_kernel(qkv, cu_lengths, max_length, dropout, softmax_scale=None, causal=False):
        assert dropout == 0 and softmax_scale is None and causal and max_length > 0
        rows = []
        for start, stop in zip(cu_lengths[:-1].tolist(), cu_lengths[1:].tolist()):
            q, k, v = qkv[start:stop].unbind(dim=1)
            q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            rows.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(rows, dim=0)

    torch.manual_seed(931)
    eager = tiny().to(torch.bfloat16)
    # Make attention gradients large enough to expose masking or packing errors.
    with torch.no_grad():
        for name, parameter in eager.named_parameters():
            if '.self_attn.' in name and name.endswith('.weight'):
                parameter.mul_(5)
    patched = copy.deepcopy(eager)
    samples = [
        {'input_ids': torch.tensor([1, 2, 3, 4, 5]), 'labels': torch.tensor([-100, 2, 3, 4, 5]),
         'attention_mask': torch.tensor([1, 1, 1, 1, 1])},
        {'input_ids': torch.tensor([1, 5, 2, 0, 0]), 'labels': torch.tensor([-100, -100, 2, -100, -100]),
         'attention_mask': torch.tensor([1, 1, 1, 0, 0])},
    ]
    loader = DataLoader(samples, batch_size=2)
    reference, _ = collect_mean(eager, loader, torch.device('cpu'), 2)
    # Save class attributes for automatic restoration after the global patch.
    monkeypatch.setattr(modeling_llama.LlamaAttention, 'forward', modeling_llama.LlamaAttention.forward)
    monkeypatch.setattr(modeling_llama.LlamaModel, '_prepare_decoder_attention_mask',
                        modeling_llama.LlamaModel._prepare_decoder_attention_mask)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (8, 0))
    monkeypatch.setattr(flash, 'flash_attn_unpadded_qkvpacked_func', cpu_kernel)
    _enable_flash_attention()
    result, _ = collect_mean(patched, loader, torch.device('cpu'), 2)
    assert reference.keys() == result.keys()
    deviations = {name: (reference[name].float() - result[name].float()).abs().max().item() for name in reference}
    worst = max(deviations, key=deviations.get)
    targets = {name: error for name, error in deviations.items() if '.self_attn.' in name or '.mlp.' in name}
    worst_target = max(targets, key=targets.get)
    print(f'BF16 eager versus V4 packing + CPU SDPA: max gradient deviation {deviations[worst]} ({worst}); '
          f'projection max {targets[worst_target]} ({worst_target})')
    assert max(deviations.values()) < .01, deviations


def test_history_real_peft_checkpoint_reconstruction(tmp_path: Path):
    torch.manual_seed(12)
    base = tiny()
    base_state = copy.deepcopy(base.state_dict())
    first = adapter(base)
    with torch.no_grad():
        for name, value in first.named_parameters():
            if 'lora_B' in name:
                value.normal_()
    stage1 = tmp_path / 'Task1_llava_lora'
    first.save_pretrained(stage1, safe_serialization=False)
    write_manifest(stage1, [])
    current = tiny()
    current.load_state_dict(base_state)
    current = adapter(current)
    history = torch.load(stage1 / 'adapter_model.bin', map_location='cpu')
    attach_history(current, [history], [2.0])
    current.eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    before = current(input_ids=inputs).logits.detach()
    second = tmp_path / 'Task2_llava_lora'
    current.save_pretrained(second, safe_serialization=False)
    write_manifest(second, [str(stage1)])
    assert load_manifest(second) == [stage1.resolve()]
    restored_base = tiny()
    restored_base.load_state_dict(base_state)
    from peft import PeftModel
    restored = PeftModel.from_pretrained(restored_base, second, is_trainable=True)
    attach_history(restored, [history], [2.0])
    restored.eval()
    assert torch.allclose(restored(input_ids=inputs).logits, before, atol=1e-6)
    assert all(not p.requires_grad for name, p in restored.named_parameters() if '_octopus_history' in name)


def test_bfloat16_microbatch_deviation_and_history_reduction():
    torch.manual_seed(73)
    left = _TinyLM().to(torch.bfloat16)
    right = copy.deepcopy(left)
    samples = [
        {'input_ids': torch.tensor([1, 2, 3, 4]), 'labels': torch.tensor([-100, 2, 3, 4])},
        {'input_ids': torch.tensor([3, 4, 5, 6]), 'labels': torch.tensor([-100, -100, 5, 6])},
        {'input_ids': torch.tensor([2, 4, 6, 8]), 'labels': torch.tensor([-100, 4, -100, 8])},
        {'input_ids': torch.tensor([1, 3, 5, 7]), 'labels': torch.tensor([-100, 3, 5, 7])},
    ]
    loader = DataLoader(samples, batch_size=2)
    micro, _ = collect_mean(left, loader, torch.device('cpu'), 1)
    full, _ = collect_mean(right, loader, torch.device('cpu'), 2)
    deviations = [(micro[name].float() - full[name].float()).abs().max().item() for name in micro]
    # The loss is equivalent, but repeated bf16 gradient additions can round.
    assert max(deviations) < .01
    assert max(deviations) > 0
    name = 'head.weight'
    histories = [{name: micro[name]}, {name: full[name]}, {name: micro[name]}]
    assert torch.equal(reduce_history(histories, {name}, 'cpu')[name],
                       torch.stack([entry[name] for entry in histories]).sum(0))


def test_evaluation_fp32_merge_before_fp16_cast_differs_from_bf16_merge(tmp_path: Path):
    from peft import PeftModel

    torch.manual_seed(4)
    original = tiny()
    with torch.no_grad():
        original.model.layers[0].self_attn.q_proj.weight.fill_(1.0)
    base = copy.deepcopy(original.state_dict())
    checkpoints = []
    for i in range(2):
        raw = tiny()
        raw.load_state_dict(base)
        wrapped = adapter(raw)
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    parameter.fill_(.03)
        path = tmp_path / f'adapter{i}'
        wrapped.save_pretrained(path, safe_serialization=False)
        checkpoints.append(path)

    results = []
    for dtype in (torch.float32, torch.bfloat16):
        model = tiny()
        model.load_state_dict(base)
        model = model.to(dtype)
        for checkpoint in checkpoints:
            model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
        results.append(model.model.layers[0].self_attn.q_proj.weight.detach().clone())
    # Evaluation loads an FP32 base, merges in order, then converts to FP16.
    # GPWC uses BF16 base and therefore must not silently reuse that merge.
    assert torch.allclose(results[0], torch.full_like(results[0], 1.0072), atol=1e-6)
    assert not torch.equal(results[0].to(torch.bfloat16), results[1])
    assert torch.isfinite(results[0].to(torch.float16)).all()


def test_stage_two_runtime_restores_history_and_cleanup_receipt(tmp_path: Path, monkeypatch):
    from mcitlib.methods._octopus.runtime import setup_stage_two
    from mcitlib.methods._octopus.state import digest
    from mcitlib.methods._octopus.workflow import main as workflow_main

    torch.manual_seed(123)
    first = adapter(tiny())
    with torch.no_grad():
        for name, parameter in first.named_parameters():
            if 'lora_B' in name:
                parameter.normal_()
    first_path = tmp_path / 'Task1_llava_lora'
    first.save_pretrained(first_path, safe_serialization=False)
    write_manifest(first_path, [])
    current = adapter(tiny())
    gradients = {}
    for name, layer in current.named_modules():
        if hasattr(layer, 'lora_A') and 'default' in layer.lora_A:
            key = name.removeprefix('base_model.model.') + '.weight'
            gradients[key] = torch.randn(layer.lora_B['default'].weight.shape[0],
                                         layer.lora_A['default'].weight.shape[1])
    path = tmp_path / 'task2_gpwc.pt'
    torch.save({'version': 1, 'history': [str(first_path.resolve())], 'gradients': gradients}, path)
    audit = path.with_suffix('.pt.audit.json')
    audit.write_text(json.dumps({'gpwc_sha256': digest(path)}))
    for key, value in {'MCITLIB_OCTOPUS_HISTORY': json.dumps([str(first_path)]),
                       'MCITLIB_OCTOPUS_GPWC': str(path), 'MCITLIB_OCTOPUS_LAMBDA1': '.02',
                       'MCITLIB_OCTOPUS_LAMBDA2': '.01'}.items():
        monkeypatch.setenv(key, value)
    objective = setup_stage_two(current, 8)
    current.train()
    inputs = torch.tensor([[1, 2, 3, 4]])
    task_loss = current(input_ids=inputs, labels=inputs).loss
    objective.extra_loss(current, None, None, task_loss).backward()
    assert any(p.grad is not None for n, p in current.named_parameters() if 'lora_B' in n)
    assert all(p.grad is None for n, p in current.named_parameters() if 'lora_A' in n)
    second_path = tmp_path / 'Task2_llava_lora'
    current.save_pretrained(second_path, safe_serialization=False)
    write_manifest(second_path, [str(first_path)], objective.provenance)
    manifest = json.loads((second_path / 'octopus_manifest.json').read_text())
    assert not Path(manifest['gpwc']['audit']).is_absolute()
    receipt = tmp_path / 'task2_gpwc.pt.receipt.json'
    monkeypatch.setattr('sys.argv', ['workflow', 'cleanup', '--input', str(path), '--checkpoint', str(second_path),
                                     '--receipt', str(receipt)])
    original_audit = audit.read_text()
    audit.write_text(original_audit + ' ')
    with pytest.raises(ValueError, match='audit checksum'):
        workflow_main()
    assert path.is_file() and not receipt.exists()
    audit.write_text(original_audit)
    workflow_main()
    assert not path.exists() and audit.is_file()
    assert json.loads(receipt.read_text())['adapter_reload'] == 'passed'
