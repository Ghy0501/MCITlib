# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise the shared Trainer with real MoE arguments and its modality sampler."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("deepspeed")

from mcitlib.backbones.llava.moe_train import TrainingArguments  # noqa: E402
from mcitlib.backbones.llava.trainer import LengthGroupedSampler, LLaVATrainer  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    @property
    def modality_lengths(self):
        return [3, 4, 5, 6]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        ids = torch.tensor([1, index + 2, 7], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def test_real_moe_trainer_sampler_and_two_optimizer_steps(tmp_path) -> None:
    config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
    model = transformers.LlamaForCausalLM(config)
    initial = model.lm_head.weight.detach().clone()
    args = TrainingArguments(output_dir=str(tmp_path), no_cuda=True, max_steps=2, per_device_train_batch_size=1,
                             gradient_accumulation_steps=1, group_by_modality_length=True, mm_projector_lr=2e-5,
                             learning_rate=1e-3, logging_steps=1, save_strategy="no", report_to="none",
                             disable_tqdm=True, dataloader_num_workers=0)
    assert args.is_SAT is False
    assert all(hasattr(args, name) for name in (
        "group_by_modality_length", "is_SAT", "train_batch_size", "world_size", "gradient_accumulation_steps",
        "mm_projector_lr", "weight_decay", "local_rank",
    ))
    trainer = LLaVATrainer(model=model, args=args, train_dataset=TinyDataset())
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, LengthGroupedSampler)
    assert sorted(iter(sampler)) == [0, 1, 2, 3]
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    optimizer = trainer.create_optimizer()
    assert optimizer is not None
    result = trainer.train()
    assert result.global_step == 2
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(initial, model.lm_head.weight.detach())
