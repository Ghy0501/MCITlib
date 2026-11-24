# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise one train-entry method in a fresh interpreter, without models or GPUs."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers

from mcitlib.backbones.llava.overrides import import_llava

import_llava()
from llava.train import llama_flash_attn_monkey_patch  # noqa: E402
from peft.utils import PeftType  # noqa: E402

assert "mcitlib.tuners._peft_registry" not in sys.modules
assert not hasattr(PeftType, "MOE_LORA_CoIN")
llama_flash_attn_monkey_patch.replace_llama_attn_with_flash_attn = lambda: None

method = os.environ["MCITLIB_METHOD"]
output = Path(sys.argv[1])
config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
model = transformers.LlamaForCausalLM(config)
if method == "hide":
    model.expert_weight = [0.0] * 8

model_args = SimpleNamespace(
    model_name_or_path="tiny-local", vision_tower="tiny-vision" if method in {"hide", "disco"} else None,
    text_tower="tiny-text", mm_text_select_layer=-1, freeze_backbone=False,
    cur_task=0, task="UCIT_ImgNet" if method == "clmoe" else "", task_embedding_dim=12,
    expert_num=6, ins_type=0, ins_emb=str(Path(__file__).resolve().parents[2] / "assets/smolora/ins_emb_multi_ucit.pkl"),
)
training_args = SimpleNamespace(
    local_rank=0, fp16=False, bf16=False, bits=16, gradient_checkpointing=False,
    lora_enable=True, lora_r=18, lora_alpha=36, lora_dropout=0.0, lora_bias="none",
    output_dir=str(output), cache_dir=str(output), model_max_length=32,
)
parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (model_args, SimpleNamespace(), training_args))
transformers.HfArgumentParser = lambda *_: parser
transformers.LlamaForCausalLM.from_pretrained = lambda *_, **__: model
transformers.PreTrainedModel.from_pretrained = classmethod(lambda cls, *_, **__: model)
transformers.AutoConfig.from_pretrained = lambda *_, **__: config


class StopAfterAdapter(Exception):
    pass


def stop_at_tokenizer(*args, **kwargs):
    raise StopAfterAdapter


transformers.AutoTokenizer.from_pretrained = stop_at_tokenizer
import peft  # noqa: E402

actual_get_peft_model = peft.get_peft_model
captured = {}


def track_model(*args, **kwargs):
    value = actual_get_peft_model(*args, **kwargs)
    captured["model"] = value
    return value


peft.get_peft_model = track_model
from mcitlib import methods as methods_module  # noqa: E402

actual_create_method = methods_module.create_method


def track_method(*args, **kwargs):
    value = actual_create_method(*args, **kwargs)
    captured["method"] = value
    return value


methods_module.create_method = track_method
from mcitlib.backbones.llava import train_entry  # noqa: E402

try:
    train_entry.main()
except StopAfterAdapter:
    pass
else:
    raise AssertionError("train entry did not reach adapter construction")

adapted = captured["model"]
from peft import peft_model as peft_model_module  # noqa: E402
from peft import utils as peft_utils  # noqa: E402
from peft.utils import save_and_load  # noqa: E402

from mcitlib.backbones.llava import moe_train  # noqa: E402

assert save_and_load._mcit_lora_like_support
assert peft_utils.get_peft_model_state_dict is save_and_load.get_peft_model_state_dict
assert peft_utils.set_peft_model_state_dict is save_and_load.set_peft_model_state_dict
assert peft_model_module.get_peft_model_state_dict is save_and_load.get_peft_model_state_dict
assert peft_model_module.set_peft_model_state_dict is save_and_load.set_peft_model_state_dict
output.mkdir(parents=True, exist_ok=True)
before = {key: value.detach().clone() for key, value in peft_utils.get_peft_model_state_dict(adapted).items()}
assert before
adapted.save_pretrained(output)
torch.save({}, output / "non_lora_trainables.bin")
with torch.no_grad():
    for name, param in adapted.named_parameters():
        if "lora_" in name:
            param.add_(1)
assert any(not torch.equal(before[key], value) for key, value in peft_utils.get_peft_model_state_dict(adapted).items())
moe_train.load_model_from_previous_task(adapted, str(output))
after = {key: value.detach().clone() for key, value in peft_utils.get_peft_model_state_dict(adapted).items()}
assert before.keys() == after.keys()
assert all(torch.equal(before[key], value) for key, value in after.items())

from mcitlib.backbones.llava.trainer import LengthGroupedSampler, LLaVATrainer  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    @property
    def modality_lengths(self):
        return [3, 4, 5, 6]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        ids = torch.tensor([1, index + 2, 7])
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


args = moe_train.TrainingArguments(
    output_dir=str(output / "trainer"), no_cuda=True, max_steps=2, per_device_train_batch_size=1,
    gradient_accumulation_steps=1, group_by_modality_length=True, mm_projector_lr=2e-5,
    learning_rate=1e-3, logging_steps=1, save_strategy="no", report_to="none",
    disable_tqdm=True, dataloader_num_workers=0,
)
assert args.is_SAT is False
trainer = LLaVATrainer(model=adapted, args=args, train_dataset=TinyDataset())
trainer.mcit_method = captured["method"]
assert isinstance(trainer._get_train_sampler(), LengthGroupedSampler)
batch = next(iter(trainer.get_train_dataloader()))
assert torch.isfinite(trainer.compute_loss(adapted, batch))
assert trainer.create_optimizer() is not None
result = trainer.train()
assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
after_train = peft_utils.get_peft_model_state_dict(adapted)
assert any(not torch.equal(after[key], value) for key, value in after_train.items())
print(f"fresh {method}: train entry, adapter construction, save and reload OK; trainer optimizer steps=2")
