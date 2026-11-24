# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise SEFE's actual newer-snapshot vision initialization on CPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("deepspeed")
if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 37):
    pytest.skip("SEFE requires its transformers 4.37 environment", allow_module_level=True)

from mcitlib.backbones.llava.overrides import import_llava  # noqa: E402

import_llava()
from mcitlib.backbones.llava import standard_train  # noqa: E402
from mcitlib.methods._sefe.llava_overrides import llava_arch  # noqa: E402
from mcitlib.methods._sefe.llava_overrides.language_model.llava_llama import (  # noqa: E402
    LlavaConfig,
    LlavaLlamaForCausalLM,
)


class VisionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 6
        self.loaded = 0

    def load_model(self):
        self.loaded += 1


class VisionHost(llava_arch.LlavaMetaModel, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(hidden_size=8)

    @property
    def dtype(self):
        return torch.float32


class StopAfterVision(Exception):
    pass


@pytest.mark.parametrize("merge_type", ["flat", "spatial_unpad"])
@pytest.mark.parametrize("fsdp", [[], ["full_shard"]])
def test_actual_sefe_vision_initialization_uses_argument_defaults(
    monkeypatch: pytest.MonkeyPatch, merge_type: str, fsdp: list[str],
) -> None:
    args = standard_train.ModelArguments(vision_tower="tiny-vision", mm_patch_merge_type=merge_type)
    host, tower = VisionHost(), VisionTower()
    monkeypatch.setattr(llava_arch, "build_vision_tower", lambda supplied: tower)
    monkeypatch.setattr(
        llava_arch, "build_vision_projector",
        lambda config: torch.nn.Linear(config.mm_hidden_size, config.hidden_size),
    )
    host.initialize_vision_modules(model_args=args, fsdp=fsdp)
    assert host.get_vision_tower() is tower
    assert host.config.mm_patch_merge_type == merge_type
    assert host.config.mm_hidden_size == 6
    assert host.mm_projector(torch.zeros(1, 6)).shape == (1, 8)
    assert hasattr(host, "image_newline") == (merge_type == "spatial_unpad")
    host.initialize_vision_modules(model_args=args, fsdp=fsdp)
    assert tower.loaded == 1


def test_sefe_standard_entry_reaches_real_vision_initializer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    assert standard_train.ModelArguments().mm_patch_merge_type == "flat"
    args = standard_train.ModelArguments(model_name_or_path="tiny-model", vision_tower="tiny-vision", version="v0.5")
    training_args = SimpleNamespace(
        local_rank=0, fp16=False, bf16=False, bits=16, gradient_checkpointing=False,
        lora_enable=False, pt_enable=False, cache_dir=str(tmp_path), model_max_length=32,
        regularization_info_path=None, fsdp=[],
    )
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (args, SimpleNamespace(), training_args))
    monkeypatch.setattr(standard_train.transformers, "HfArgumentParser", lambda *_: parser)
    host, tower = VisionHost(), VisionTower()
    monkeypatch.setattr(llava_arch, "build_vision_tower", lambda supplied: tower)
    monkeypatch.setattr(
        llava_arch, "build_vision_projector",
        lambda config: torch.nn.Linear(config.mm_hidden_size, config.hidden_size),
    )

    def stop_after_vision():
        assert host.get_vision_tower() is tower
        assert host.config.mm_patch_merge_type == "flat"
        raise StopAfterVision

    fake_model = SimpleNamespace(config=SimpleNamespace(), get_model=lambda: host, get_vision_tower=stop_after_vision)
    monkeypatch.setattr(standard_train.LlavaLlamaForCausalLM, "from_pretrained", lambda *_, **__: fake_model)
    monkeypatch.setattr(
        standard_train.transformers.AutoTokenizer, "from_pretrained",
        lambda *_, **__: SimpleNamespace(unk_token="<unk>"),
    )
    monkeypatch.setenv("MCITLIB_METHOD", "sefe")
    monkeypatch.delenv("MCITLIB_MODEL_OVERRIDE", raising=False)
    with pytest.raises(StopAfterVision):
        standard_train.train()


def test_sefe_initialized_model_reaches_optimizer_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from peft import LoraConfig, get_peft_model

    config = LlavaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
    )
    model = LlavaLlamaForCausalLM(config)
    monkeypatch.setattr(llava_arch, "build_vision_tower", lambda supplied: VisionTower())
    monkeypatch.setattr(
        llava_arch, "build_vision_projector",
        lambda values: torch.nn.Linear(values.mm_hidden_size, values.hidden_size),
    )
    model.get_model().initialize_vision_modules(standard_train.ModelArguments(vision_tower="tiny-vision"), fsdp=[])
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.SGD([parameter for _, parameter in trainable], lr=0.01)
    ids = torch.tensor([[1, 2, 3]])
    output = model(input_ids=ids, labels=ids)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    assert any(not torch.equal(initial[name], parameter.detach()) for name, parameter in trainable)
