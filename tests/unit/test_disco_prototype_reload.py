# SPDX-License-Identifier: Apache-2.0
"""Real DISCO private-model initialization and prototype checkpoint regressions."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("deepspeed")

from mcitlib.backbones.llava.overrides import import_llava  # noqa: E402

import_llava()
from mcitlib.backbones.llava.moe_train import (  # noqa: E402
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    prepare_method_for_save,
)
from mcitlib.methods._disco.llava_overrides.language_model.llava_llama import (  # noqa: E402
    LlavaConfig,
    LlavaLlamaForCausalLM,
)

FIELDS = ("global_text_feature", "global_image_feature", "text_boundary", "image_boundary")


def make_base(path: Path, *, layers: int = 1, partial: bool = False) -> None:
    config = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    model = LlavaLlamaForCausalLM(config)
    path.mkdir()
    config.save_pretrained(path)
    state = {key: value for key, value in model.state_dict().items()
             if not key.startswith(tuple(name + "." for name in FIELDS))}
    if partial:
        state["global_text_feature.0"] = torch.full((768,), 3, dtype=torch.bfloat16)
        state["text_boundary.3"] = torch.full((1,), 7, dtype=torch.bfloat16)
    torch.save(state, path / "pytorch_model.bin")


def load_base(path: Path, *, low_memory: bool = True) -> LlavaLlamaForCausalLM:
    return LlavaLlamaForCausalLM.from_pretrained(
        path, low_cpu_mem_usage=low_memory, torch_dtype=torch.float16,
        device_map={"": "cpu"} if low_memory else None,
    )


@pytest.mark.parametrize("low_memory", [False, True])
def test_missing_defaults_preserve_loaded_prototypes(tmp_path: Path, low_memory: bool) -> None:
    make_base(tmp_path / "base", partial=True)
    model = load_base(tmp_path / "base", low_memory=low_memory)
    for name in FIELDS:
        for index, parameter in enumerate(getattr(model, name)):
            expected = 3 if (name, index) == ("global_text_feature", 0) else (
                7 if (name, index) == ("text_boundary", 3) else int("boundary" in name)
            )
            assert torch.equal(parameter, torch.full_like(parameter, expected)), (name, index)


def test_save_hook_preserves_disco_and_hide_statistics() -> None:
    from mcitlib.methods._hide.llava_overrides.language_model.llava_llama import (
        LlavaConfig as HideConfig,
    )
    from mcitlib.methods._hide.llava_overrides.language_model.llava_llama import (
        LlavaLlamaForCausalLM as HideModel,
    )

    cfg = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
               num_attention_heads=4, num_key_value_heads=4)
    for method, model, names, count in (
        ("disco", LlavaLlamaForCausalLM(LlavaConfig(**cfg)), FIELDS, 32),
        ("hide", HideModel(HideConfig(**cfg)), ("text_anchors", "image_anchors", "text_boundary", "image_boundary"), 40),
    ):
        model.requires_grad_(False)
        prepare_method_for_save(model, "olora")
        assert not any(parameter.requires_grad for parameter in model.parameters())
        prepare_method_for_save(model, method)
        state = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        assert len(state) == count
        assert all(sum(key.startswith(name + ".") for key in state) == count // len(names) for name in names)


def install_tiny_real_clip_towers(model) -> None:
    from transformers import (
        BatchEncoding,
        CLIPTextConfig,
        CLIPTextModel,
        CLIPVisionConfig,
        CLIPVisionModelWithProjection,
    )

    from mcitlib.backbones.llava.text_tower import CLIPTextTower, CLIPVisionTower

    vision = CLIPVisionTower.__new__(CLIPVisionTower)
    torch.nn.Module.__init__(vision)
    vision.vision_tower = CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
        image_size=8, patch_size=4, projection_dim=768,
    ))
    vision.select_layer = -1
    vision.select_feature = "patch"
    vision.is_loaded = True
    text = CLIPTextTower.__new__(CLIPTextTower)
    torch.nn.Module.__init__(text)
    text.text_tower = CLIPTextModel(CLIPTextConfig(
        vocab_size=32, hidden_size=768, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=12, max_position_embeddings=16,
    ))
    text.is_loaded = True
    model.get_model().vision_tower = vision
    model.get_model().text_tower = text
    model.get_model().mm_projector = torch.nn.Linear(16, 16)
    vision.requires_grad_(False)
    text.requires_grad_(False)
    model.get_model().mm_projector.requires_grad_(True)

    class LmTokenizer:
        pad_token_id = 0
        def batch_decode(self, inputs, **kwargs):
            return ["USER:\n<image> What is it? ASSISTANT:"] * len(inputs)

    def clip_tokenizer(inputs, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[1, 4, 2]] * len(inputs)),
                              "attention_mask": torch.ones((len(inputs), 3), dtype=torch.long)})

    model.set_tokenizer(LmTokenizer())
    model.set_clip_tokenizer(clip_tokenizer)


def multimodal(model) -> None:
    ids = torch.tensor([[1, -200, 4, 2]])
    images = torch.randn(1, 3, 8, 8)
    output = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images)
    assert output[4] is not None and torch.isfinite(output[4]).all()


def writer(root: Path) -> None:
    from mcitlib.methods.disco import DISCOMethod
    from mcitlib.tuners._peft_registry import install_method_adapter

    torch.manual_seed(123)
    make_base(root / "base", layers=32)
    raw = load_base(root / "base").float()
    assert all(torch.count_nonzero(p) == 0 for p in raw.global_text_feature)
    assert all(torch.equal(p, torch.ones_like(p)) for p in raw.text_boundary)
    install_method_adapter("disco")
    method = DISCOMethod({"r": 6, "lora_alpha": 12, "expert_num": 6,
                          "cur_task": 0, "target_modules": ["q_proj"]})
    model = method.build_trainable(raw, None)
    install_tiny_real_clip_towers(raw)
    raw.set_cur_task(0, 6)
    model.train()
    multimodal(raw)
    assert torch.isfinite(raw.global_text_feature[0]).all()
    assert torch.count_nonzero(raw.global_text_feature[0]) > 0
    assert torch.isfinite(raw.global_image_feature[0]).all()
    assert torch.count_nonzero(raw.global_image_feature[0]) > 0
    prepare_method_for_save(model, "disco")
    non_lora = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
    prototype = {key: value for key, value in non_lora.items() if any(name in key for name in FIELDS)}
    assert len(prototype) == 32
    adapter = root / "Task1_llava_lora"
    model.save_pretrained(adapter, state_dict=get_peft_state_maybe_zero_3(model.named_parameters(), "none"))
    torch.save(non_lora, adapter / "non_lora_trainables.bin")


def reader(root: Path) -> None:
    from peft import PeftModel
    from transformers import BatchEncoding

    from mcitlib.tuners._peft_registry import install_method_adapter

    torch.manual_seed(456)
    raw = load_base(root / "base")
    state = torch.load(root / "Task1_llava_lora/non_lora_trainables.bin", map_location="cpu")
    state = {(key[11:] if key.startswith("base_model.") else key): value for key, value in state.items()}
    if any(key.startswith("model.model.") for key in state):
        state = {(key[6:] if key.startswith("model.") else key): value for key, value in state.items()}
    raw.load_state_dict(state, strict=False)
    for name in FIELDS:
        for i, param in enumerate(getattr(raw, name)):
            assert torch.equal(param, state[f"{name}.{i}"].to(param.dtype))
    install_method_adapter("disco")
    model = PeftModel.from_pretrained(raw, root / "Task1_llava_lora").merge_and_unload()
    install_tiny_real_clip_towers(model)
    model.set_eval(6)
    model.eval()
    # The production builder requests float16, while DISCO's explicit prototype
    # tensors remain bfloat16. Obtain a real CLIP text feature and represent it
    # in the float16 dtype used by the production text tower.
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.global_text_feature)
    clip_input = BatchEncoding({"input_ids": torch.tensor([[1, 4, 2]]),
                                "attention_mask": torch.ones((1, 3), dtype=torch.long)})
    half_guide = model.get_text_tower()(clip_input).half()
    similarities = torch.nn.functional.cosine_similarity(torch.stack(list(model.global_text_feature)),
                                                          half_guide.mean(dim=0, keepdim=True), dim=1)
    weights = torch.softmax(similarities[:6] / 0.05, dim=0)
    assert torch.isfinite(similarities).all() and torch.isfinite(weights).all()
    # CPU LLaMA operators need float32; the GPU smoke must still validate FP16 logits.
    model.float()
    multimodal(model)
    with torch.inference_mode():
        ids = torch.tensor([[1, -200, 4, 2]])
        images = torch.randn(1, 3, 8, 8)
        logits = model(input_ids=ids, images=images, use_cache=False).logits
        generated = model.generate(input_ids=ids, images=images, do_sample=False, max_new_tokens=4,
                                   return_dict_in_generate=True, output_scores=True)
    assert torch.isfinite(logits).all()
    assert generated.sequences.shape[1] > ids.shape[1] and generated.scores
    assert all(torch.isfinite(score).all() for score in generated.scores)
    for layer in model.model.layers:
        weights = torch.tensor(layer.self_attn.q_proj.mask_signal)
        assert weights.numel() == 6 and torch.isfinite(weights).all()
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_fresh_process_private_clip_prototypes_save_reload_and_route(tmp_path: Path) -> None:
    for action in ("writer", "reader"):
        result = subprocess.run([sys.executable, __file__, action, str(tmp_path)], capture_output=True, text=True,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    {"writer": writer, "reader": reader}[sys.argv[1]](Path(sys.argv[2]))
