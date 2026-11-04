# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Compare shared CLIP capability with v3 using in-memory toy backends."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from torch import nn


class ToyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10)

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self):
        return self.weight.device

    def forward(self, input_ids, **kwargs):
        selected = self.weight[input_ids]
        return SimpleNamespace(last_hidden_state=selected, pooler_output=selected.mean(dim=1))


class ToyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10)

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self):
        return self.weight.device

    def forward(self, images, output_hidden_states=True):
        pooled = images.mean(dim=1) @ self.weight
        return SimpleNamespace(image_embeds=pooled, hidden_states=[pooled, pooled * 2, pooled * 3])


class TextInputs:
    def to(self, device):
        return {"input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]], device=device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.implementation == "v3":
        from llava.model.multimodal_encoder import clip_encoder as module
    else:
        from mcitlib.backbones.llava import text_tower as module
    module.CLIPTextModel = SimpleNamespace(from_pretrained=lambda name: ToyText())
    module.CLIPVisionModelWithProjection = SimpleNamespace(from_pretrained=lambda name: ToyVision())
    module.CLIPImageProcessor = SimpleNamespace(from_pretrained=lambda name: object())
    text = module.CLIPTextTower("unused", SimpleNamespace(mm_text_select_layer=-1))
    pooled = text(TextInputs())
    both = text(TextInputs(), return_hidden_states=True)
    vision = module.CLIPVisionTower("unused", SimpleNamespace(
        mm_vision_select_layer=-2, mm_vision_select_feature="patch"
    ))
    clip, selected = vision(torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 10)
    torch.save({
        "text_pooler": pooled, "text_hidden": both[0], "text_both_pooler": both[1],
        "clip": clip, "selected": selected,
        "text_select_layer": text.select_layer, "vision_select_layer": vision.select_layer,
    }, args.out)


if __name__ == "__main__":
    main()
