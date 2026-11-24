# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""InternViT visual features for the LLaVA-formatted InternVL-Chat model."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .detect import is_intern_vit, vision_name
from .intern_vit.configuration_intern_vit import InternVisionConfig
from .intern_vit.modeling_intern_vit import InternVisionModel


class InternVisionTower(nn.Module):
    """Return InternViT patches, optionally with a separate CLIP guide vector."""

    def __init__(self, vision_tower, args, delay_load=False, *, with_guide=False):
        super().__init__()
        self.vision_tower_name = str(vision_tower)
        self.select_layer = getattr(args, 'mm_vision_select_layer', -4)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.with_guide = with_guide
        self.guide_tower_name = (getattr(args, 'mm_text_tower', None) or getattr(args, 'text_tower', None)
                                 or os.environ.get('MCITLIB_INTERNVL_CLIP_TOWER'))
        if self.with_guide and not self.guide_tower_name:
            raise ValueError('InternVL guide methods require a separate CLIP tower path')
        self.is_loaded = False
        self.cfg_only = InternVisionConfig.from_pretrained(self.vision_tower_name)
        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None, *, torch_dtype=None):
        if self.is_loaded:
            return
        # The InternVL method implementations use 336px input for the 224px
        # pretrained InternViT; the model interpolates its position embeddings.
        size = 448 if '448' in self.vision_tower_name else 336
        self.image_processor = CLIPImageProcessor(
            crop_size=size, do_center_crop=True, do_normalize=True, do_resize=True,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size=size,
        )
        kwargs = {'device_map': device_map} if device_map is not None else {}
        if torch_dtype is not None:
            kwargs['torch_dtype'] = torch_dtype
        self.vision_tower = InternVisionModel.from_pretrained(self.vision_tower_name, config=self.cfg_only, **kwargs)
        self.vision_tower.requires_grad_(False)
        if self.with_guide:
            self.clip_vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.guide_tower_name, **kwargs)
            self.clip_vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, outputs):
        features = outputs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            return features[:, 1:]
        if self.select_feature == 'cls_patch':
            return features
        raise ValueError(f'unexpected vision feature selection: {self.select_feature}')

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            raise RuntimeError('InternViT tower has not been loaded')
        if isinstance(images, list):
            result = [self.forward(image.unsqueeze(0)) for image in images]
            if self.with_guide:
                return torch.cat([item[0] for item in result], dim=0), [item[1] for item in result]
            return [item for item in result]
        pixels = images.to(device=self.device, dtype=self.dtype)
        outputs = self.vision_tower(pixels, output_hidden_states=True)
        features = self.feature_select(outputs).to(images.dtype)
        if not self.with_guide:
            return features
        guide = self.clip_vision_tower(pixels, output_hidden_states=True).image_embeds.to(images.dtype)
        return guide, features

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def num_patches(self):
        size = 448 if '448' in self.vision_tower_name else 336
        return (size // self.config.patch_size) ** 2


def build_intern_vision_tower(config, *, with_guide=False, **kwargs):
    name = vision_name(config)
    if not is_intern_vit(name):
        raise ValueError(f'not an InternViT tower: {name!r}')
    if hasattr(config, 'mm_vision_tower'):
        config.mm_vision_tower = str(name)
    return InternVisionTower(name, config, with_guide=with_guide, **kwargs)
