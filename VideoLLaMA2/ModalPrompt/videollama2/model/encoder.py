import os

import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel,   CLIPImageProcessor,   CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig,
)
from transformers import CLIPTextModel, CLIPTextConfig


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, load_pretrained=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = "eager"

        if not load_pretrained:
            self.vision_tower = CLIPVisionModel(config=config)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):   # images: [b*t, 3, 336, 336]
        if type(images) is list:
            image_features = []
            clip_image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                clip_image_feature = image_forward_out.pooler_output.to(images.dtype)
                image_features.append(image_feature)
                clip_image_features.append(clip_image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            clip_image_features = image_forward_outs.pooler_output.to(images.dtype)

        return clip_image_features, image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size

class CLIPTextTower(nn.Module):
    def __init__(self, text_tower, args, delay_load=False):
        super().__init__()
        
        self.is_loaded = False

        self.text_tower_name = text_tower
        self.select_layer = args.mm_text_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_tower_name)

    def load_model(self):
        self.text_tower = CLIPTextModel.from_pretrained(self.text_tower_name)
        self.text_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, text_inputs, return_hidden_states=False):
        # if type(texts_input_ids) is list:
        #     text_features = []
        #     for text_input_ids in texts_input_ids:
        #         text_forward_out = self.text_tower(text_input_ids.to(self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
        #         text_feature = self.feature_select(text_forward_out).to(text_input_ids.dtype)
        #         text_features.append(text_feature)
        # else:
        text_forward_outs = self.text_tower(**(text_inputs.to(self.device)))
        if return_hidden_states:
            text_hidden_features = text_forward_outs.last_hidden_state.to(self.dtype)
            text_features = text_forward_outs.pooler_output.to(self.dtype)

            return [text_hidden_features, text_features]
        else:
            text_features = text_forward_outs.pooler_output.to(self.dtype)
            return text_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.text_tower.dtype

    @property
    def device(self):
        return self.text_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.text_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size


class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, load_pretrained=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = 'eager'

        if not load_pretrained:
            self.vision_tower = SiglipVisionModel(config=config)
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if  'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    return vision_tower

def build_text_tower(text_tower_cfg, **kwargs):
    text_tower = getattr(text_tower_cfg, 'mm_text_tower', getattr(text_tower_cfg, 'text_tower', None))
    is_absolute_path_exists = os.path.exists(text_tower)
    if is_absolute_path_exists or text_tower.startswith("openai") or text_tower.startswith("laion"):
        return CLIPTextTower(text_tower, args=text_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {text_tower}')
