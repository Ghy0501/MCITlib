import os
from .clip_encoder import CLIPVisionTower, CLIPTextTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_text_tower(text_tower_cfg, **kwargs):
    text_tower = getattr(text_tower_cfg, 'mm_text_tower', getattr(text_tower_cfg, 'text_tower', None))
    is_absolute_path_exists = os.path.exists(text_tower)
    if is_absolute_path_exists or text_tower.startswith("openai") or text_tower.startswith("laion"):
        return CLIPTextTower(text_tower, args=text_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {text_tower}')
