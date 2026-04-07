import os
from .clip_encoder import CLIPVisionTower, CLIPTextTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)

def build_text_tower(text_tower_cfg, **kwargs):
    text_tower = getattr(text_tower_cfg, 'mm_text_tower', getattr(text_tower_cfg, 'text_tower', None))
    is_absolute_path_exists = os.path.exists(text_tower)
    if is_absolute_path_exists or text_tower.startswith("openai") or text_tower.startswith("laion"):
        return CLIPTextTower(text_tower, args=text_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {text_tower}')

# ============================================================================================================
