# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU regression for SEFE's automatically mapped InternViT tower."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

# Transformers may map the tower to CUDA even in a CPU regression.
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise unittest.SkipTest('set CUDA_VISIBLE_DEVICES to an empty string for this CPU-only test')

import torch
import transformers

from mcitlib.backbones.internvl.intern_vit.configuration_intern_vit import InternVisionConfig
from mcitlib.backbones.internvl.intern_vit.modeling_intern_vit import InternVisionModel
from mcitlib.backbones.internvl.vision_tower import InternVisionTower
from mcitlib.backbones.llava.overrides import allow_t437_llava_registration


@unittest.skipUnless(
    tuple(int(part) for part in transformers.__version__.split('.')[:2]) >= (4, 36),
    'SEFE private builder requires its Transformers 4.37 environment',
)
class TestInternVitAutoDeviceMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with allow_t437_llava_registration():
            from mcitlib.methods._sefe.llava_overrides import builder

        cls.builder = builder

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'vision'
        config = InternVisionConfig(image_size=16, patch_size=8, hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=2, num_attention_heads=4, use_flash_attn=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            # The real InternViT checkpoint is stored in bfloat16.
            InternVisionModel(config).eval().to(torch.bfloat16).save_pretrained(self.path)

    def tower(self):
        return InternVisionTower(self.path, SimpleNamespace(mm_vision_select_layer=-2,
                                  mm_vision_select_feature='patch'), delay_load=True)

    def build(self, tower, *, device_map):
        model = SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=False, mm_use_im_patch_token=False),
                                get_vision_tower=lambda: tower, resize_token_embeddings=Mock())
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 16
        with patch.object(self.builder.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
             patch.object(self.builder.LlavaLlamaForCausalLM, 'from_pretrained', return_value=model):
            result = self.builder.load_pretrained_model('intern_model', None, 'intern_model',
                                                        device_map=device_map, device='cpu')
        self.assertIs(result[1], model)
        model.resize_token_embeddings.assert_called_once_with(16)
        return result

    def test_original_missing_no_split_metadata_reproduces_failure(self):
        with patch.object(InternVisionModel, '_no_split_modules', None):
            with self.assertRaisesRegex(ValueError, 'does not support `device_map='):
                InternVisionModel.from_pretrained(self.path, device_map='auto', local_files_only=True)

    def test_sefe_auto_loads_real_tower_in_half_precision_and_preserves_weights(self):
        tower = self.tower()
        with patch.object(InternVisionModel, 'from_pretrained', wraps=InternVisionModel.from_pretrained) as load:
            self.build(tower, device_map='auto')
        self.assertEqual(load.call_args.kwargs['device_map'], 'auto')
        self.assertEqual(load.call_args.kwargs['torch_dtype'], torch.float16)
        self.assertEqual(tower.vision_tower.hf_device_map, {'': 'cpu'})
        self.assertEqual(tower.dtype, torch.float16)
        self.assertIn('InternVisionEncoderLayer', tower.vision_tower._get_no_split_modules('auto'))
        self.assertTrue(all(not parameter.requires_grad for parameter in tower.parameters()))

        reference = InternVisionModel.from_pretrained(self.path, local_files_only=True).eval().half().float()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(tower.vision_tower.state_dict()[name].float(), expected, rtol=0, atol=0)
        half_bytes = sum(parameter.numel() * parameter.element_size() for parameter in tower.parameters())
        float_bytes = sum(parameter.numel() * parameter.element_size() for parameter in reference.parameters())
        self.assertEqual(float_bytes, 2 * half_bytes)
        # Torch 2.0 cannot execute CPU fp16 matrix multiplication. Compare the
        # fp16-rounded weights in fp32 to check the real selected image features.
        tower.vision_tower.float().eval()
        pixels = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) / 255
        with torch.no_grad():
            actual = tower(pixels)
            expected = reference(pixels, output_hidden_states=True).hidden_states[-2][:, 1:]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_explicit_cpu_device_map_keeps_existing_builder_behavior(self):
        tower = self.tower()
        with patch.object(InternVisionModel, 'from_pretrained', wraps=InternVisionModel.from_pretrained) as load:
            self.build(tower, device_map='cpu')
        self.assertEqual(load.call_args.kwargs['device_map'], 'cpu')
        self.assertNotIn('torch_dtype', load.call_args.kwargs)
        self.assertEqual(tower.device.type, 'cpu')
        self.assertEqual(tower.dtype, torch.float16)

    def test_other_tower_keeps_original_auto_load_arguments(self):
        tower = SimpleNamespace(is_loaded=False, vision_tower_name='openai/clip-vit',
                                load_model=Mock(), to=Mock(), image_processor=object())
        self.build(tower, device_map='auto')
        tower.load_model.assert_called_once_with(device_map='auto')
        tower.to.assert_not_called()

    def test_generic_tower_auto_does_not_force_a_dtype(self):
        tower = self.tower()
        tower.load_model(device_map='auto')
        self.assertEqual(tower.device.type, 'cpu')
        self.assertEqual(tower.dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
