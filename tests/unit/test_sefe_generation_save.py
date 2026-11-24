# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Regression tests for saving SEFE models with inherited generation settings."""
from __future__ import annotations

import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from mcitlib.backbones.llava.overrides import allow_t437_llava_registration
from mcitlib.methods._sefe.merge import merge_lora


@unittest.skipUnless(
    tuple(int(part) for part in transformers.__version__.split(".")[:2]) >= (4, 37),
    "SEFE merge requires its Transformers 4.37 environment",
)
class TestSefeGenerationSave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with allow_t437_llava_registration():
            from mcitlib.methods._sefe.llava_overrides import builder

        cls.builder = builder

    def make_model(self, do_sample):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        with torch.random.fork_rng():
            torch.manual_seed(13)
            model = LlamaForCausalLM(config).eval()
        generation = model.generation_config
        generation.do_sample = do_sample
        generation.temperature = 0.9
        generation.top_p = 0.6
        generation.max_new_tokens = 4
        generation.repetition_penalty = 1.2
        generation.no_repeat_ngram_size = 2
        return model

    def merge(self, model, destination):
        tokenizer = Mock()
        with patch.object(self.builder, "load_pretrained_model", return_value=(tokenizer, model, None, 0)) as load:
            merge_lora(SimpleNamespace(
                model_path="adapters/checkpoint-1", model_base="base", save_model_path=str(destination)
            ))
        load.assert_called_once_with("adapters/checkpoint-1", "base", "adapters_checkpoint-1", device_map="cpu")
        tokenizer.save_pretrained.assert_called_once_with(str(destination))

    def test_greedy_model_saves_and_reloads_without_changing_outputs_or_weights(self):
        model = self.make_model(do_sample=False)
        initial_config = copy.deepcopy(model.generation_config.to_dict())
        weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        prompt = torch.tensor([[1, 5, 7]])
        # The inherited sampling-only settings warn but are ignored for greedy generation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            before = model.generate(prompt)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError) as failure:
                model.save_pretrained(root / "before-fix")
            self.assertIn("temperature", str(failure.exception))
            self.assertIn("top_p", str(failure.exception))

            destination = root / "merged"
            self.merge(model, destination)
            expected_config = {**initial_config, "temperature": 1.0, "top_p": 1.0}
            self.assertEqual(model.generation_config.to_dict(), expected_config)
            loaded = LlamaForCausalLM.from_pretrained(destination, local_files_only=True).eval()

            for name, value in weights.items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
            self.assertTrue(torch.equal(before, loaded.generate(prompt)))
            for name, value in expected_config.items():
                if name not in {"_from_model_config", "transformers_version"}:
                    self.assertEqual(getattr(loaded.generation_config, name), value, name)

    def test_sampling_configuration_is_preserved(self):
        model = self.make_model(do_sample=True)
        initial_config = copy.deepcopy(model.generation_config.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            self.merge(model, Path(tmp) / "merged")
            self.assertEqual(model.generation_config.to_dict(), initial_config)
            loaded = LlamaForCausalLM.from_pretrained(Path(tmp) / "merged", local_files_only=True)
            for name, value in initial_config.items():
                if name not in {"_from_model_config", "transformers_version"}:
                    self.assertEqual(getattr(loaded.generation_config, name), value, name)

    def test_other_invalid_generation_settings_still_fail_validation(self):
        model = self.make_model(do_sample=False)
        model.generation_config.early_stopping = True  # Requires beam search.
        tokenizer = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.builder, "load_pretrained_model", return_value=(tokenizer, model, None, 0)):
                with self.assertRaises(ValueError) as failure:
                    merge_lora(SimpleNamespace(
                        model_path="adapters/checkpoint-1", model_base="base", save_model_path=tmp
                    ))
        self.assertIn("early_stopping", str(failure.exception))
        self.assertTrue(model.generation_config.early_stopping)
        tokenizer.save_pretrained.assert_not_called()

    def test_model_without_generation_config_can_still_be_saved(self):
        self.merge(SimpleNamespace(save_pretrained=Mock()), "merged")


if __name__ == "__main__":
    unittest.main()
