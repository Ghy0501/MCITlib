#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU: SAME's exact counters under reduced precision, PEFT reload and recomputation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from peft import PeftModel
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig, LlamaForCausalLM

from mcitlib.methods.same import SAMEMethod
from mcitlib.tuners._peft_registry import install_method_adapter
from mcitlib.tuners.variants.same import SAMELinear


def tiny():
    return LlamaForCausalLM(LlamaConfig(vocab_size=40,hidden_size=8,intermediate_size=16,
                                        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=32))


def method(task=0):
    return SAMEMethod({'r':4,'lora_alpha':8,'expert_num':2,'cur_task':task,'lora_dropout':0.0})


def state_values(model):
    return [{name:value.detach().clone() for name,value in layer.state.items()}
            for layer in model.modules() if isinstance(layer,SAMELinear)]


class SameCounterPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        install_method_adapter('same')

    def test_real_bfloat16_forward_1000_and_qr_cadence(self):
        for dtype in (torch.bfloat16,):
            with self.subTest(dtype=str(dtype)):
                torch.manual_seed(8)
                layer=SAMELinear('default',8,12,r=4,lora_alpha=8,expert_num=2,lora_dropout=0.0)
                with torch.no_grad():
                    for name in ('u','s','u_prev','s_prev','util','importance','mask'):
                        layer.state[name].fill_(0.1234567)
                old={name:layer.state[name].detach().clone() for name in ('u','s','u_prev','s_prev','util','importance','mask')}
                layer.to(dtype=dtype)
                self.assertEqual(layer.state['step'].dtype,torch.int64)
                self.assertEqual(layer.state['count'].dtype,torch.int64)
                for name,before in old.items():
                    self.assertEqual(layer.state[name].dtype,dtype)
                    self.assertTrue(torch.equal(layer.state[name],before.to(dtype)))
                x=torch.randn(2,4,8,dtype=dtype)
                calls=[]
                original=torch.linalg.qr
                def observe(*args,**kwargs):
                    calls.append(int(layer.state['step'].item()))
                    return original(*args,**kwargs)
                with torch.no_grad(),mock.patch('torch.linalg.qr',side_effect=observe):
                    for _ in range(1000):
                        layer(x)
                self.assertEqual(int(layer.state['step']),1000)
                self.assertEqual(int(layer.state['count']),8000)
                self.assertEqual(calls,list(range(0,1000,20)))
                layer.snapshot()
                prev_u,prev_s=layer.state['u_prev'].clone(),layer.state['s_prev'].clone()
                importance=layer.state['importance'].clone()
                layer.next_task()
                self.assertEqual(int(layer.state['step']),0)
                self.assertEqual(int(layer.state['count']),0)
                self.assertTrue(torch.equal(layer.state['u_prev'],prev_u))
                self.assertTrue(torch.equal(layer.state['s_prev'],prev_s))
                self.assertTrue(torch.equal(layer.state['importance'],importance))

    def test_half_conversion_and_state_roundtrip_preserve_exact_counters(self):
        layer=SAMELinear('default',8,12,r=4,lora_alpha=8,expert_num=2,lora_dropout=0.0)
        with torch.no_grad():
            layer.state['step'].fill_(1000)
            layer.state['count'].fill_(8000)
            for name in ('u','s','u_prev','s_prev','util','importance','mask'):
                layer.state[name].fill_(0.1234567)
        old={name:layer.state[name].detach().clone() for name in ('u','s','u_prev','s_prev','util','importance','mask')}
        layer.half()
        self.assertEqual(layer.state['step'].dtype,torch.int64)
        self.assertEqual(layer.state['count'].dtype,torch.int64)
        self.assertEqual(int(layer.state['step']),1000)
        self.assertEqual(int(layer.state['count']),8000)
        for name,before in old.items():
            self.assertEqual(layer.state[name].dtype,torch.float16)
            self.assertTrue(torch.equal(layer.state[name],before.half()))
        with tempfile.TemporaryDirectory(prefix='same-counter-half-') as directory:
            path=Path(directory)/'state.pt'
            torch.save(layer.state_dict(),path)
            restored=SAMELinear('default',8,12,r=4,lora_alpha=8,expert_num=2,lora_dropout=0.0).half()
            restored.load_state_dict(torch.load(path,map_location='cpu'))
            for name,value in layer.state.items():
                self.assertTrue(torch.equal(value,restored.state[name]),name)

    def test_peft_same_task_roundtrip_261_to_460_and_task_boundary_reset(self):
        torch.manual_seed(83)
        base=tiny().to(torch.bfloat16)
        base_weights=copy.deepcopy(base.state_dict())
        original=method().build_trainable(base,None).to(torch.bfloat16).train()
        ids=torch.tensor([[1,2,3,4],[4,3,2,1]])
        with torch.no_grad():
            for _ in range(261):
                original(input_ids=ids)
        with tempfile.TemporaryDirectory(prefix='same-counter-peft-') as directory:
            path=Path(directory)
            # Native LLaVA builds adapter state from deduplicated named_parameters(),
            # rather than passing model.state_dict() with duplicate aliases.
            native={name:value.detach().clone() for name,value in original.named_parameters()
                    if 'lora_' in name}
            original.save_pretrained(path,state_dict=native,safe_serialization=False)
            config=json.loads((path/'adapter_config.json').read_text())
            self.assertEqual(config['counter_schema'],'int64-v1')
            saved=torch.load(path/'adapter_model.bin',map_location='cpu')
            counters={k:v for k,v in saved.items() if k.endswith(('lora_state.step','lora_state.count'))}
            self.assertEqual(len(counters),6)
            self.assertTrue(all(v.dtype==torch.int64 for v in counters.values()))
            at_save=state_values(original)
            rng=torch.get_rng_state().clone()
            with torch.no_grad():
                for _ in range(199):
                    original(input_ids=ids)
            expected=state_values(original)
            base2=tiny().to(torch.bfloat16)
            base2.load_state_dict(base_weights)
            reloaded=PeftModel.from_pretrained(base2,path,is_trainable=True).to(torch.bfloat16).train()
            for before,after in zip(at_save,state_values(reloaded)):
                for name in before:
                    self.assertTrue(torch.equal(before[name],after[name]),'immediate reload: '+name)
            torch.set_rng_state(rng)
            with torch.no_grad():
                for _ in range(199):
                    reloaded(input_ids=ids)
            actual=state_values(reloaded)
            self.assertEqual(len(actual),3)
            for before,after in zip(expected,actual):
                for name in before:
                    self.assertTrue(torch.equal(before[name],after[name]),name)
                self.assertEqual(int(after['step']),460)
                self.assertEqual(int(after['count']),3680)
            method().finish_task(reloaded)
            reloaded.save_pretrained(path/'boundary',safe_serialization=False)
            base3=tiny().to(torch.bfloat16)
            base3.load_state_dict(base_weights)
            continued=PeftModel.from_pretrained(base3,path/'boundary',is_trainable=True).to(torch.bfloat16)
            saved_state=state_values(continued)
            method(task=1).load_previous(continued,path/'boundary')
            for before,after in zip(saved_state,state_values(continued)):
                self.assertEqual(int(after['step']),0)
                self.assertEqual(int(after['count']),0)
                self.assertTrue(torch.equal(before['u_prev'],after['u_prev']))
                self.assertTrue(torch.equal(before['s_prev'],after['s_prev']))

    def test_peft_eval_and_training_reject_legacy_float_counters_before_cast(self):
        from peft.utils import set_peft_model_state_dict
        torch.manual_seed(7)
        model=method().build_trainable(tiny(),None).to(torch.bfloat16)
        with tempfile.TemporaryDirectory(prefix='same-counter-old-') as directory:
            path=Path(directory)
            model.save_pretrained(path,state_dict={name:value.detach().clone()
                                  for name,value in model.named_parameters() if 'lora_' in name},
                                  safe_serialization=False)
            weights=torch.load(path/'adapter_model.bin',map_location='cpu')
            for name in weights:
                if name.endswith('lora_state.step'):
                    weights[name]=torch.tensor(256,dtype=torch.bfloat16)
                if name.endswith('lora_state.count'):
                    weights[name]=torch.tensor(4000,dtype=torch.bfloat16)
            torch.save(weights,path/'adapter_model.bin')
            with self.assertRaisesRegex(RuntimeError,'legacy floating.*saturated'):
                PeftModel.from_pretrained(tiny(),path)
            target=method(task=1).build_trainable(tiny(),None).to(torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError,'legacy floating.*saturated'):
                set_peft_model_state_dict(target,weights)
            self.assertTrue(all(int(x['step'])==0 and int(x['count'])==0 for x in state_values(target)))

    def test_peft_rejects_adapter_without_any_same_state_but_allows_base_partial_load(self):
        from peft.utils import set_peft_model_state_dict
        model=method().build_trainable(tiny(),None)
        with tempfile.TemporaryDirectory(prefix='same-counter-missing-') as directory:
            path=Path(directory)
            model.save_pretrained(path,safe_serialization=False)
            weights=torch.load(path/'adapter_model.bin',map_location='cpu')
            bad={k:v for k,v in weights.items() if '.lora_state.' not in k}
            torch.save(bad,path/'adapter_model.bin')
            with self.assertRaisesRegex(RuntimeError,'SAME adapter state is incomplete'):
                PeftModel.from_pretrained(tiny(),path)
            target=method(task=1).build_trainable(tiny(),None)
            with self.assertRaisesRegex(RuntimeError,'SAME adapter state is incomplete'):
                set_peft_model_state_dict(target,bad)
            # The actual LLaVA continuation path first performs a separate
            # strict=False load of non-LoRA weights. That remains legal.
            target.load_state_dict({'base_model.model.lm_head.weight':
                                    target.base_model.model.lm_head.weight.detach().clone()},strict=False)

    def test_peft_rejects_missing_layer_and_empty_adapter(self):
        from peft.utils import set_peft_model_state_dict
        model=method().build_trainable(tiny(),None)
        with tempfile.TemporaryDirectory(prefix='same-counter-missing-layer-') as directory:
            path=Path(directory)
            model.save_pretrained(path,safe_serialization=False)
            weights=torch.load(path/'adapter_model.bin',map_location='cpu')
            # Delete all keys of one whole SAME projection, including A/B,
            # router and state.  The other projections remain intact.
            bad_layer={k:v for k,v in weights.items() if '.gate_proj.' not in k}
            self.assertTrue(bad_layer)
            for bad in (bad_layer,{}):
                with self.subTest(remaining=len(bad)):
                    torch.save(bad,path/'adapter_model.bin')
                    with self.assertRaisesRegex(RuntimeError,'SAME adapter state is incomplete'):
                        PeftModel.from_pretrained(tiny(),path)
                    target=method(task=1).build_trainable(tiny(),None)
                    with self.assertRaisesRegex(RuntimeError,'SAME adapter state is incomplete'):
                        set_peft_model_state_dict(target,bad)

    def test_nonreentrant_backward_observes_once(self):
        torch.manual_seed(51)
        layer=SAMELinear('default',8,12,r=4,lora_alpha=8,expert_num=2,lora_dropout=0.0).to(torch.bfloat16).train()
        x=torch.randn(2,4,8,dtype=torch.bfloat16,requires_grad=True)
        calls=[]
        original=torch.linalg.qr
        def observe(*args,**kwargs):
            calls.append(int(layer.state['step']))
            return original(*args,**kwargs)
        with mock.patch('torch.linalg.qr',side_effect=observe):
            checkpoint(layer,x,use_reentrant=False).float().sum().backward()
        self.assertEqual(int(layer.state['step']),1)
        self.assertEqual(int(layer.state['count']),8)
        self.assertEqual(calls,[0])
        self.assertIsNotNone(x.grad)


if __name__=='__main__':
    unittest.main(verbosity=2)
