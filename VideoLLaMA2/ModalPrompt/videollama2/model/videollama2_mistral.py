# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
sys.path.append('/your_path/MCITlib_v3/VideoLLaMA2/ModalPrompt')
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM
from copy import deepcopy


class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama2_mistral"


class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)


class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)
        self.model = Videollama2MistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.hidden_size = config.hidden_size
        self.training = False

    def set_cur_task(self, cur_task):
        '''
        set cur task and cur task prompt transform for traning
        '''
        assert cur_task>=1, 'Task order start from 1.'
        self.cur_task = cur_task

        for name, param in self.prompt_transform.named_parameters():
            param.requires_grad = True if int(name.split('.')[0])==(cur_task - 1) else False

    def set_proto_input_ids(self):
        '''
        set prompt ids: [[32000,...,32009],[32010,...32019],...]
        '''
        proto_ids_list = []
        
        for i in range(self.num_tasks):
            # 生成 proto_ids
            proto_ids = self.tokenizer(
                ''.join(self.prefix_tokens_list[self.mapping_list[str(i+1)]]), 
                return_tensors='pt'
            ).input_ids.squeeze()[1:].to(self.device)
            
            # 打印当前任务的 proto_ids 形状和内容
            print(f"Task {i+1}: proto_ids shape = {proto_ids.shape}, proto_ids = {proto_ids.tolist()}")
            
            proto_ids_list.append(proto_ids)
        
        # 打印完整的 proto_ids_list 的信息
        print(f"Total proto_ids_list length: {len(proto_ids_list)}")
        for idx, tensor in enumerate(proto_ids_list):
            print(f"proto_ids_list[{idx}]: shape = {tensor.shape}")

        # 堆叠张量
        self.proto_ids_list = torch.stack(proto_ids_list, dim=0)

    def get_proto_input_ids(self, cur_task=1, device = None):
        assert cur_task>=1, 'do not need to get proto inputs'
        
        return self.proto_ids_list[:cur_task].to(device)

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def reset_vocab_size(self):
        '''
        set vocab size to normal size and embed token mask for continual prompt (before changing of vocab_size)
        '''
        self.embed_tokens_mask_1d = torch.zeros(self.config.vocab_size)
        prompt_prefix = (self.num_tasks - self.cur_task + 1) * self.prefix_len 
        prompt_suffix = (self.num_tasks - self.cur_task ) * self.prefix_len 
        self.embed_tokens_mask_1d[-prompt_prefix:-prompt_suffix] = 1

        self.config.vocab_size = self.config.vocab_size - self.prefix_len * self.num_tasks

    def set_previou_transform_for_save(self, cur_task):
        '''
        set all previous task to requires_grad = True for the convenience of get_peft_state_non_lora_maybe_zero_3()
        '''
        for name, param in self.prompt_transform.named_parameters():
            param.requires_grad = True if int(name.split('.')[0])<=(cur_task - 1) else False

    def set_origin_embed_tokens(self):
        '''
        set original embed_tokens for backward resetting
        '''
        self.origin_embed_tokens_weights = self.model.embed_tokens.weight.detach()

    def set_num_task(self, num_tasks):
        self.num_tasks = num_tasks

    def set_comtinual_eval(self, tokenizer, clip_tokenizer, prefix_len, cur_task, num_tasks):
        '''
        set continual attributes for evaluation
        '''
        self.num_tasks = num_tasks
        self.set_comtinual_prompts_tokenizer(tokenizer, prefix_len)
        self.set_clip_tokenizer(clip_tokenizer)
        self.set_proto_input_ids()
        self.cur_task = cur_task

    def set_comtinual_prompts_tokenizer(self, tokenizer, prefix_len = 10, same_prompt=None):
        # adding special prefix tokens (CHANGE for new tasks)
        assert self.num_tasks!=None, 'num tasks should exist.'
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.tasks = ['task'+str(i+1) for i in range(self.num_tasks)]  # shoule be ['task1','task2','task3'...]
        self.mapping_list = {str(i+1):str(self.tasks[i]) for i in range(self.num_tasks)}  # shoule be ['1':'task1','2':'task2','3':'task3',...]

        if prefix_len > 0:
            self.prefix_tokens_list = {}

            if same_prompt: # assume we have just 1 task (i.e. 1 prompt for each task)
                self.prefix_tokens_list[0] = self.add_prompt_tokens(prefix_len, prompt_name='PRE0_')
                # self.vocab_size = self.vocab_size + prefix_len 
                self.prompt_transform = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                    nn.SiLU(),
                                                    nn.Linear(self.hidden_size, 768),
                                        )
            else:
                for i in range(self.num_tasks):
                    # new prefix for each task
                    # Task 1 = PRE1_1, ... PRE1_10 ; Task 2 = PRE2_1, ... PRE2_10
                    self.prefix_tokens_list[self.tasks[i]], self.tokenizer = self.add_prompt_tokens(prefix_len, prompt_name='PRE'+str(i+1)+'_')
                self.prompt_transform = nn.ModuleList(
                                        nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                    nn.SiLU(),
                                                    nn.Linear(self.hidden_size, 768),
                                        ) for i in range(self.num_tasks)
                                        )
                # self.vocab_size = self.vocab_size + prefix_len* self.num_tasks
        else:
            self.prefix_tokens_list = {self.tasks[i]: [] for i in range(self.num_tasks)} # empty prompt for each task

        self.prompt_transform.to(device = self.device, dtype = torch.float16)
        return self.tokenizer

    def add_prompt_tokens(self, prefix_len, prompt_name='PRE'):
        tokenizer = self.tokenizer
        model = self.model
        # model.embed_tokens.weight.requires_grad = True

        # tokens_list - ['[PRE1]', '[PRE2]', '[PRE3]']
        tokens_list = ['['+ prompt_name + str(i) + ']' for i in np.arange(1, prefix_len+1)]
        special_tokens_dict = {'additional_special_tokens': tokens_list}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        # tokenizer.tokenize('[PRE1_1]') # ['[PRE1_1]']
        # model.embed_tokens.weight.requires_grad = False

        with torch.no_grad():
            for i in range(len(tokens_list)):
                random_init_j = np.random.randint(self.vocab_size)
                random_init_w = deepcopy(model.embed_tokens.weight[random_init_j].detach())
                model.embed_tokens.weight[self.vocab_size+(int(prompt_name[-2])-1)*self.prefix_len + i] = random_init_w
        model.embed_tokens.weight.requires_grad = True
        return tokens_list, tokenizer
    
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs.labels = labels

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videollama2_mistral", Videollama2MistralConfig)
AutoModelForCausalLM.register(Videollama2MistralConfig, Videollama2MistralForCausalLM)
