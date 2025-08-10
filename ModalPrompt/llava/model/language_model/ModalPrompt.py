from .llava_llama import LlavaLlamaForCausalLM
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

class ModalPrompt(LlavaLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
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
            proto_ids_list.append(self.tokenizer(''.join(self.prefix_tokens_list[self.mapping_list[str(i+1)]]),return_tensors='pt').input_ids.squeeze()[1:].to(self.device))
        
        self.proto_ids_list = torch.stack(proto_ids_list,dim=0)

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



    