# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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

import os
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
import numpy as np

from .projector import load_mm_projector, build_vision_projector
from .encoder import build_vision_tower, build_text_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP
from einops import repeat, rearrange
from torch.nn.functional import cosine_similarity

class Videollama2MetaModel:

    def __init__(self, config):
        super(Videollama2MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    def get_text_tower(self):
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args, load_pretrained=True)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            for p in self.mm_projector.parameters():
                p.requires_grad = False
            print('successfully tune off the grad of mm projector.')
        else:
            # Change to forze it for prompt learning in case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
            print('successfully tune on the grad of mm projector.')

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)

    def initialize_text_modules(self, model_args, fsdp=None):
        text_tower = model_args.text_tower

        if self.get_text_tower() is None:
            text_tower = build_text_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.text_tower = [text_tower]
            else:
                self.text_tower = text_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                text_tower = self.text_tower[0]
            else:
                text_tower = self.text_tower
            text_tower.load_model()


class Videollama2MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def encode_images_or_videos(self, images):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        data_batch = []
        for i, (data, modal) in enumerate(images):
            if modal == 'image':
                data = data.expand(num_frames, -1, -1, -1)
            else:
                data = data
            data_batch.append(data)

        data_batch = torch.stack(data_batch, dim=0)

        assert len(data_batch.size()) == 5
        batch_size = data_batch.size(0)

        frames = einops.rearrange(data_batch, 'b t c h w -> (b t) c h w')
        clip_frame_features, frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)
        clip_frame_features = einops.rearrange(clip_frame_features, '(b t) h -> b t h', b = batch_size)
        clip_frame_features = clip_frame_features.mean(dim=1)

        return clip_frame_features, self.temporal_aggregator(frames_features)

    def temporal_aggregator(self, frames_features):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")

        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        clip_frame_features, mm_features = self.encode_images_or_videos(images)
        # assert image_features.shape[1] == 576, 'vision tower not a withprojection version.'
        text_tower = self.get_text_tower()
        
        input_np = input_ids.cpu().detach().numpy()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_pad = np.where(input_np<0, pad_id, input_np)
        decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
        decoded_clip_inputs = []
        for decode_input in decoded_hidden_inputs:
            # 取 <</SYS>> 之后的实际问题
            if '<</SYS>>' in decode_input:
                question_part = decode_input.split('<</SYS>>')[-1]
            else:
                question_part = decode_input
            # 去掉 [/INST] 及之后
            question_part = question_part.split('[/INST]')[0].strip()
            decoded_clip_inputs.append(question_part)

        clip_text_inputs = self.clip_tokenizer(
                decoded_clip_inputs,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

        # text_guide_features: bs, 768
        bs = input_ids.shape[0]
        text_guide_features = text_tower(clip_text_inputs)

        # 'input_ids': [1, 32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009]
        prompt_ids_list = self.get_proto_input_ids(self.cur_task, self.device)

        #  num_task, 768
        proto_tokens = [sub_prompt_transform(self.model.embed_tokens(prompt_ids_list[i])).mean(dim=0) for (i,sub_prompt_transform) in enumerate(self.prompt_transform[:self.cur_task])]
        proto_embeddings = torch.stack(proto_tokens, dim=0)
        # print(proto_embeddings)
        lam = 0.5
        transfer_num = 1
        assert transfer_num in [1,2,3], "not implemented yet."

        clip_frame_features = torch.nn.functional.interpolate(clip_frame_features.unsqueeze(0), size=768, mode='linear', align_corners=True).squeeze(0)
        # bs, num_task
        # print(image_guide_features)
        guide_coef = lam * cosine_similarity(repeat(clip_frame_features,'b c -> b n c', n=self.cur_task), repeat(proto_embeddings, 'n c -> b n c',b = bs),dim=-1) + \
                (1-lam) * cosine_similarity(repeat(text_guide_features,'b c -> b n c', n=self.cur_task), repeat(proto_embeddings, 'n c -> b n c',b = bs),dim=-1)

        if self.training:
            if transfer_num ==1:
                input_ids = torch.cat((input_ids[:,0:1], prompt_ids_list[-1:].repeat(bs, 1), input_ids[:,1:]),dim=1)
                attention_mask = torch.cat((torch.ones_like(prompt_ids_list[-1:].repeat(bs, 1), dtype= torch.bool, device = attention_mask.device), attention_mask),dim=1)
                labels = torch.cat((IGNORE_INDEX * torch.ones_like(prompt_ids_list[-1:].repeat(bs, 1), dtype= torch.bool, device = attention_mask.device), labels),dim=1)
            
            elif self.cur_task <= transfer_num -1:
                input_ids = torch.cat((input_ids[:,0:1], rearrange(prompt_ids_list, 'n c -> 1 (n c)').repeat(bs, 1), input_ids[:,1:]),dim=1)
                attention_mask = torch.cat((torch.ones_like(rearrange(prompt_ids_list, 'n c -> 1 (n c)').repeat(bs, 1), dtype=torch.bool, device = attention_mask.device), attention_mask), dim=1)
                labels = torch.cat((IGNORE_INDEX * torch.ones_like(rearrange(prompt_ids_list,'n c -> 1 (n c)').repeat(bs, 1), device = attention_mask.device), labels), dim=1)

            else:
                select_topk = transfer_num - 1
                # bs, select_topk, always select cur prompts and put it at the nearest position
                top_guide = torch.flip(torch.topk(guide_coef[:,:-1], k=select_topk,dim=-1)[1], dims=[1])
                if select_topk == 2:
                    select_proto_ids = torch.cat((prompt_ids_list.gather(0, repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len)), prompt_ids_list.gather(0, repeat(top_guide[:,1], 'b -> b l', l=self.prefix_len))), dim=1)
                elif select_topk == 1: 
                    select_proto_ids = prompt_ids_list.gather(0, repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len))
                input_ids = torch.cat((input_ids[:,0:1], select_proto_ids, prompt_ids_list[-1:].repeat(bs, 1), input_ids[:,1:]),dim=1)
                attention_mask = torch.cat((torch.ones_like(select_proto_ids, dtype=torch.bool, device=attention_mask.device), torch.ones_like(prompt_ids_list[-1:].repeat(bs, 1), dtype= torch.bool, device = attention_mask.device), attention_mask),dim=1)
                labels = torch.cat((IGNORE_INDEX * torch.ones_like(select_proto_ids,device = attention_mask.device), IGNORE_INDEX * torch.ones_like(prompt_ids_list[-1:].repeat(bs, 1), dtype= torch.bool, device = attention_mask.device), labels),dim=1)
            
            # cosine similarity loss = 1 - cosine similarity to ensure they are non-negative
            self.image_cos_sim_loss = torch.tensor(1, dtype= clip_frame_features.dtype, device=self.device)-cosine_similarity(clip_frame_features, repeat(proto_embeddings[self.cur_task - 1],'c -> b c', b = bs), dim=-1).mean(dim=0)
            self.text_cos_sim_loss = torch.tensor(1, dtype= text_guide_features.dtype, device=self.device)-cosine_similarity(text_guide_features, repeat(proto_embeddings[self.cur_task - 1],'c -> b c', b = bs), dim=-1).mean(dim=0)

        else: # for inference
            if transfer_num == 3:
                if self.cur_task == 1:
                    top_guide = torch.flip(torch.topk(guide_coef, k=self.cur_task, dim=-1)[1],dims=[1])
                    select_proto_ids = prompt_ids_list.gather(0,repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len))
                elif self.cur_task == 2:
                    top_guide = torch.flip(torch.topk(guide_coef, k=self.cur_task, dim=-1)[1],dims=[1])
                    select_proto_ids = torch.cat((prompt_ids_list.gather(0,repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len)), prompt_ids_list.gather(0,repeat(top_guide[:,1], 'b -> b l', l=self.prefix_len))), dim=1)
                else:
                    top_guide = torch.flip(torch.topk(guide_coef, k=transfer_num, dim=-1)[1],dims=[1])
                    select_proto_ids = torch.cat((prompt_ids_list.gather(0,repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len)), prompt_ids_list.gather(0,repeat(top_guide[:,1], 'b -> b l', l=self.prefix_len)), prompt_ids_list.gather(0,repeat(top_guide[:,2], 'b -> b l', l = self.prefix_len))), dim=1)
            elif transfer_num == 2:
                if self.cur_task == 1:
                    top_guide = torch.flip(torch.topk(guide_coef, k=self.cur_task, dim=-1)[1],dims=[1])
                    select_proto_ids = prompt_ids_list.gather(0,repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len))
                else:
                    top_guide = torch.flip(torch.topk(guide_coef, k=transfer_num, dim=-1)[1],dims=[1])
                    # print(guide_coef)
                    # print(top_guide)
                    select_proto_ids = torch.cat((prompt_ids_list.gather(0,repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len)), prompt_ids_list.gather(0,repeat(top_guide[:,1], 'b -> b l', l=self.prefix_len))), dim=1)
            elif transfer_num == 1:
                top_guide = torch.flip(torch.topk(guide_coef, k=transfer_num, dim=-1)[1],dims=[1])
                select_proto_ids = prompt_ids_list.gather(0,repeat(top_guide[:,0], 'b -> b l', l=self.prefix_len))

            
            input_ids = torch.cat((input_ids[:,0:1], select_proto_ids, input_ids[:,1:]),dim=1)
            attention_mask = torch.cat((torch.ones_like(select_proto_ids, dtype= torch.bool, device=attention_mask.device), attention_mask),dim=1)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            while mm_token_indices.numel() > 0:
                cur_mm_features = mm_features[cur_mm_idx]
                mm_token_start = mm_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_mm_idx += 1
                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
