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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

from .multimodal_encoder.builder import build_image_tower, build_video_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector

from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from einops import repeat, rearrange
from torch.nn.functional import cosine_similarity

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if getattr(config, "mm_image_tower", None) is not None:
           self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_video_tower", None) is not None:
           self.video_tower = build_video_tower(config, delay_load=True)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower", None) is not None:
           self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)


    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower
    
    def get_text_tower(self):
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # ==============================================
        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        assert image_tower is not None or video_tower is not None
        # ==============================================
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # ==========================================================================

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()

        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    video_tower = self.video_tower[0]
                else:
                    video_tower = self.video_tower
                video_tower.load_model()

        # ==========================================================================

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # ==========================================================================
        if image_tower is not None and video_tower is not None:  # TODO: support different hidden_size
            assert image_tower.hidden_size == video_tower.hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        else:
            self.config.mm_hidden_size = max(getattr(image_tower, 'hidden_size', -1),
                                             getattr(video_tower, 'hidden_size', -1))
        # ===================================================================================

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
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

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

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()
    
    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def encode_images(self, images):
        clip_image_features, image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return clip_image_features.to, image_features

    def encode_videos(self, videos):  # [mini_b, c, t, h, w]
        b, _, t, _, _ = videos.shape
        clip_video_features, video_features = self.get_model().get_video_tower()(videos)  # [mini_b, t, n, c]
        video_features = self.get_model().mm_projector(video_features)
        return clip_video_features, video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # ====================================================================================================
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()

        if (image_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        '''
            images is a list, if batch_size=6
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(t, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(t, 3, 224, 224),   # sample 5
                video(t, 3, 224, 224),   # sample 5
                video(t, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
            will be converted to image_features, all video_feature will be flatten as image
            [
                [n, c],                  # sample 1
                [n, c),                  # sample 2
                *(t * [new_n, c]),       # sample 3
                [n, c],                  # sample 4
                [n, c],                  # sample 4
                *(t * [new_n, c]),       # sample 5
                *(t * [new_n, c]),       # sample 5
                *(t * [new_n, c]),       # sample 6
                [n, c],                  # sample 6
            ]
        '''
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        is_all_image = len(image_idx) == len(images)
        video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # mini_b c h w
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []  # mini_b c t h w

        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4:  # batch consists of images, [mini_b, c, h, w]
            if image_tower is not None:
                image_features_minibatch = self.encode_images(images_minibatch)  # [mini_b, l, c]
            else:
                image_features_minibatch = torch.randn(1).to(self.device)  # dummy feature for video-only training under tuning
            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        if getattr(videos_minibatch, 'ndim', 0) == 5:  # batch consists of videos, [mini_b, c, t, h, w]
            clip_video_features, video_features_minibatch = self.encode_videos(videos_minibatch)  # fake list [mini_b, t, l, c]
            for i, pos in enumerate(video_idx):
                t = video_features_minibatch[i].shape[0]
                tmp_image_features[pos] = [video_features_minibatch[i][j] for j in range(t)]

        new_tmp = []
        for image in tmp_image_features:
            # print(len(new_tmp), len(image))
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
                # print('add video')
            else:
                new_tmp.append(image)
        image_features = new_tmp
        # print(len(image_features), *[i.shape for i in image_features])
        # print(len(image_features), image_features[0].shape)
        # ====================================================================================================
        
        # assert image_features.shape[1] == 576, 'vision tower not a withprojection version.'
        text_tower = self.get_text_tower()

        input_pad = np.where(input_ids.cpu().detach().numpy()!=-200,input_ids.cpu().detach().numpy(),self.tokenizer.pad_token_id)
        decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
        decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] for decode_input in decoded_hidden_inputs]

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
        
        original_dtype = clip_video_features.dtype

        clip_video_features = torch.nn.functional.interpolate(
            clip_video_features.to(torch.float32).unsqueeze(0), 
            size=768, 
            mode='linear', 
            align_corners=True
        ).to(original_dtype).squeeze(0)
        # bs, num_task
        guide_coef = lam * cosine_similarity(repeat(clip_video_features,'b c -> b n c', n=self.cur_task), repeat(proto_embeddings, 'n c -> b n c',b = bs),dim=-1) + \
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
            self.image_cos_sim_loss = torch.tensor(1, dtype= clip_video_features.dtype, device=self.device)-cosine_similarity(clip_video_features, repeat(proto_embeddings[self.cur_task - 1],'c -> b c', b = bs), dim=-1).mean(dim=0)
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

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images, cur_input_ids)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # print(cur_image_idx)
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
