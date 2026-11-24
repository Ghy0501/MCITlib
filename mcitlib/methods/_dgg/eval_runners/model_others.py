# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""UCIT generation matching DGG's released model_answer evaluation path."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def continuation(output_ids, input_ids):
    """Support V4's input-id return and upstream's inputs-embeds return."""
    length = input_ids.shape[1]
    if output_ids.shape[1] >= length and torch.equal(output_ids[:, :length], input_ids):
        return output_ids[:, length:]
    return output_ids


class Questions(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        row = self.questions[index]
        image_token = DEFAULT_IMAGE_TOKEN
        if self.model_config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        conversation = conv_templates[self.conv_mode].copy()
        conversation.append_message(conversation.roles[0], image_token + "\n" + row["text"])
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        with Image.open(os.path.join(self.image_folder, row["image"])) as image:
            image_tensor = process_images([image.convert("RGB")], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    with open(os.path.expanduser(args.question_file), encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or args.num_chunks < 1 or not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("invalid DGG evaluation questions or chunk")
    chunk_size = math.ceil(len(questions) / args.num_chunks)
    questions = questions[args.chunk_idx * chunk_size:(args.chunk_idx + 1) * chunk_size]
    mode = args.conv_mode
    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in mode:
        mode += "_mmtag"
    dataset = Questions(questions, args.image_folder, tokenizer, image_processor, model.config, mode)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    output = Path(args.answers_file).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for (input_ids, images), row in tqdm(zip(loader, questions), total=len(questions)):
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=images.to(device="cuda", dtype=torch.float16, non_blocking=True),
                                            do_sample=args.temperature > 0, temperature=args.temperature,
                                            top_p=args.top_p, num_beams=args.num_beams,
                                            max_new_tokens=args.max_new_tokens, use_cache=True)
            answer = tokenizer.batch_decode(continuation(output_ids, input_ids), skip_special_tokens=True)[0].strip()
            file.write(json.dumps({"question_id": row["question_id"], "prompt": row["text"], "text": answer,
                                   "answer_id": shortuuid.uuid(), "model_id": model_name, "metadata": {}}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    eval_model(parser.parse_args())


if __name__ == "__main__":
    main()
