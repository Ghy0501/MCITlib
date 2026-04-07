import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path,KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math,re


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    waits_to_eval_prompts = open("to_eval_prompt.txt", "r")
    # open text file to read
    waits_to_eval_prompts = waits_to_eval_prompts.readlines()

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for question_path in waits_to_eval_prompts:
        if question_path.strip() == "":
            continue
        question_path = question_path.strip()
        question_path = os.path.join(os.path.expanduser(question_path),'prompt_to_eval.json')
        with open(question_path, "r") as f:
            questions = json.load(f)
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        answer_path = os.path.dirname(question_path)
        answers_file = os.path.expanduser(os.path.join(answer_path, f"prompt_{args.chunk_idx}.jsonl"))
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")

        for i, line in enumerate(tqdm(questions)):
            idx = i
            question = line[-1]['content']
            qs = question.replace('<image>', '').strip()
            cur_prompt = qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
        ans_file.close()

        if args.chunk_idx == 0:
            ## compute scores
            dirs = os.listdir(answer_path)
            prompt_dirs = []
            pattern = re.compile(r'prompt_[1-9]')
            for dir_ in dirs:
                res = pattern.findall(dir_)
                if len(res) > 0:
                    prompt_dirs.append(os.path.join(answer_path,dir_))
            prompts = []
            for path in prompt_dirs:
                prompt = [json.loads(line) for line in open(path)]
                prompts += prompt

            total = 0
            scores = 0
            for prompt in prompts:
                total += 10
                score = prompt['text']
                score.isnumeric()
                try:
                    score = float(score)
                    if score <= 10.0:
                        scores += score
                except:
                    if '/' in score:
                        score = score.split('/')[0]
                        if score.isnumeric() and float(score) <= 10.0:
                            scores += float(score)
                    else:
                        print(score)
            
            final_score = scores / total

            output_file = os.path.join(answer_path,'Prompt_Result.text')
            with open(output_file, 'w') as f:
                f.write('Final score:{}'.format(round(final_score*100.0)))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
