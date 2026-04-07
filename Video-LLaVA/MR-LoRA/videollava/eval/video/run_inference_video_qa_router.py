import math
import os
import argparse
import json
import time

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize

# 定义路由 Prompt 的前后缀
ROUTER_PROMPT_PREFIX = """You are a helpful assistant router for Video Question Answering (VideoQA) tasks. There are eight expert models, each specializing in a specific domain of video understanding.

Your task is to select the most suitable model based on the provided video content, user question, and model descriptions. Consider the temporal and visual expertise of each model carefully.

Important Instructions:
• Respond only with the letter (A, B, C, D, E, F, G, H) corresponding to the most suitable model.
• Do not attempt to answer the user's question directly.

Model Pool:
• A: A movie expert. This model is adept at interpreting cinematic sequences, understanding narrative plots, character interactions, and subtitles within movies or TV shows.
• B: A reasoning expert. This model is designed for complex video logic, analyzing cause-and-effect relationships (e.g., "why did the glass break?"), inferring intent, and temporal deduction.
• C: A GUI (Graphical User Interface) expert. This model specializes in understanding screen recordings, tracking cursor movements, identifying software interactions, and analyzing web or app navigation flows.
• D: A traffic analysis expert. This model focuses on dashcam or surveillance footage, analyzing vehicle behavior, traffic flow, road accidents, and driving scenarios.
• E: A science expert. This model has proficiency in interpreting videos of scientific experiments, physical phenomena, chemical reactions, and dynamic educational visualizations.
• F: A video counting expert. This model excels at quantifying objects or actions within a video clip. It answers "how many times" an action occurred or counts objects moving across frames.
• G: A spatial understanding expert. This model specializes in analyzing the relative positions and movements of objects over time, tracking trajectories, and understanding 3D geometry in dynamic scenes.
• H: A sports expert. This model excels at recognizing specific sports, analyzing athletes' movements, understanding game rules, and interpreting scoreboards in match highlights.

Here is the user's question: """

ROUTER_PROMPT_SUFFIX = """

You only need to select the suitable model and do not answer the question. JUST answer with the model's letter from the given choices directly."""


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    # [修改点]：构建路由 Prompt
    # 将原始问题 qs 包裹在 Router 的指令中
    routing_qs = ROUTER_PROMPT_PREFIX + qs + ROUTER_PROMPT_SUFFIX

    # 处理视频 Token
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + routing_qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + routing_qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=True,
            temperature=0.1, # 保持较低的 temperature 以获得稳定的路由选择
            max_new_tokens=16, # [修改点]：只需要输出一个字母，不需要生成长文本，减小 max_new_tokens
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    # 这里输出的 outputs 理论上应该是 "A", "B", "C" 等字母
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # 初始化计时变量
    total_inference_time = 0.0
    processed_count = 0

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats: 
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                
                # 计时开始
                start_time = time.time()

                # Run inference on the video and add the output to the list
                # 现在的 output 将是 Router 的选择 (例如 "A", "B"...)
                output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                
                # 计时结束
                end_time = time.time()
                elapsed = end_time - start_time
                total_inference_time += elapsed
                processed_count += 1
                
                # 将 Router 的选择保存为 'pred'
                sample_set['pred'] = output
                # 如果你想明确字段名，也可以改为 sample_set['agent_selection'] = output
                
                output_list.append(sample_set)
                
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()

    # 计算并记录当前 Chunk 的平均时间
    if processed_count > 0:
        avg_time = total_inference_time / processed_count
        
        # 汇总文件路径
        times_file_path = os.path.join(args.output_dir, "times.txt")
        
        # 使用追加模式 'a' 写入
        with open(times_file_path, "a") as f:
            f.write(f"Chunk {args.chunk_idx}: Avg Time = {avg_time:.4f}s (Total: {total_inference_time:.2f}s, Count: {processed_count})\n")
        
        print(f"Chunk {args.chunk_idx} finished. Avg time: {avg_time:.4f}s saved to {times_file_path}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)