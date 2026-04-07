import os
import math
import json
import argparse
import warnings
from tqdm import tqdm

import torch
import sys

# 添加当前路径以便导入 videollama2 模块
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


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

    parser.add_argument('--model_path', help='Path to the model directory.', required=True)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing questions.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--num_workers", type=int, required=False, default=8)
    parser.add_argument("--prefix-len", type=int, default=10)
    parser.add_argument("--cur-task", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=6)
    parser.add_argument("--text-tower",type=str)

    return parser.parse_args()


def get_model_output(model, processor, tokenizer, video_path, question, args):
    """
    Encapsulates the VideoLLaMA2 inference logic.
    """
    # VideoLLaMA2 的 processor 处理视频路径
    video_tensor = processor['video'](video_path)
    
    # 将 tensor 移动到指定设备 (VideoLLaMA2 内部通常会自动处理，但显式转换更安全)
    # 注意：mm_infer 内部通常期望 tensor 在正确的 device 上，或者它会自动移动
    # 这里我们假设 model_init 已经把模型放到了 device 上，
    # video_tensor 是由 processor 生成的，通常是 CPU tensor，mm_infer 会处理它。
    
    output = mm_infer(
        video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='video',
        do_sample=False, # 使用贪婪搜索以获得确定性结果
    )
    
    return output


def run_inference(args):
    """
    Run inference using the VideoLLaMA2 model.
    """
    # 1. Initialize the model
    disable_torch_init()
    # model_init 加载模型、处理器和分词器
    # 注意：VideoLLaMA2 的 model_init 通常会自动将模型加载到 GPU
    model, processor, tokenizer = model_init(args.model_path, args.model_base, args.prefix_len, args.cur_task, args.text_tower, args.num_tasks)
    # 2. Load Data
    # 加载问题文件
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    
    # 加载答案文件 (如果存在且需要)
    if os.path.exists(args.gt_file_answers):
        gt_answers = json.load(open(args.gt_file_answers, "r"))
        gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)
    else:
        # 如果没有单独的答案文件，假设答案在问题文件中或者不需要答案
        gt_answers = None

    # 3. Prepare Output
    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    ans_file = open(answers_file, "w")

    video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv']

    # 4. Iterate and Infer
    # 使用 tqdm 显示进度
    for index, sample in enumerate(tqdm(gt_questions)):
        video_name = sample.get('video_name', sample.get('id')) # 兼容不同数据集的键名
        question = sample.get('question', sample.get('Q')) # 兼容 Q 或 question
        question_id = sample.get('question_id', sample.get('id', index))
        
        # 获取对应的 ground truth answer
        answer = ""
        if gt_answers:
            # 假设 gt_answers 和 gt_questions 是对齐的列表
            answer = gt_answers[index].get('answer', gt_answers[index].get('A', ''))
        elif 'answer' in sample:
            answer = sample['answer']
        elif 'A' in sample:
            answer = sample['A']

        # 构造结果字典
        sample_set = {
            'id': question_id, 
            'video_name': video_name,
            'question': question, 
            'answer': answer
        }

        # 查找视频文件
        video_path = None
        for fmt in video_formats:
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        
        if video_path is None:
            # 如果找不到视频，记录错误或跳过
            print(f"Warning: Video {video_name} not found in {args.video_dir}")
            sample_set['pred'] = "Error: Video not found"
        else:
            try:
                # 执行推理
                output = get_model_output(model, processor, tokenizer, video_path, question, args)
                sample_set['pred'] = output
            except Exception as e:
                print(f"Error processing video {video_name}: {e}")
                sample_set['pred'] = f"Error: {str(e)}"

        # 实时写入文件 (JSONL 格式)
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush() # 确保内容被写入磁盘

    ans_file.close()
    print(f"Inference finished. Results saved to {answers_file}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)