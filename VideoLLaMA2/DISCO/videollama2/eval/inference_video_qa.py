import os
import math
import json
import time      # [新增]
import argparse
import warnings
from tqdm import tqdm

import torch
import sys

sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to the model directory.', required=True)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing questions.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--text-tower", type=str)
    parser.add_argument("--num-task", type=int, default=0)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--num_workers", type=int, required=False, default=8)
    return parser.parse_args()


def get_model_output(model, processor, tokenizer, video_path, question, args):
    video_tensor = processor['video'](video_path)
    output = mm_infer(
        video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='video',
        do_sample=False,
    )
    return output


def run_inference(args):
    # 1. Initialize the model
    disable_torch_init()
    print(args.text_tower)
    model, processor, tokenizer = model_init(args.model_path, args.model_base, num_task=args.num_task, text_tower=args.text_tower)

    # 2. Load Data
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    if os.path.exists(args.gt_file_answers):
        gt_answers = json.load(open(args.gt_file_answers, "r"))
        gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)
    else:
        gt_answers = None

    # 3. Prepare Output
    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    ans_file = open(answers_file, "w")

    video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv']

    # [新增] 初始化计时变量
    total_inference_time = 0.0
    processed_count = 0

    # 4. Iterate and Infer
    for index, sample in enumerate(tqdm(gt_questions)):
        video_name = sample.get('video_name', sample.get('id'))
        question = sample.get('question', sample.get('Q'))
        question_id = sample.get('question_id', sample.get('id', index))

        answer = ""
        if gt_answers:
            answer = gt_answers[index].get('answer', gt_answers[index].get('A', ''))
        elif 'answer' in sample:
            answer = sample['answer']
        elif 'A' in sample:
            answer = sample['A']

        sample_set = {
            'id': question_id,
            'video_name': video_name,
            'question': question,
            'answer': answer
        }

        video_path = None
        for fmt in video_formats:
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if video_path is None:
            print(f"Warning: Video {video_name} not found in {args.video_dir}")
            sample_set['pred'] = "Error: Video not found"
        else:
            try:
                # [新增] 计时开始
                start_time = time.time()

                output = get_model_output(model, processor, tokenizer, video_path, question, args)

                # [新增] 计时结束并累加
                elapsed = time.time() - start_time
                total_inference_time += elapsed
                processed_count += 1

                sample_set['pred'] = output
            except Exception as e:
                print(f"Error processing video {video_name}: {e}")
                sample_set['pred'] = f"Error: {str(e)}"

        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"Inference finished. Results saved to {answers_file}")

    # [新增] 计算并写入统计结果
    if processed_count > 0:
        avg_time = total_inference_time / processed_count
        times_file_path = os.path.join(args.output_dir, "times.txt")

        # 追加模式，支持多 Chunk 并行写入同一文件
        with open(times_file_path, "a") as f:
            f.write(
                f"Chunk {args.chunk_idx}: "
                f"Avg Time = {avg_time:.4f}s  "
                f"(Total: {total_inference_time:.2f}s, Count: {processed_count})\n"
            )
        print(f"Chunk {args.chunk_idx} finished. Avg time: {avg_time:.4f}s saved to {times_file_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)