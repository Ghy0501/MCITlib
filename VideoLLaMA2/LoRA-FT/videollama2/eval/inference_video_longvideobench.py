import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='TypedStorage is deprecated'
)


# =========================
# Utils
# =========================
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def load_subtitle_json(sub_path):
    """
    Load LongVideoBench subtitle json and concatenate all lines
    """
    if not os.path.exists(sub_path):
        return ""

    try:
        data = json.load(open(sub_path, 'r'))
        lines = [x['line'].strip() for x in data if x.get('line')]
        return "\n".join(lines)
    except:
        traceback.print_exc()
        return ""


# =========================
# Dataset
# =========================
class LongVideoBenchDataset(Dataset):

    def __init__(self, data_list, video_root, subtitle_root, processor):
        self.data_list = data_list
        self.video_root = video_root
        self.subtitle_root = subtitle_root
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        video_name = item['video']
        video_path = os.path.join(self.video_root, video_name)

        # subtitle: xxx.mp4 -> xxx_en.json
        base = os.path.splitext(video_name)[0]
        subtitle_path = os.path.join(
            self.subtitle_root, f"{base}_en.json"
        )

        video_tensor = self.processor(video_path)
        subtitle_text = load_subtitle_json(subtitle_path)

        question = item['question']
        options = item['candidates']
        answer = item['answer']

        letters = []
        options_string = ''
        answer_idx = -1

        for i, opt in enumerate(options):
            letter = chr(ord('A') + i)
            letters.append(letter)
            options_string += f"({letter}) {opt}\n"
            if opt == answer:
                answer_idx = i

        instruct = (
            f"Question: {question}\n"
            f"Options:\n{options_string}"
            f"Answer with the option's letter from the given choices directly and only give the best option."
        )

        return {
            "video": video_tensor,
            "video_path": video_path,
            "subtitle": subtitle_text,
            "instruct": instruct,
            "letters": letters,
            "options": options,
            "answer_idx": answer_idx
        }


# =========================
# Dataloader
# =========================
def build_longvideobench_eval(args, processor):
    data = json.load(open(args.question_file, 'r'))
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    dataset = LongVideoBenchDataset(
        data_list=data,
        video_root=args.video_folder,
        subtitle_root=args.subtitle_folder,
        processor=processor
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )


# =========================
# Answer Parsing
# =========================
def mvbench_dump(vid, instruct, letters, options, output):
    output = output.replace('Answer', '').replace('answer', '')

    pred_answer = re.findall(
        f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*',
        output
    )

    try:
        if len(pred_answer) > 0:
            pred = pred_answer[0].strip().strip('()')
            return letters.index(pred)

        for i, opt in enumerate(options):
            if opt.lower() in output.lower():
                return i
    except:
        traceback.print_exc()

    return 0


# =========================
# Inference
# =========================
def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path, args.model_base)

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    fout = open(args.answer_file, 'w')

    loader = build_longvideobench_eval(args, processor['video'])

    for batch in tqdm(loader):
        video_tensor = batch['video'][0]
        subtitle = batch['subtitle'][0]
        vid = batch['video_path'][0]
        instruct = batch['instruct'][0]
        letters = list(zip(*batch['letters']))[0]
        options = list(zip(*batch['options']))[0]
        gt = batch['answer_idx'][0].item()

        # -------- video only --------
        output = mm_infer(
            video_tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False
        )
        pred = mvbench_dump(vid, instruct, letters, options, output)

        # -------- video + subtitle --------
        if subtitle.strip():
            instruct_sub = (
                f"This video's subtitles are listed below:\n{subtitle}\n\n"
                + instruct
            )
        else:
            instruct_sub = instruct

        output_sub = mm_infer(
            video_tensor,
            instruct_sub,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False
        )
        pred_sub = mvbench_dump(vid, instruct_sub, letters, options, output_sub)

        fout.write(json.dumps({
            "video": vid,
            "pred": pred,
            "pred_sub": pred_sub,
            "gt": gt
        }) + "\n")

    fout.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', help='Path to the model directory.', required=True)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument('--video-folder', required=True)
    parser.add_argument('--subtitle-folder', required=True)
    parser.add_argument('--question-file', required=True)
    parser.add_argument('--answer-file', required=True)

    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()
    run_inference(args)
