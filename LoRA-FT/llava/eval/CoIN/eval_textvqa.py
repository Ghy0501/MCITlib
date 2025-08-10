import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./cl_dataset/TextVQA/TextVQA_0.5.1_val.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN/MiniGPTv2/TextVQA/Zero_Shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def prompt_processor(prompt):
    if prompt.startswith('OCR tokens:'):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token:' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    annotations = json.load(open(annotation_file))['data']
    annotations = {str(annotation['question_id']): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[str(result['question_id'])]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))

    
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
