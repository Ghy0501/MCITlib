import os
import argparse
import json
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./LLaVA/playground/Instructions_slim/OCRVQA/test_1.json')
    parser.add_argument('--result-file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune')
    return parser.parse_args()

def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    annotations = json.load(open(annotation_file))
    annotations = {data['question_id']: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    pred_list = []
    for result in results:
        annotation = annotations[result['question_id']]
        ground_truth = annotation['answer']
        problem = result['prompt']
        image = annotation['image']
        if 'Unanswerable' in result['text'] :
            continue
        
        pred: str = result['text'].lower().replace(' ', '').replace('.', '')
        gt: str =  ground_truth.lower()
        if gt == pred: # all are multiple choice questions or TF questions
            right += 1
        # save the result as jsonl
        pred_list.append(dict(
            question=problem,
            pred=result['text'].lower(),
            ground_truth=ground_truth.lower(),
            image=image,
            score=1 if pred == gt else 0,
        ))
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
        output_file = os.path.join(args.output_dir, 'Result.json')
        with open(output_file, 'w') as f:
            for item in pred_list:
                json.dump(item, f)
                f.write('\n')        

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
