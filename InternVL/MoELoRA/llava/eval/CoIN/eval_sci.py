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
        
        pred: str = result['text'].lower()
        gt: str =  ground_truth.lower()
        if image.split('/')[-1].split('_')[0]=="AI2D" or image.split('/')[-1].split('_')[0]=="TQA" or image.split('/')[-1].split('_')[0]=="VQA" or image.split('/')[-1].split('_')[0]=="SciVerse":
            if gt == pred:
                item_score = 1
                right += item_score
            else:
                item_score = 0
        else: # MapQA
            if 'Which states' in problem:
                gt_list = gt.split(',')
                len_gt = len(gt_list)
                pred_map_list = pred.split(',')
         
                count = 0
                for gt in gt_list:
                    if gt in pred_map_list:
                        count += 1
                item_score = count / len_gt
                right += item_score
            elif gt in pred:
                item_score = 1
                right += item_score
            else:
                item_score = 0

        # save the result as jsonl
        pred_list.append(dict(
            question=problem,
            pred=result['text'].lower(),
            ground_truth=ground_truth.lower(),
            image=image,
            score=item_score,
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
