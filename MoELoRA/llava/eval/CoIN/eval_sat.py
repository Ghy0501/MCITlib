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
        # type = annotation['type']
        dimension = annotation['type'] # 2D or 3D
        task = annotation['task']
       
        pred: str = result['text'].lower().strip().replace(' ', '').replace('\n', '').replace('.', '')
        gt: str =  ground_truth.lower().strip().replace(' ', '').replace('\n', '').replace('.', '')

        score = 0
        if gt in pred:
            right += 1
            score = 1
        # save the result as jsonl
        pred_list.append(dict(
            question=problem,
            pred=result['text'].lower(),
            ground_truth=ground_truth.lower(),
            image=image,
            score=score,
            dimension=dimension,
            task=task,
        ))
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
   
    right_2d = 0
    total_2d = 0
    right_3d = 0
    total_3d = 0
    count_right = 0
    conut_total = 0
    for item in pred_list:
        if item['task'] == 'Count':
            if item['score']==1:
                count_right += 1
                conut_total += 1
            else:
                conut_total += 1
    for item in pred_list:
        if item['dimension'] == '2D':
            if item['score']==1:
                right_2d += 1
                total_2d += 1
            else:
                total_2d += 1
        if item['dimension'] == '3D':
            if item['score']==1:
                right_3d += 1
                total_3d += 1
            else:
                total_3d += 1
    # print('2D Accuracy: {:.2f}%\n'.format(100. * right_2d / total_2d))
    # print('3D Accuracy: {:.2f}%\n'.format(100. * right_3d / total_3d))
    print('Count Accuracy: {:.2f}%\n'.format(100. * count_right / conut_total))
 
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
            # f.write('3D Accuracy: {:.2f}%\n'.format(100. * right_3d / total_3d))
            # f.write('2D Accuracy: {:.2f}%\n'.format(100. * right_2d / total_2d))
            f.write('Count Accuracy: {:.2f}%\n'.format(100. * count_right / conut_total))

        output_file = os.path.join(args.output_dir, 'Result.json')
        with open(output_file, 'w') as f:
            for item in pred_list:
                json.dump(item, f)
                f.write('\n')        

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
