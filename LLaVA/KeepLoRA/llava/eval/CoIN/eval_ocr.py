# eval for OCRBench
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
    
    OCRBench_score = {
            'Regular Text Recognition': 0,
            'Irregular Text Recognition': 0,
            'Artistic Text Recognition': 0,
            'Handwriting Recognition': 0,
            'Digit String Recognition': 0,
            'Non-Semantic Text Recognition': 0,
            'Scene Text-centric VQA': 0,
            'Doc-oriented VQA': 0,
            'Key Information Extraction': 0,
            'Handwritten Mathematical Expression Recognition': 0,
        }
    
    total = len(results)
    right = 0
    pred_list = []
    for result in results:
        # import pdb; pdb.set_trace()
        annotation = annotations[result['question_id']]
        ground_truths = annotation['answer']
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]
        problem = result['prompt']
        image = annotation['image']
        predict = result['text']
        category = annotation['answer_type']
        hit_score = 0
        if category == 'Handwritten Mathematical Expression Recognition':
            for j in range(len(ground_truths)):
                ground_truth = ground_truths[j].strip().replace(' ', '').replace('\n', '')
                predict = predict.strip().replace(' ', '').replace('\n', '')
                if ground_truth in predict:
                    OCRBench_score[category] += 1
                    hit_score = 1
                    break
        else:
            for j in range(len(ground_truths)):
                ground_truth = ground_truths[j].lower().strip().replace('\n', '')
                predict = predict.lower().strip().replace('\n', '')
                if ground_truth in predict:
                    OCRBench_score[category] += 1
                    hit_score = 1
                    break
        # pred: str = result['text'].lower().strip().replace(' ', '').replace('.', '').replace('\n', '')
        # gt: str =  ground_truth.lower().strip().replace(' ', '').replace('.', '').replace('\n', '')
 
        # if gt in pred:
        #     right += 1

        # save the result as jsonl
        pred_list.append(dict(
            question=problem,
            pred=predict,
            ground_truth=ground_truths,
            image=image,
            category=category,
            score=hit_score,
        ))
    final_score_dict = {}
    final_score_dict['Text Recognition'] = \
        (OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
            + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
            + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition'])
    final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
    final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
    final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
    final_score_dict['Handwritten Mathematical Expression Recognition'] = \
        (OCRBench_score['Handwritten Mathematical Expression Recognition'])
    final_score_dict['Final Score'] = \
        (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
            + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
            + final_score_dict['Handwritten Mathematical Expression Recognition'])
    final_score_dict['Final Score Norm'] = (float(final_score_dict['Final Score']) / 10)
        
    print('Samples: {}\nFinal_score_dict: {}\n'.format(total, final_score_dict))
    
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nFinal_score_dict: {}\n'.format(total, final_score_dict))
        
        output_file = os.path.join(args.output_dir, 'Result.json')
        with open(output_file, 'w') as f:
            for item in pred_list:
                json.dump(item, f)
                f.write('\n')        

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
