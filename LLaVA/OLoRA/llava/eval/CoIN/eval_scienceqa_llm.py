import json, os
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers-file', type=str, default='./results/CoIN/ScienceQA/Zero_shot/merge.jsonl')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    answers = open(os.path.expanduser(args.answers_file))
    promts_answers = []

    for i, ans_js in enumerate(answers):
        ans = json.loads(ans_js)
        text = ans['text']
        label = ans['label']

        system_dict = {"role": "system",
                    "content": "You are a helpful and precise assistant for checking the quality of the answer.",}
        user_dict = {"role": "user",
        "content": f"Please only answer the question in yes or no. Is the \"Prediction\" correctly predicting the right \"Label\"? Label: {label}, Prediction: {text}"}
        
        promts_answers.append([system_dict, user_dict])

    json.dump(promts_answers,open('./results/CIFAR100/Original/cifar100_answers_eval_diversity10_another_prompts.jsonl','w'),indent=4)