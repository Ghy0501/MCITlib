import argparse
import json
import os

import openai
from openai import OpenAI
import time

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(content: str, max_tokens: int, client):
    while True:
        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content


def parse_score(review):
    return 1 if 'YES' in review.upper() else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-a', '--answer-list')
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    client = OpenAI(api_key="sk-YsGphZGFuB28yupn5bF52a98A8F1449cBcDd8c064443882f", base_url="https://orisound.cn/v1")

    f_q = open(os.path.expanduser(args.question))
    f_ans = open(os.path.expanduser(args.answer_list))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    handles = []
    idx = 0
    ques_js = json.load(f_q)
    prediction_right = 0
    for i, ans_js in enumerate(f_ans):
        ques = ques_js[i]
        ans = json.loads(ans_js)

        rule = rule_dict['llava_classify']
        
        prompt = rule['prompt']
        role = rule['role']
        label = ques['answer']
        Prediction = ans['text']
        content = (f'[Context]\{prompt} Label:{label}; Prediction:{Prediction}')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'category': 'llava_classify'
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens, client)
            prediction_right += parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = prediction_right
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1

        if idx % 10 == 0:
            print(f"Accuracy: {prediction_right}/{idx} = {prediction_right/idx}")

        print(idx)
    review_file.close()
