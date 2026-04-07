import argparse
import json
import os

import tqdm
import ray
import time
import openai
from openai import OpenAI

NUM_SECONDS_TO_SLEEP = 3


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set the OPENAI_API_KEY environment variable (see .env.example).")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_eval(content: str, max_tokens: int):
    client = _client()
    response = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[{
            'role': 'user',
            'content': content,
        }],
        temperature=0.2,  # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-p', '--prompt_dir')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    question_path = os.path.join(os.path.expanduser(args.prompt_dir),'prompt_to_eval.json')
    with open(question_path, "r") as f:
        questions = json.load(f)
    answer_path = os.path.dirname(question_path)
    answers_file = os.path.expanduser(os.path.join(answer_path, f"chat_gpt_prompt_result.jsonl"))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    js_list = []
    idx = 0
    for question in zip(questions):
        content = question[0][1]['content']
        js_list.append({'id': idx+1,})
        idx += 1
        review = get_eval(content, args.max_tokens)
        print("{}:success!".format(idx))
        scores = review
        # js_list[idx]['text'] = review
        # js_list[idx]['tuple'] = scores
        # ans_file.write(json.dumps(js_list[idx]) + '\n')
        ans_file.write(json.dumps({"question_id": idx, "text": scores}) + "\n")
        ans_file.flush()
    ans_file.close()
