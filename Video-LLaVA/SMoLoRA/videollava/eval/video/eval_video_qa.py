import os
import argparse
import json
import ast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "/mnt/ShareDB-3TB/syc/vllm_deploy/Qwen/Qwen3-30B-A3B-Instruct-2507"
BATCH_SIZE = 128
MAX_NEW_TOKENS = 128


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)


tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype="auto",
    local_files_only=True
)
model.eval()

gen_cfg = model.generation_config
gen_cfg.do_sample = False
gen_cfg.temperature = None
gen_cfg.top_p = None
gen_cfg.top_k = None
gen_cfg.num_beams = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--result_path", required=True)
    return parser.parse_args()


def clean_response_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[-1]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def build_prompt(question, answer, pred):
    return (
        f"Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, "
        "with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', "
        "where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
        "Only provide the Python dictionary string. "
        "For example: {'pred': 'yes', 'score': 4.8}."
    )


def process_batch(file_names, prediction_set, output_dir):
    messages_list = []
    keys = []

    for fname in file_names:
        key = fname[:-5]
        if key not in prediction_set:
            continue

        qa = prediction_set[key]
        prompt = build_prompt(qa["q"], qa["a"], qa["pred"])

        messages = [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
                    "for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they "
                    "match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        messages_list.append(messages)
        keys.append(key)

    if not messages_list:
        return

    texts = [
        tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=True
        )
        for m in messages_list
    ]

    model_inputs = tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    for i, key in enumerate(keys):
        output_ids = outputs[i][len(model_inputs.input_ids[i]):]
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        try:
            cleaned = clean_response_text(content)
            resp = ast.literal_eval(cleaned)

            if prediction_set[key]["a"].lower() == "no" and resp["pred"].lower() == "no":
                resp["pred"] = "yes"
                resp["score"] = 5

            result = [resp, prediction_set[key]]

            with open(os.path.join(output_dir, f"{key}.json"), "w") as f:
                json.dump(result, f)

        except Exception as e:
            print(f"[ERROR] {key}: {e}")


def main():
    args = parse_args()

    print(f"Loading predictions from {args.pred_path}")
    with open(args.pred_path, "r") as f:
        samples = [eval(line.strip()) for line in f]

    prediction_set = {}
    id_files = []

    for s in samples:
        prediction_set[s["id"]] = {
            "q": s["question"],
            "a": s["answer"],
            "pred": s["pred"]
        }
        id_files.append(f"{s['id']}.json")

    os.makedirs(args.output_dir, exist_ok=True)

    completed = set(os.listdir(args.output_dir))
    remaining = [f for f in id_files if f not in completed]

    print(f"Remaining to process: {len(remaining)}")

    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Batch Processing"):
        batch_files = remaining[i:i + BATCH_SIZE]
        process_batch(batch_files, prediction_set, args.output_dir)


    combined = {}
    for fname in os.listdir(args.output_dir):
        if fname.endswith(".json"):
            with open(os.path.join(args.output_dir, fname), "r") as f:
                combined[fname[:-5]] = json.load(f)

    with open(args.output_json, "w") as f:
        json.dump(combined, f)


    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0

    for result in combined.values():
        count += 1
        score_sum += float(result[0]["score"])
        if "yes" in result[0]["pred"].lower():
            yes_count += 1
        elif "no" in result[0]["pred"].lower():
            no_count += 1

    if count > 0:
        acc = yes_count / (yes_count + no_count)
        avg = score_sum / count
        print(f"Accuracy: {acc:.4f}, Avg Score: {avg:.4f}")
        with open(args.result_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\nAverage score: {avg:.4f}\n")


if __name__ == "__main__":
    main()
