# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CL-VISTA correctness scoring with a local chat-capable causal LM.

Prompts and scoring convention: MCITlib v3, Video-LLaVA LoRA-FT,
videollava/eval/video/eval_video_qa_qwen.py (Apache-2.0). This version
validates IDs and writes only scores for the current prediction set.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

BATCH_SIZE = 128
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
    "for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine the matching score. "
    "Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer and provide a score from 0 to 5.\n"
    "- 0: No match at all\n"
    "- 1: Very poor match\n"
    "- 2: Poor match\n"
    "- 3: Fair match\n"
    "- 4: Good match\n"
    "- 5: Excellent/Perfect match"
)


def build_prompt(question, answer, pred):
    return (
        f"Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a score where the score is an integer value between 0 and 5, "
        "with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with key 'score' only, "
        "where value of 'score' is an INTEGER between 0 and 5. "
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
        "Only provide the Python dictionary string. "
        "For example: {'score': 4}."
    )


def parse_score(reply: str) -> int:
    text = reply.strip().removeprefix("```").removesuffix("```").strip()
    try:
        return max(0, min(5, int(ast.literal_eval(text)["score"])))
    except (ValueError, TypeError, KeyError, SyntaxError, OverflowError):
        return 0


def read_predictions(path):
    """Use historical string keys, rejecting collisions before any files are written."""
    predictions = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not {"id", "question", "answer", "pred"} <= row.keys():
                raise ValueError(f"Missing prediction fields on line {line_number}")
            raw = row["id"]
            key = str(raw)
            if (type(raw) not in (int, str) or key in ("", ".", "..")
                    or any(ch in key for ch in ("/", "\\", "\x00"))):
                raise ValueError(f"Invalid scorer ID on line {line_number}: {raw!r}")
            if key in predictions:
                raise ValueError(f"Duplicate scorer ID: {key!r}")
            qid = row.get("question_id", raw)
            if type(qid) is not type(raw) or qid != raw:
                raise ValueError(f"question_id and id differ on line {line_number}")
            predictions[key] = {"q": row["question"], "a": row["answer"], "pred": row["pred"]}
    if not predictions:
        raise ValueError("No prediction rows")
    return predictions


def load_judge(model_path, *, attn_implementation=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path).expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("Expected a local model directory")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, max_length=4096)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    options = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", local_files_only=True, **options
    )
    model.eval()
    config = model.generation_config
    config.do_sample = False
    config.temperature = config.top_p = config.top_k = None
    config.num_beams = config.num_return_sequences = 1
    return model, tokenizer


def _batch_scores(rows, model, tokenizer):
    import torch

    texts = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(row["q"], row["a"], row["pred"])},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = "" if rendered is None else str(rendered)
        texts.append(text.encode("utf-8", "replace").decode("utf-8"))
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=4096, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    for ids, output in zip(inputs["input_ids"], outputs, strict=True):
        yield parse_score(tokenizer.decode(output[len(ids):], skip_special_tokens=True))


def score_rows(predictions, model, tokenizer, *, output_dir, output_json, result_path, batch_size=BATCH_SIZE):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty score directory: {output_dir}")
    keys = list(predictions)
    combined = {}
    total = 0
    for start in range(0, len(keys), batch_size):
        batch = keys[start:start + batch_size]
        scores = _batch_scores((predictions[key] for key in batch), model, tokenizer)
        for key, score in zip(batch, scores, strict=True):
            record = [{"score": score}, predictions[key]]
            with (output_dir / f"{key}.json").open("x", encoding="utf-8") as handle:
                json.dump(record, handle)
            combined[key] = record
            total += score
    count = len(combined)
    average = total / count
    with Path(output_json).open("x", encoding="utf-8") as handle:
        json.dump(combined, handle)
    with Path(result_path).open("x", encoding="utf-8") as handle:
        handle.write(f"Original Average Score (0-5): {average:.4f}\n"
                     f"Normalized Average Score (0-100): {average * 20:.4f}\n"
                     f"Number of samples: {count}\n")
    return count


def score_predictions(args, model, tokenizer):
    return score_rows(read_predictions(args.pred_path), model, tokenizer,
                      output_dir=args.output_dir, output_json=args.output_json, result_path=args.result_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model_path", "pred_path", "output_dir", "output_json", "result_path"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    score_predictions(args, *load_judge(args.model_path))


if __name__ == "__main__":
    main()
