import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from typing import List


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def sentence_bert(sentences: List[str]):
    tokenizer = AutoTokenizer.from_pretrained('/mnt/ShareDB-3TB/syc/model/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('/mnt/ShareDB-3TB/syc/model/all-MiniLM-L6-v2')

    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Instruction embedding generation')
    parser.add_argument(
        '--model_path',
        default="/mnt/ShareDB-3TB/syc/model/all-MiniLM-L6-v2",
        type=str,
        help='model_path'
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)

    # ===================== Instruction Pool =====================
    list_instruction = [
        # ===== Multiple Choice =====
        "Answer with the option’s letter from the given choices directly.",
        "Select the correct answer from the given choices and respond with the letter of the chosen option.",
        "Determine the correct option from the provided choices and reply with its corresponding letter.",
        "Pick the correct answer from the listed options and provide the letter of the selected option.",
        "Identify the correct choice from the options and respond with the letter of the correct option.",

        # ===== Binary (Yes / No) =====
        "Answer Yes or No directly.",
        "Please respond with only Yes or No.",
        "Is the statement true or false? Answer with Yes or No.",
        "Reply with a simple Yes or No.",
        "Determine if the statement is correct and answer Yes or No.",

        # ===== Short Answer =====
        "Answer briefly with a single word or short phrase.",
        "Provide a concise answer in 1 to 5 words.",
        "Reply using a short phrase only, do not use a full sentence.",
        "Keep your answer short and to the point.",
        "Answer directly using as few words as possible.",

        # ===== Long Answer =====
        "Provide a detailed and comprehensive answer.",
        "Answer with a complete, descriptive paragraph.",
        "Elaborate on your answer with full sentences and details.",
        "Give a thorough explanation in your response.",
        "Describe your reasoning in detail using a complete paragraph."
    ]

    print(f"[INFO] Total instructions: {len(list_instruction)}")

    # ===================== Embedding =====================
    instruction_emb = sentence_bert(list_instruction)
    print("[INFO] Embedding shape:", instruction_emb.size())

    # ===================== Save =====================
    with open('./ins_emb_single.pkl', 'wb') as f:
        pickle.dump(
            {
                "instructions": list_instruction,
                "embeddings": instruction_emb
            },
            f
        )

    print("[INFO] Saved instruction embeddings to ./ins_emb_single.pkl")
