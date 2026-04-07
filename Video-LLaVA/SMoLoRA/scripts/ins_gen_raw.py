import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
import json
import argparse
import copy
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
# from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List



import pickle

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sentence_bert(sentences):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # sentences = ["<image>\nContext: The passage below describes an experiment. Read the passage and then follow the instructions below.\n\nTom placed a ping pong ball in a catapult, pulled the catapult's arm back to a 45\u00b0 angle, and launched the ball. Then, Tom launched another ping pong ball, this time pulling the catapult's arm back to a 30\u00b0 angle. With each launch, his friend Justin measured the distance between the catapult and the place where the ball hit the ground. Tom and Justin repeated the launches with ping pong balls in four more identical catapults. They compared the distances the balls traveled when launched from a 45\u00b0 angle to the distances the balls traveled when launched from a 30\u00b0 angle.\nFigure: a catapult for launching ping pong balls.\nIdentify the question that Tom and Justin's experiment can best answer.\nA. Do ping pong balls stop rolling along the ground sooner after being launched from a 30\u00b0 angle or a 45\u00b0 angle?\nB. Do ping pong balls travel farther when launched from a 30\u00b0 angle compared to a 45\u00b0 angle?\nAnswer with the option's letter from the given choices directly."]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # print("Sentence embeddings:")
    # print(sentence_embeddings)
    # print(sentence_embeddings.size())
    return sentence_embeddings


if __name__ == "__main__":

    # collect_data()

    parser = argparse.ArgumentParser(description='test')  
    parser.add_argument('--model_path',default="sentence-transformers/all-MiniLM-L6-v2", type=str,help='model_path')  
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)

    list_instruction = ["Answer with the option's letter from the given choices directly.",
                        "Answer the question using a single word or phrase.",
                        "What is happening in the presented picture?\nPlease describe it in one complete sentence.",
                        "What is the object in the image?\nAnswer the question using a single word or phrase.",
                        "Answer the question using a single word or phrase.",
                        "Answer the question using a single word or phrase.",
                        "What is the background of the image?\nAnswer the question using a single word or phrase."]
    
    list_instruction_sci = ["Answer with the option's letter from the given choices directly.",
                    "Select the correct answer by choosing the corresponding letter from the options provided.",
                    "Select the correct letter from the given options to answer the question.",
                    "Identify the correct answer by choosing the appropriate letter from the choices.",
                    "Pick the correct answer by selecting the letter associated with the correct choice."]
    
    list_instruction_img = ["What is the main object present in the image?\nProvide your answer using a word or brief phrase.",
                    "Which specific object does the image depict?\nGive your answer in one word or a short phrase.",
                    "What category does the object in the image belong to?\nAnswer using a single word or phrase.",
                    "What is the primary object visible in the image?\nAnswer using a brief phrase or single word.",
                    "What is the object in the image?\nAnswer briefly with a word or a short phrase."]
    
    list_instruction_vqa = ["Answer the question with a single word or a brief, descriptive phrase.",
                    "Use a single word or a short phrase to respond to the question.",
                    "Use one word or a concise phrase to respond to the question.",
                    "Answer the question with just one word or a brief phrase.",
                    "Answer using only one word or a short, descriptive phrase."]
    
    list_instruction_cap = ['What is happening in the presented picture?\nPlease describe it in one complete sentence.',
                    'What does the image display clearly and succinctly?\nProvide a full sentence explaining it.',
                    'What is depicted in the displayed picture?\nSummarize it using a single, concise sentence.',
                    'How would you interpret the scene in the picture?\nExpress your answer in one informative sentence.',
                    'What is the captured scene about?\nExplain it clearly in one simple sentence.']
    
    list_instruction_place = ["What is the background of the image?\nAnswer the question using a single word or phrase.",
                    "What is the background depicted in the image?\nProvide your answer using a word or brief phrase.",
                    "Which type of background does the image show?\nGive your answer in one word or a short phrase.",
                    "What category best describes the background in the image?\nAnswer using a brief phrase or single word.",
                    "What is the primary background visible in the image?\nAnswer briefly with a word or a short phrase."]
    
    
    list_multi = [list_instruction_sci, list_instruction_vqa, list_instruction_cap, list_instruction_img, list_instruction_vqa, list_instruction_vqa, list_instruction_place]
    
    
    ########################## single指令embedding保存 ###########################
    instruction_emb = sentence_bert(list_instruction)
    print(instruction_emb.size())

    ########################## multi指令embedding保存 ###########################
    # instruction_emb = sentence_bert(list_multi[0])
    # instruction_emb = torch.mean(instruction_emb, dim=0, keepdim=True)
    # # print(instruction_emb)
    # for i in range(1,7):
    #     now_emb = sentence_bert(list_multi[i])
    #     now_emb = torch.mean(now_emb, dim=0, keepdim=True)
    #     instruction_emb = torch.cat([instruction_emb, now_emb], dim=0)


    with open('./ins_emb_single.pkl', 'wb') as f:
        pickle.dump(instruction_emb, f)
        print(instruction_emb)