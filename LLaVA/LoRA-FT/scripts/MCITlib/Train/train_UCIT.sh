#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/UCIT/train/task1.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/UCIT/train/task2.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/UCIT/train/task3.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/UCIT/train/task4.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/UCIT/train/task5.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/LLaVA/UCIT/train/task6.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6