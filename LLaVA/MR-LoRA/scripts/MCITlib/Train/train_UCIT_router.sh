#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router/task1.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router/task2.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router/task3.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router/task4.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router/task5.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router/task6.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6