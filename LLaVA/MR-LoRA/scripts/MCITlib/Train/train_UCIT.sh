#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task1.json

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task2.json

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task3.json

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task4.json

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task5.json

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task6.json
bash scripts/MCITlib/Eval_UCIT/Eval_ucit.sh 6

python scripts/UCIT_extract_router_result.py