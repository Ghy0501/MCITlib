#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/UCIT/train_router/task1.json
bash scripts/MCITlib/Eval_UCIT_router/Eval_UCIT_router.sh 1

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/UCIT/train_router/task2.json
bash scripts/MCITlib/Eval_UCIT_router/Eval_UCIT_router.sh 2

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/UCIT/train_router/task3.json
bash scripts/MCITlib/Eval_UCIT_router/Eval_UCIT_router.sh 3

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/UCIT/train_router/task4.json
bash scripts/MCITlib/Eval_UCIT_router/Eval_UCIT_router.sh 4

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/UCIT/train_router/task5.json
bash scripts/MCITlib/Eval_UCIT_router/Eval_UCIT_router.sh 5

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/UCIT/train_router/task6.json
bash scripts/MCITlib/Eval_UCIT_router/Eval_UCIT_router.sh 6