#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3


bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/SEFE/InternVL/UCIT/train/task1.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/SEFE/InternVL/UCIT/train/task2.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/SEFE/InternVL/UCIT/train/task3.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/SEFE/InternVL/UCIT/train/task4.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/SEFE/InternVL/UCIT/train/task5.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5


bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/SEFE/InternVL/UCIT/train/task6.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6