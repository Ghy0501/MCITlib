#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
    $HARD_PATH/configs/train_configs/DISCO/LLaVA/MLLM-DCL/train/task1.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 1 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
    $HARD_PATH/configs/train_configs/DISCO/LLaVA/MLLM-DCL/train/task2.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 2 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
    $HARD_PATH/configs/train_configs/DISCO/LLaVA/MLLM-DCL/train/task3.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 3 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
    $HARD_PATH/configs/train_configs/DISCO/LLaVA/MLLM-DCL/train/task4.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 4 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
    $HARD_PATH/configs/train_configs/DISCO/LLaVA/MLLM-DCL/train/task5.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 5 $HARD_PATH