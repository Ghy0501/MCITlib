#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-ACL/train_router/task1.json
bash scripts/MCITlib/Eval_MLLM_ACL_router/Eval_ACL_router.sh 1

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-ACL/train_router/task2.json
bash scripts/MCITlib/Eval_MLLM_ACL_router/Eval_ACL_router.sh 2

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-ACL/train_router/task3.json
bash scripts/MCITlib/Eval_MLLM_ACL_router/Eval_ACL_router.sh 3

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-ACL/train_router/task4.json
bash scripts/MCITlib/Eval_MLLM_ACL_router/Eval_ACL_router.sh 4