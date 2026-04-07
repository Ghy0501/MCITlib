#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/train/task1.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_dcl.sh 1

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/train/task2.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_dcl.sh 2

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/train/task3.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_dcl.sh 3

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/train/task4.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_dcl.sh 4

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/train/task5.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_dcl.sh 5

python scripts/DCL_extract_router_result.py