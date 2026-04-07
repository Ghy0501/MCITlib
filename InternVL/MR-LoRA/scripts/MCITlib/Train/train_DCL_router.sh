#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/train_router/task1.json
bash scripts/MCITlib/Eval_MLLM_DCL_router/Eval_DCL_router.sh 1

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/train_router/task2.json
bash scripts/MCITlib/Eval_MLLM_DCL_router/Eval_DCL_router.sh 2

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/train_router/task3.json
bash scripts/MCITlib/Eval_MLLM_DCL_router/Eval_DCL_router.sh 3

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/train_router/task4.json
bash scripts/MCITlib/Eval_MLLM_DCL_router/Eval_DCL_router.sh 4

pip install -e .
bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/train_router/task5.json
bash scripts/MCITlib/Eval_MLLM_DCL_router/Eval_DCL_router.sh 5