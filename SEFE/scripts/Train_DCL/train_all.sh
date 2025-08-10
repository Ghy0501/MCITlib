#!/bin/bash
pip install -e .
bash scripts/Train_DCL/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task1.json configs/data_configs/MLLM-DCL/RS.json
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 1
pip install -e .
bash scripts/Train_DCL/Task2.sh configs/model_configs/LLaVA/MLLM-DCL/train/task2.json configs/data_configs/MLLM-DCL/Med.json
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 2
pip install -e .
bash scripts/Train_DCL/Task3.sh configs/model_configs/LLaVA/MLLM-DCL/train/task3.json configs/data_configs/MLLM-DCL/AD.json
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 3
pip install -e .
bash scripts/Train_DCL/Task4.sh configs/model_configs/LLaVA/MLLM-DCL/train/task4.json configs/data_configs/MLLM-DCL/Sci.json
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 4
pip install -e .
bash scripts/Train_DCL/Task5.sh configs/model_configs/LLaVA/MLLM-DCL/train/task5.json configs/data_configs/MLLM-DCL/Fin.json
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 5