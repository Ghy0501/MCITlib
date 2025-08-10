#!/bin/bash
pip install -e .
bash scripts/ModalPrompt/Train_DCL/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task1.json configs/data_configs/MLLM-DCL/RS.json
pip install -e .
bash scripts/ModalPrompt/Train_DCL/Task2.sh configs/model_configs/LLaVA/MLLM-DCL/train/task2.json configs/data_configs/MLLM-DCL/Med.json
pip install -e .
bash scripts/ModalPrompt/Train_DCL/Task3.sh configs/model_configs/LLaVA/MLLM-DCL/train/task3.json configs/data_configs/MLLM-DCL/AD.json
pip install -e .
bash scripts/ModalPrompt/Train_DCL/Task4.sh configs/model_configs/LLaVA/MLLM-DCL/train/task4.json configs/data_configs/MLLM-DCL/Sci.json
pip install -e .
bash scripts/ModalPrompt/Train_DCL/Task5.sh configs/model_configs/LLaVA/MLLM-DCL/train/task5.json configs/data_configs/MLLM-DCL/Fin.json