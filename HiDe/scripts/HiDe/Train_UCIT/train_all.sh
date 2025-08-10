#!/bin/bash
pip install -e .
bash scripts/HiDe/Train_UCIT/Task1.sh configs/model_configs/LLaVA/UCIT/train/task1.json configs/data_configs/UCIT/ImageNet-R.json
bash scripts/HiDe/Eval_UCIT/Eval_finetune1.sh 1
pip install -e .
bash scripts/HiDe/Train_UCIT/Task2.sh configs/model_configs/LLaVA/UCIT/train/task2.json configs/data_configs/UCIT/ArxivQA.json
bash scripts/HiDe/Eval_UCIT/Eval_finetune1.sh 2
pip install -e .
bash scripts/HiDe/Train_UCIT/Task3.sh configs/model_configs/LLaVA/UCIT/train/task3.json configs/data_configs/UCIT/VizWiz.json
bash scripts/HiDe/Eval_UCIT/Eval_finetune1.sh 3
pip install -e .
bash scripts/HiDe/Train_UCIT/Task4.sh configs/model_configs/LLaVA/UCIT/train/task4.json configs/data_configs/UCIT/IconQA.json
bash scripts/HiDe/Eval_UCIT/Eval_finetune1.sh 4
pip install -e .
bash scripts/HiDe/Train_UCIT/Task5.sh configs/model_configs/LLaVA/UCIT/train/task5.json configs/data_configs/UCIT/CLEVR-Math.json
bash scripts/HiDe/Eval_UCIT/Eval_finetune1.sh 5
pip install -e .
bash scripts/HiDe/Train_UCIT/Task6.sh configs/model_configs/LLaVA/UCIT/train/task6.json configs/data_configs/UCIT/Flickr30k.json
bash scripts/HiDe/Eval_UCIT/Eval_finetune1.sh 6

