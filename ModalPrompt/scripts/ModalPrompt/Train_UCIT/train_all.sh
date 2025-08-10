#!/bin/bash
pip install -e .
bash scripts/ModalPrompt/Train_UCIT/Task1.sh configs/model_configs/LLaVA/UCIT/train/task1.json configs/data_configs/UCIT/ImageNet-R.json
pip install -e .
bash scripts/ModalPrompt/Train_UCIT/Task2.sh configs/model_configs/LLaVA/UCIT/train/task2.json configs/data_configs/UCIT/ArxivQA.json
pip install -e .
bash scripts/ModalPrompt/Train_UCIT/Task3.sh configs/model_configs/LLaVA/UCIT/train/task3.json configs/data_configs/UCIT/VizWiz.json
pip install -e .
bash scripts/ModalPrompt/Train_UCIT/Task4.sh configs/model_configs/LLaVA/UCIT/train/task4.json configs/data_configs/UCIT/IconQA.json
pip install -e .
bash scripts/ModalPrompt/Train_UCIT/Task5.sh configs/model_configs/LLaVA/UCIT/train/task5.json configs/data_configs/UCIT/CLEVR-Math.json
pip install -e .
bash scripts/ModalPrompt/Train_UCIT/Task6.sh configs/model_configs/LLaVA/UCIT/train/task6.json configs/data_configs/UCIT/Flickr30k.json

