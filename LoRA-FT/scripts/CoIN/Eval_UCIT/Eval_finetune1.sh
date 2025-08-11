# #!/bin/bash

if [ "$1" == "1" ]; then
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_imagenet.sh configs/model_configs/LLaVA/UCIT/eval/task1.json configs/data_configs/UCIT/ImageNet-R.json
elif [ "$1" == "2" ]; then
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_imagenet.sh configs/model_configs/LLaVA/UCIT/eval/task2.json configs/data_configs/UCIT/ImageNet-R.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_arxivqa.sh configs/model_configs/LLaVA/UCIT/eval/task2.json configs/data_configs/UCIT/ArxivQA.json
elif [ "$1" == "3" ]; then
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_imagenet.sh configs/model_configs/LLaVA/UCIT/eval/task3.json configs/data_configs/UCIT/ImageNet-R.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_arxivqa.sh configs/model_configs/LLaVA/UCIT/eval/task3.json configs/data_configs/UCIT/ArxivQA.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_vizwiz.sh configs/model_configs/LLaVA/UCIT/eval/task3.json configs/data_configs/UCIT/VizWiz.json
elif [ "$1" == "4" ]; then
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_imagenet.sh configs/model_configs/LLaVA/UCIT/eval/task4.json configs/data_configs/UCIT/ImageNet-R.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_arxivqa.sh configs/model_configs/LLaVA/UCIT/eval/task4.json configs/data_configs/UCIT/ArxivQA.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_vizwiz.sh configs/model_configs/LLaVA/UCIT/eval/task4.json configs/data_configs/UCIT/VizWiz.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_iconqa.sh configs/model_configs/LLaVA/UCIT/eval/task4.json configs/data_configs/UCIT/IconQA.json
elif [ "$1" == "5" ]; then
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_arxivqa.sh configs/model_configs/LLaVA/UCIT/eval/task5.json configs/data_configs/UCIT/ArxivQA.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_imagenet.sh configs/model_configs/LLaVA/UCIT/eval/task5.json configs/data_configs/UCIT/ImageNet-R.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_vizwiz.sh configs/model_configs/LLaVA/UCIT/eval/task5.json configs/data_configs/UCIT/VizWiz.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_iconqa.sh configs/model_configs/LLaVA/UCIT/eval/task5.json configs/data_configs/UCIT/IconQA.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_clevr.sh configs/model_configs/LLaVA/UCIT/eval/task5.json configs/data_configs/UCIT/CLEVR-Math.json
else
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_imagenet.sh configs/model_configs/LLaVA/UCIT/eval/task6.json configs/data_configs/UCIT/ImageNet-R.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_arxivqa.sh configs/model_configs/LLaVA/UCIT/eval/task6.json configs/data_configs/UCIT/ArxivQA.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_vizwiz.sh configs/model_configs/LLaVA/UCIT/eval/task6.json configs/data_configs/UCIT/VizWiz.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_iconqa.sh configs/model_configs/LLaVA/UCIT/eval/task6.json configs/data_configs/UCIT/IconQA.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_clevr.sh configs/model_configs/LLaVA/UCIT/eval/task6.json configs/data_configs/UCIT/CLEVR-Math.json
    pip install -e .
    bash scripts/CoIN/Eval_UCIT/eval_flickr30k.sh configs/model_configs/LLaVA/UCIT/eval/task6.json configs/data_configs/UCIT/Flickr30k.json
fi