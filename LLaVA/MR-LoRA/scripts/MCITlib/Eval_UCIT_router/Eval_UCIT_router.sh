# #!/bin/bash

TASK_ID=$1
HARD_PATH=/your_path/MCITlib_v3

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task1.json
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task2.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_arxivqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task2.json
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task3.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_arxivqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task3.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_vizwiz.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task3.json
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task4.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_arxivqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task4.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_vizwiz.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task4.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_iconqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task4.json
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval_UCIT_router/eval_arxivqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task5.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task5.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_vizwiz.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task5.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_iconqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task5.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_clevr.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task5.json
else
    bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task6.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_arxivqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task6.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_vizwiz.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task6.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_iconqa.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task6.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_clevr.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task6.json
    bash scripts/MCITlib/Eval_UCIT_router/eval_flickr30k.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router/task6.json
fi