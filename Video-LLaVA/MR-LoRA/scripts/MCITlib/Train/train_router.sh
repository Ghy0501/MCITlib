#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/counting.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task1.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/space.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task2.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task3.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/movie.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task4.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/gui.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task5.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/science.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task6.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/sports.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task7.json

bash scripts/MCITlib/Train/Task1_router.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/star.json \
   $HARD_PATH/configs/train_configs/MR-LoRA/Video-LLaVA/CL-VISTA/train_router/task8.json
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 1
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 2
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 3
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 4
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 5
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 6
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 7
bash scripts/MCITlib/Eval_router/Eval_CVU_router.sh 8