#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

# pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/counting.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task1.json

# pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/space.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task2.json
#
#pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task3.json

#pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/movie.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task4.json

#pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/gui.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task5.json

bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/science.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task6.json

bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/sports.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task7.json

bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/star.json \
   $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/train/task8.json

bash scripts/MCITlib/Eval/Eval_CVU.sh 1
bash scripts/MCITlib/Eval/Eval_CVU.sh 2
bash scripts/MCITlib/Eval/Eval_CVU.sh 3
bash scripts/MCITlib/Eval/Eval_CVU.sh 4
bash scripts/MCITlib/Eval/Eval_CVU.sh 5
bash scripts/MCITlib/Eval/Eval_CVU.sh 6
bash scripts/MCITlib/Eval/Eval_CVU.sh 7
bash scripts/MCITlib/Eval/Eval_CVU.sh 8