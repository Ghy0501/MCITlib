#!/bin/bash
set -e

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/extract_weights.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task0.json \
   0.6


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/counting.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task1.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Task1.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/counting.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task1.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/counting.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task1.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 1


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/space.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task2.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/space.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task2.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/space.json  \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task2.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 2


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task3.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task3.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json  \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task3.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 3


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/movie.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task4.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/movie.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task4.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/movie.json  \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task4.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 4


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/gui.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task5.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/gui.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task5.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/gui.json  \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task5.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 5


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/science.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task6.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/science.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task6.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/science.json  \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task6.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 6


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/sports.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task7.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/sports.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task7.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/sports.json  \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task7.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 7


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/star.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_pre/task8.json \
   0.1 \
   fixed_rank
bash scripts/MCITlib/Train/Taskn.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/star.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train/task8.json
bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/videollava.json \
   $HARD_PATH/configs/data_configs/CL-VISTA/star.json   \
   $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/train_post/task8.json \
   0.1 \
   energy
bash scripts/MCITlib/Eval/Eval_CVU.sh 8