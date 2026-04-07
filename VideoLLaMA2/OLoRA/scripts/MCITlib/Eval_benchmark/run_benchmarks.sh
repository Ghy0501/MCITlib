HARD_PATH=/your_path/MCITlib_v3
export CUDA_VISIBLE_DEVICES=0
bash scripts/MCITlib/Eval_benchmark/run_longvideobench.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/longvideobench.json $HARD_PATH/configs/train_configs/OLoRA/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_mvbench.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/mvbench.json $HARD_PATH/configs/train_configs/OLoRA/VideoLLaMA2/CL-VISTA/eval/task8.json
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/MCITlib/Eval_benchmark/run_qa_mmbench.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/mmbench-video.json $HARD_PATH/configs/train_configs/OLoRA/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_mmvu.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/mmvu.json $HARD_PATH/configs/train_configs/OLoRA/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_nextqa.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/nextqa.json $HARD_PATH/configs/train_configs/OLoRA/VideoLLaMA2/CL-VISTA/eval/task8.json
