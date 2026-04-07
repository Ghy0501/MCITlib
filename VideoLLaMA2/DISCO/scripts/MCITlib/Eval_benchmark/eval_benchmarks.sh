HARD_PATH=/your_path/MCITlib_v3
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/MCITlib/Eval_benchmark/eval_longvideobench.sh $HARD_PATH/configs/train_configs/DISCO/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_mvbench.sh $HARD_PATH/configs/train_configs/DISCO/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_mmvu.sh $HARD_PATH/configs/train_configs/DISCO/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_mmbench.sh $HARD_PATH/configs/train_configs/DISCO/VideoLLaMA2/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_nextqa.sh $HARD_PATH/configs/train_configs/DISCO/VideoLLaMA2/CL-VISTA/eval/task8.json
