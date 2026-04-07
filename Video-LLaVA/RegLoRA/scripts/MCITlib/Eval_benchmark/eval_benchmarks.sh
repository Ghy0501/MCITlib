HARD_PATH=/your_path/MCITlib_v3
bash scripts/MCITlib/Eval_benchmark/eval_benchmark_mvbench.sh $HARD_PATH/configs/train_configs/RegLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_longvideobench.sh $HARD_PATH/configs/train_configs/RegLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_mmbench.sh $HARD_PATH/configs/train_configs/RegLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_nextqa.sh $HARD_PATH/configs/train_configs/RegLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/eval_qa_mmvu.sh $HARD_PATH/configs/train_configs/RegLoRA/Video-LLaVA/CL-VISTA/eval/task8.json