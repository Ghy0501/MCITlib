HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Eval_benchmark/run_qa_longvideobench.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/longvideobench.json $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_mmbench.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/mmbench-video.json $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_nextqa.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/nextqa.json $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_mmvu.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/mmvu.json $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_benchmark_mvbench.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/mvbench.json $HARD_PATH/configs/train_configs/HiDe/Video-LLaVA/CL-VISTA/eval/task8.json
