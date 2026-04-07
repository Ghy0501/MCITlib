# SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning
This is the official code implementation of "[SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.pdf)".

![main_pic](imgs/fig_main.png) 



## Benchmark install
The CVIT benchmark we have constructed encompasses 10 datasets along with their corresponding instruction sets.
### Instruction Tuning Files
You can download instruction tuning files of our CVIT benchmark from [CVIT benchmark](https://huggingface.co/datasets/zackie29/SMoLoRA).

### Dataset Images
All datasets used in the benchmark are publicly available. You can download the corresponding images directly from each dataset’s official website.

## Training and Evaluation
### Install
```
git clone https://github.com/Minato-Zackie/SMoLoRA.git
cd ./SMoLoRA
conda create -n smolora python=3.10 -y
conda activate smolora
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
For installing FlashAttention, version conflicts may cause the installation to fail. We therefore recommend manually downloading the appropriate wheel package [flash_attn-2.5.8+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64](https://github.com/Dao-AILab/flash-attention/releases?page=3) and installing it yourself.

### Model Preparation
Please download the pretrained language model [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and the [alignment module](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5) in advance.

### Training and Eval
Run all training and evaluation procedures:
```
bash ./scripts/SMoLoRA/Train_Eval/Train_all.sh
```
Evaluate the MIF metric:
```
bash ./scripts/SMoLoRA/Eval_IF/eval_if.sh
```

## Acknowledgement
Our project is based on [LLaVA](https://github.com/haotian-liu/LLaVA) and [CoIN](https://github.com/zackschen/CoIN). We sincerely thank them for their outstanding contributions.
