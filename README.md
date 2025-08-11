# 🌅 MCITlib: Multimodal Continual Instruction Tuning Library and Benchmark

## ✨ Introduction

Welcome to MCITlib — a comprehensive library for continual instruction tuning based on multimodal large language models. This repository continuously integrates a range of existing methods, including MoELoRA, HiDe-LLaVA, and CL-MoE. In addition, MCITlib provides evaluation results for these methods across diverse benchmarks and model architectures. Through MCITlib, we hope to foster broader interest and engagement in this promising research field.

## 📰 News

- **[2025.08.10]** Initial version of MCITlib is released. :tada:

## 🥇 Methods Provided
- `LoRA-FT`: Baseline method which simply updates LoRA parameters on new tasks. [[Paper]](https://arxiv.org/pdf/2106.09685v1/1000) ![](https://img.shields.io/badge/ICLR-2022-blue)
- `O-LoRA`: Orthogonal subspace learning for language model continual learning. [[Paper]](https://arxiv.org/pdf/2310.14152) ![](https://img.shields.io/badge/EMNLP_findings-2023-blue)
- `MoELoRA`: CoIN: A Benchmark of Continual Instruction Tuning for Multimodal Large Language Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
- `ModalPrompt`: ModalPrompt: Dual-Modality Guided Prompt for Continual Learning of Large Multimodal Models [[Paper]](https://arxiv.org/pdf/2410.05849) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- `CL-MoE`: CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering [[Paper]](https://arxiv.org/pdf/2503.00413?) ![](https://img.shields.io/badge/CVPR-2025-blue)
- `HiDe`: HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model [[Paper]](https://arxiv.org/pdf/2503.12941?) ![](https://img.shields.io/badge/ACL-2025-blue)
- `SEFE`: SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning [[Paper]](https://arxiv.org/pdf/2505.02486?) ![](https://img.shields.io/badge/ICML-2025-blue)
- `DISCO`: Federated Continual Instruction Tuning [[Paper]](https://arxiv.org/pdf/2503.12897) ![](https://img.shields.io/badge/ICCV-2025-blue)

## 🏦 Benchmarks

We currently report results on the [UCIT](https://github.com/Ghy0501/HiDe-LLaVA) and [MLLM-DCL](https://github.com/bjzhb666/MLLM-CL) benchmarks. Please refer to the provided links to download the corresponding images and instruction sets, and organize them in the following directory structure:
```
|--your_path
    |-- Domain_data
        |-- AD
        |-- Med
        |-- RS
        |-- Sci
        |-- Fin
    |-- UCIT
        |-- datasets
        |-- ArxivQA
        |-- CLEVR-Math
        |-- Flickr30k
        |-- IconQA
        |-- ImageNet-R
        |-- VizWiz
```
We also plan to extend our reproduction to other benchmarks in the near future.

## 🎨 Models

We currently provide a reproduction based on the [LLaVA-1.5-7B](https://arxiv.org/pdf/2310.03744) model. Please download it to your local directory.
```
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir /your_path/llava-v1.5-7b
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir /your_path/clip-vit-large-patch14-336
```
We also plan to extend our reproduction to other MLLM architectures in the near future.

Note: For methods such as HiDe that require loading an additional `text_tower`, please modify the `config` file in `llava-v1.5-7b` accordingly. For more details, refer to the [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA) repository.


## 🏃 How to run
### 1. Clone this repository
```
git clone https://github.com/Ghy0501/MCITlib.git
cd MCITlib
```
### 2. Install Package
```
conda create -n MCITlib python=3.10 -y
conda activate MCITlib
cd LoRA-FT
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
```
For installing [flash-attn](https://github.com/Dao-AILab/flash-attention/releases), we recommend downloading version 2.6.3 from the official repository according to your CUDA and PyTorch versions, and placing it in a local directory for manual installation. For example:
```
pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
We also provide an `environment.yaml` file to help users identify missing dependencies and version mismatches. However, due to potential library conflicts, automatic installation may fail to install certain packages. We therefore recommend manually installing them based on the provided error messages and version specifications. For essential evaluation-related dependencies, please refer to the [UCIT](https://github.com/Ghy0501/HiDe-LLaVA) and [MLLM-CL](https://github.com/bjzhb666/MLLM-CL) repositories.

### 3. Training and Evaluation

We provide predefined training and testing hyperparameters in the `configs` files within each method's directory, which can be adjusted as needed. The corresponding training and testing scripts are located in the `scripts` directory. Once all paths are correctly configured, the scripts should execute without issues. For example:
```
cd LoRA-FT
sh scripts/CoIN/Train_DCL/train_all.sh
```
The program will automatically perform both training and inference. However, for ModalPrompt, training and inference must be executed separately. Please refer to its [repository](https://github.com/AuroraZengfh/ModalPrompt) for detailed instructions.

## 🤝 Acknowledgments

We thank the following repos providing helpful functions in our work.
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [CoIN](https://github.com/zackschen/CoIN)
- [O-LoRA](https://github.com/cmnfriend/O-LoRA)
- [ModalPrompt](https://github.com/AuroraZengfh/ModalPrompt)
- [CL-MoE](https://github.com/ECNU-ICALK/CL-MoE)
- [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA)
- [SEFE](https://github.com/jinpeng0528/SEFE)
- [FCIT](https://github.com/Ghy0501/FCIT)

## 🙂 Contact

If you have any questions or suggestions for new features, please open an issue or contact the author, Haiyang Guo (guohaiyang2023@ia.ac.cn).
