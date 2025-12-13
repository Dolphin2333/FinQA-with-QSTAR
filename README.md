# Improving financial numerical reasoning without expert data 📊💡

This repository accompanies an empirical study on **data-free methods for financial numerical reasoning** on the FinQA benchmark. We systematically investigate how much of the performance gap between domain-specialized models and general-purpose LLMs can be recovered without additional expert-annotated training data.

Our analysis focuses on three approaches:
1. (User-end) Prompt Engineering
2. (Inference-time) Self-Consistency
3. (Fine-tuning) Quiet Self-Taught Reasoner (QSTAR)

Experiments are conducted using Fin-R1 and Qwen-2.5-7B-Instruct.

## 🔍 Project Overview

Financial numerical reasoning is a core skill in real-world financial analysis, yet remains challenging for LLMs. While Fin-R1 achieves state-of-the-art results on FinQA, it relies on costly expert-annotated chain-of-thought data and improves accuracy by only ~4% over strong general-purpose models. This project asks:

> **How much of Fin-R1’s performance gain can be recovered using data-free methods alone?**

To answer this, we evaluate three approaches: Prompt Engineering, Self-Consistency, and Quiet Self-Taught Reasoner. Our findings show that:
- Prompt engineering substantially narrows the reported gap between Fin-R1 and general-purpose models
- Self-consistency improves accuracy by ~5.8% in general-purpose LLMs without increasing inference time
- Quiet-STaR is computationally prohibitive under our budgets and does not yield favorable inference trade-offs

## 📄 Report and Compiled Results

A full PDF report is provided at [`Final_Report.pdf`](./Final_Report.pdf), containing detailed experimental setup, analysis, and discussion of results.

All main experimental results are compiled under the CSV files in [`outputs/`](./outputs/). These CSV files contain the finalized numbers used in the report and are the recommended entry point for downstream analysis or plotting.

## 📁 Repository Structure
```text
.
├── data/                      
│   ├── train.json                      # FinQA dataset files (train/dev/test)
│   ├── dev.json
│   ├── test.json
|   └── prompts.csv                     # 10 prompts used for prompt engineering
│
├── outputs/                   
│   ├── prompt_engineering_results.csv  # Predictions and aggregated results
│   ├── self_consistency_results.csv
│   ├── quiet_star_results.csv
│   ├── FinR1-full-test.json            # Output file when run evaluations
│   └── Qwen7b-Instruct-full-test.json
│
├── scripts/                            # Main entry-point scripts
│   ├── run_baseline.py
│   ├── run_self_consistency.py
│   ├── evaluate_json.py
│   ├── train_qstar.py
│   └── evaluate_checkpoints.py
│
├── src/                                # Core implementation
│   ├── load_data.py
│   ├── load_model.py
│   ├── infer.py
│   ├── eval_finqa.py
│   ├── table_utils.py
│   └── modeling_qwen2_qstar.py
│
├── requirements.txt
├── Final_Report.pdf
└── README.md
```

## 🌲 Environment setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Dolphin2333/FinQA-with-QSTAR.git
cd FinQA-with-QSTAR
```

### 2️⃣ Create a Python Environment

All package versions are specified in `requirements.txt` for reproducibility.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To ensure the Python can import from `src.*`, run
```bash
export PYTHONPATH=$(pwd)
```

### 3️⃣ Prepare the FinQA Dataset

FinQA dataset has already been copied to the [`data/`](data/) folder. Alternatively, you can download `train.json`, `dev.json`, and `test.json` from the official FinQA repository:

👉 [https://github.com/czyssrs/FinQA](https://github.com/czyssrs/FinQA)

and place them under [`data/`](data/).

## 🏃‍♀️ Running experiments

### 📝 Prompt engineering

```bash
python scripts/run_baseline.py \
    --dataset-dir data \
    --split test \
    --model-name Qwen/Qwen2.5-7B-Instruct \   # or SUFE-AIFLM-Lab/Fin-R1
    --prompt-id 7 \                           # One of the 10 prompts in data/prompts.csv
    --max-new-tokens 4000 \
    --temperature 0.7 \
    --top-p 0.8 \
    --seed 42 \                               # Random seed for reproducibility
    --limit 100  \                            # Remove this option if you want to evaluate on the whole test dataset
    --output outputs/finr1-test.json
```

This code evaluate the two baseline models using different prompt designs from [`data/prompts.csv`](./data/prompts.csv).

⚠️ The first run downloads ~14 GB of model weights. Use a GPU node with at least 24 GB VRAM.

**If you are working on NYU HPC:** You can directly modify the command in [`sbatch/run_finr1.sbatch`](./sbatch/run_finr1.sbatch) or [`sbatch/run_qwen_instruct.sbatch`](./sbatch/run_qwen_instruct.sbatch) and send slurm job
```bash
sbatch sbatch/run_finr1.sbatch
```
or
```bash
sbatch sbatch/run_qwen_instruct.sbatch
```


If you already saved your output at `outputs/finr1-test.json`, you can run only the evaluation script
```bash
python scripts/evaluate_json.py --json-file outputs/finr1-test.json
```

The results for prompt engineering experiments are compiled at [`outputs/prompt_engineering_results.csv`](./outputs/prompt_engineering_results.csv).



### 🗳️ Self-consistency

```bash
python scripts/run_self_consistency.py \
    --dataset-dir data \
    --split test \
    --model-name Qwen/Qwen2.5-7B-Instruct \   # or SUFE-AIFLM-Lab/Fin-R1
    --num-sequences 9 \                       # Number of generations to be sampled in parallel
    --temperature 1.0 \                       # Increase temperature to increase sampling diversity
    --top-p 0.6 \                             # Increase top-p to increase sampling diversity
    --max-new-tokens 4000 \
    --prompt-id 7 \                           # One of the 10 prompts in data/prompts.csv
    --seed 42 \                               # Random seed for reproducibility
    --limit 100  \                            # Remove this option if you want to evaluate on the whole test dataset
    --output outputs/qwen-self-consistency.json
```

This script samples multiple reasoning paths per question and performs majority voting over final boxed answers.

**If you are working on NYU HPC:** You can directly modify the command in [`sbatch/run_sc_qwen.sbatch`](./sbatch/run_sc_qwen.sbatch) and send slurm job
```bash
sbatch sbatch/run_sc_qwen.sbatch
```

Similar to prompt engineering experiments, if you already saved your output at `outputs/qwen-self-consistency.json`, you can run only the evaluation script
```bash
python scripts/evaluate_json.py --json-file outputs/qwen-self-consistency.json
```

The results for self-consistency experiments are compiled at [`outputs/self_consistency_results.csv`](./outputs/self_consistency_results.csv).

### 🤔 QSTAR

Training script
```bash
python scripts/train_qstar.py \
    --dataset-dir data \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --n-ahead 8 \                               # Number of thoughts
    --n-ahead-talk 4 \                          # Length of thoughts
    --full-batch-size 1 \
    --gradient-accumulation-steps \             # Keep effective batch size = 8
    --max-steps 1000 \                          # Number of training steps
    --eval-steps 100 \
    --logging-steps 100 \
    --save-steps 100 \
    --learning-rate 1e-6 \                      # Hyperparameters used in the original paper
    --warmup-steps 20 \
    --weight-decay 0.001 \
    --prompt-id 7 \                             # One of the 10 prompts in data/prompts.csv
    --seed 42 \                                 # Random seed for reproducibility           
    --root-prefix ..                            # Where to save the model
```

⚠️ Training requires at least 2 80GB A100 GPUs and takes about 12 hours.

**If you are working on NYU HPC:** You can directly modify the command in [`sbatch/run_qstar.sbatch`](./sbatch/run_qstar.sbatch) and send slurm job
```bash
sbatch sbatch/run_qstar.sbatch
```

Evaluation script
```bash
python scripts/run_self_consistency.py \
    --dataset-dir data \
    --split test \
    --model-name Qwen/Qwen2.5-7B-Instruct \   # or SUFE-AIFLM-Lab/Fin-R1
    --n-ahead 8 \                             # Number of thoughts
    --n-ahead-talk 4 \                        # Length of thoughts
    --peft-dir ../.cache/qstar/1764960658     # Where to LoRA checkpoints are saved
    --prompt-id 7 \                           # One of the 10 prompts in data/prompts.csv
    --max-new-tokens 4000 \
    --temperature 0.7 \
    --top-p 0.8 \
    --seed 42 \                               # Random seed for reproducibility
    --limit 100  \                            # Remove this option if you want to evaluate on the whole test dataset
    --output outputs/qwen-qstar.json
```

The results for QSTAR experiments are compiled at [`outputs/quiet_star_results.csv`](./outputs/quiet_star_results.csv).

## 🧩 Core Code Components

- [`src/load_data.py`](./src/load_data.py) - FinQA parsing and normalization
- [`src/load_model.py`](./src/load_model.py) - Hugging Face model and tokenizer loading
- [`src/infer.py`](./src/infer.py) - Prompt construction and generation logic
- [`src/eval_finqa.py`](./src/eval_finqa.py) - Numeric-aware evaluation and answer matching
- [`src/table_utils.py`](./src/table_utils.py) - Table-to-text conversion for prompts
- [`src/modeling_qwen2_qstar.py`](./src/modeling_qwen2_qstar.py) - Patch the base Qwen2 implementation in Huggingface to add QSTAR



## References

Chen, Zhiyu, et al. "Finqa: A dataset of numerical reasoning over financial data." Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021.

Liu, Zhaowei, et al. "Fin-r1: A large language model for financial reasoning through reinforcement learning." arXiv preprint arXiv:2503.16252 (2025).

Wang, Xuezhi, et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." The Eleventh International Conference on Learning Representations.

Zelikman, Eric, et al. "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking." First Conference on Language Modeling.