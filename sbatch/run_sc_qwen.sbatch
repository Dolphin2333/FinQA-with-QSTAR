#!/bin/bash
#SBATCH --job-name=qwen_sc_full
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --output=logs/qwen_sc_full.out
#SBATCH --error=logs/qwen_sc_full.err

cd /scratch/hz3916/FinQA-with-QSTAR
source .venv/bin/activate

export HF_HOME=/scratch/hz3916/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/hz3916/FinQA-with-QSTAR

python scripts/run_self_consistency.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --split test \
  --num-sequences 5 \
  --output outputs/qwen2_5_7b_self_consistency_full.json
