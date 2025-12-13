#!/bin/bash
#SBATCH --job-name=qwen_sc
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/qwen_sc_full.out
#SBATCH --error=logs/qwen_sc_full.err


# --- Setup ---
cd /scratch/$USER/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export HF_HOME=/scratch/$USER/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/$USER/FinQA-with-QSTAR

# --- Run Self-consistency Test ---
echo "Starting self-consistency test on Qwen/Qwen2.5-7B-Instruct..."

python scripts/run_self_consistency.py \
  --dataset-dir data \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --num-sequences 5 \
  --output outputs/qwen2_5_7b_self_consistency_full.json

echo "Job finished."