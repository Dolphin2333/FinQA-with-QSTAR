#!/bin/bash
#SBATCH --job-name=finqa-qwen-full
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1           
#SBATCH --time=04:00:00             
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/scratch/hz3916/FinQA-with-QSTAR/logs/%x-%j.out
#SBATCH --error=/scratch/hz3916/FinQA-with-QSTAR/logs/%x-%j.err


# --- Setup ---
cd /scratch/hz3916/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export HF_HOME=/scratch/hz3916/.cache/huggingface  
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/hz3916/FinQA-with-QSTAR # make sure Python is able to load src/*

# --- Run Full Test ---
echo "Starting full test on Qwen/Qwen2.5-7B-Instruct..."

python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 4000 \
  --output outputs/Qwen7b-Instruct-full-test.json

echo "Job finished."