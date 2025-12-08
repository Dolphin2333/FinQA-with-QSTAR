#!/bin/bash
#SBATCH --job-name=finqa-qwen-test1000
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1           
#SBATCH --time=01:00:00             
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
echo "Starting test100 on Qwen/Qwen2.5-7B-Instruct..."

python /scratch/hz3916/FinQA-with-QSTAR/scripts/run_baseline.py \
  --dataset-dir data \
  --split test_100 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 4000 \
  --output outputs/Qwen7b-Instruct-test_100.json

echo "Job finished."