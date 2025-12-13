#!/bin/bash
#SBATCH --job-name=finqa-qwen
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1           
#SBATCH --time=12:00:00             
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# --- Setup ---
cd /scratch/$USER/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export HF_HOME=/scratch/$USER/.cache/huggingface  
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/$USER/FinQA-with-QSTAR

# --- Run Full Test ---
echo "Starting full test on Qwen/Qwen2.5-7B-Instruct..."

python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 4000 \
  --output outputs/Qwen7b-Instruct-full-test.json

echo "Job finished."