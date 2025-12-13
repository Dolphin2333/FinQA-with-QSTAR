#!/bin/bash
#SBATCH --job-name=qwen_qstar
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-2   
#SBATCH --gres=gpu:a100:2      
#SBATCH --time=12:00:00             
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# --- Setup ---
cd /scratch/$USER/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export HF_HOME=/scratch/$USER/.cache/huggingface  
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/$USER/FinQA-with-QSTAR

# --- Train QSTAR Test ---
echo "Training Quiet-STAR version of Qwen/Qwen2.5-7B-Instruct..."

python scripts/train_qstar.py \
  --n-ahead 8 \
  --n-ahead-talk 4 \
  --max-steps 1000 \
  --eval-steps 100 \
  --logging-steps 100 \
  --save-steps 100 \
  --learning-rate 1e-6 \
  --warmup-steps 20 \
  --weight-decay 0.001

echo "Job finished."