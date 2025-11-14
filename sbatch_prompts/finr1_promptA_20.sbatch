#!/bin/bash
#SBATCH --job-name=finr1-promptA-20
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=/scratch/ssn9062/FinQA-with-QSTAR/logs/%x-%j.out
#SBATCH --error=/scratch/ssn9062/FinQA-with-QSTAR/logs/%x-%j.err

cd /scratch/ssn9062/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export PYTHONPATH=/scratch/ssn9062/FinQA-with-QSTAR
export HF_HOME=/scratch/ssn9062/.cache/huggingface

python scripts/run_baseline.py \
  --dataset-dir data \
  --split dev \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --limit 20 \
  --max-new-tokens 200 \
  --output outputs/finr1_promptA_20.json
