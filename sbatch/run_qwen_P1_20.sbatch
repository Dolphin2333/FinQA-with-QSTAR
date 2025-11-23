#!/bin/bash
#SBATCH --job-name=qwen-P1-20
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=/scratch/ssn9062/FinQA-with-QSTAR/logs/%x-%j.out
#SBATCH --error=/scratch/ssn9062/FinQA-with-QSTAR/logs/%x-%j.err

cd /scratch/ssn9062/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export HF_HOME=/scratch/ssn9062/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/ssn9062/FinQA-with-QSTAR

python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --limit 20 \
  --max-new-tokens 4000 \
  --output outputs/qwen7B_P1_20.json
