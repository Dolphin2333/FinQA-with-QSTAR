#!/bin/bash
#SBATCH --job-name=finqa-test100
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# —— make sure the paths are created ——
mkdir -p logs
mkdir -p outputs

# cd my project path
cd /scratch/hz3916/FinQA-with-QSTAR
source .venv/bin/activate

export HF_HOME=/scratch/hz3916/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/hz3916/FinQA-with-QSTAR

# —— run test 100 -- with random seed 3916
python scripts/run_baseline.py \
  --dataset-dir data/FinQA \
  --split test_100 \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --output outputs/finr1-test100.json
