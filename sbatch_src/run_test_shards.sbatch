#!/bin/bash
#SBATCH --job-name=finqa-shards
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

mkdir -p logs outputs
cd /scratch/hz3916/FinQA-with-QSTAR
source .venv/bin/activate

export HF_HOME=/scratch/hz3916/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/hz3916/FinQA-with-QSTAR

SPLIT="test_100_shard${SLURM_ARRAY_TASK_ID}"      # e.g. test_100_shard0.json
OUT="outputs/finr1-${SPLIT}.json"

python scripts/run_baseline.py \
  --dataset-dir data/FinQA \
  --split "${SPLIT}" \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --output "${OUT}"
