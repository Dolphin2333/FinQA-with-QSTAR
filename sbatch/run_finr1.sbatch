#!/bin/bash
#SBATCH --job-name=finqa-finr1
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --logs/%x-%j.out
#SBATCH --logs/%x-%j.err


# cd my project path
cd /scratch/$USER/FinQA-with-QSTAR
mkdir -p logs outputs
source .venv/bin/activate

export HF_HOME=/scratch/$USER/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/$USER/FinQA-with-QSTAR

# --- Run Full Test ---
echo "Starting full test on SUFE-AIFLM-Lab/Fin-R1..."

python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --output outputs/finr1-test100.json

echo "Job finished."
