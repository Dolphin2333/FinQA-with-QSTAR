# HPC Baseline Quickstart

If you are working directly on the HPC (no local changes), you can skip any push/pull steps.

## 1. Clone the repository on NYU HPC

```bash
ssh <netid>@hpc.nyu.edu                # log in to the cluster
cd /scratch/<netid>                    # switch to your scratch space
git clone https://github.com/Dolphin2333/FinQA-with-QSTAR.git
cd FinQA-with-QSTAR
```

If HTTPS is blocked, configure SSH keys and clone via `git@github.com:...`.

## 2. Create a Python environment

```bash
python3 -m venv .venv                  # create a virtualenv (since no conda module is available)
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: set `HF_HOME=/scratch/<netid>/hf_cache` to control the Hugging Face cache location before running the script.

## 3. Prepare the FinQA dataset

Download `train.json`, `dev.json`, and `test.json` from the official FinQA repository (<https://github.com/czyssrs/FinQA>) and upload/copied them into `data/` on the HPC node.

```bash
mkdir -p data
scp <local_path>/FinQA/*.json <netid>@hpc.nyu.edu:/scratch/<netid>/FinQA-with-QSTAR/data/
```

If you cloned the FinQA repo directly on the HPC, simply copy the JSON files from `FinQA/dataset/` into `data/`.

## 4. Run the baseline script

```bash
export PYTHONPATH=$(pwd)               # ensure Python can import src/*
python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --limit 100 \
  --output outputs/finr1-test.json
```

If you hit `ModuleNotFoundError: No module named 'src'`, re-run the `export PYTHONPATH=$(pwd)` step and launch the script again.

Remove `--limit` once you verify the pipeline works. The first run downloads ~14 GB of weights, so reserve a GPU node with ≥24 GB of VRAM (or tweak `torch_dtype`/`device_map` in `src/load_model.py`).

If you want to run the baseline for other model, e.g. Qwen2.5-7B or Qwen/Qwen2.5-7B-Instruct, run:

```bash
export PYTHONPATH=$(pwd)

#  use --limit 5 to quickly test
python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name Qwen/Qwen2.5-7B \
  --max-new-tokens 4000 \
  --limit 5 \
  --output outputs/Qwen7b-test5.json
```


### Sample artifacts: `test_100` dataset and example outputs

To make quick sanity checks easier, this repo includes references to small
sample files built from the first 100 test items:

- `data/test_100.json`: a lightweight subset of `test.json` for fast
  local/HPC smoke tests without running the full evaluation.
- `outputs/*test_100*.json`: example prediction files produced by the
  baseline on the `test_100` subset. These serve as format references and
  expected output examples; you can safely delete/regenerate them.

These samples are purely for demonstration and quick iteration. The main
pipeline still targets the full `test.json` split unless you explicitly point
`--split` to a `test_100`-style file.

Note on outputs: if additional files are added under `outputs/`, please include
in this document (or a nearby README) a short note of the exact command or
script used to produce them (e.g., the `scripts/run_baseline.py` invocation or
the SLURM job snippet). This helps others reproduce results consistently.

#### How `test_100` was created

For transparency and reproducibility, the `test_100.json` subset was created
by shuffling the official `test.json` with a fixed random seed and then taking
the first 100 items:

```bash
python - << 'EOF'
import json, random
random.seed(3916)
data = json.load(open("data/test.json"))
random.shuffle(data)
json.dump(data[:100], open("data/test_100.json", "w"), indent=2)
EOF
```




## Code Structure

Overview of key modules under `src/` for quick orientation:

- `scripts/run_baseline.py`: Baseline entry point
  - Parses CLI args: `--dataset-dir` (required), `--split` (default: `test`), `--model-name` (default: Fin-R1), `--max-new-tokens` (default: 4000), `--temperature` (default: 0.7), `--top-p` (default: 0.8), `--limit` (optional), and `--output` (optional path for JSON outputs).
  - Orchestrates the full pipeline: loads data via `load_finqa_split`, loads the model with `load_baseline`, runs generation with `run_inference`, computes accuracy with `compute_accuracy`, and optionally writes raw generations and a detailed predictions JSON to `--output`.

- `src/load_model.py`: Model and tokenizer loading
  - Defaults to `SUFE-AIFLM-Lab/Fin-R1`, returns `(model, tokenizer)`, and sets pad token/padding-side compatibility.

- `src/load_data.py`: FinQA data loading and normalization
  - Reads samples from `data/{train,dev,test}.json`, builds `FinQASample` (question, answer, program, table, context, etc.), and provides `iter_answers` for gold answers.

- `src/table_utils.py`: Table-to-text conversion
  - Converts header + rows into concise sentences for prompt context (e.g., "For X, the Y is Z.").

- `src/infer.py`: Inference and prompt construction
  - `build_prompt` assembles context and table text, instructing the model to return the final answer in `\\boxed{}`;
  - `BoxedStoppingCriteria` truncates generation to avoid trailing text after the boxed answer;
  - `run_inference` batches generation with temperature, top-p, and repetition penalty.

- `src/eval_finqa.py`: Evaluation utilities and matching logic
  - Performs numeric/text matching and accuracy computation; supports stripping currency symbols, percent signs, thousand separators, parentheses-negative format, and magnitude alignment;
  - `compute_accuracy` returns accuracy, parsed predictions, and a match mask.

If you want these notes mirrored in the README or expanded with function-level descriptions, add docstrings within each module as needed.





-----

## 3.5 ✂️ Data Sharding and Parallel Test Setup

To facilitate **parallel testing** on the HPC, we will split the remaining FinQA test data into smaller shards.

### A. Create 10 Data Shards

Run the following Python script to exclude the first 100 used samples (`test_100.json`) from `test.json` and create 10 new files, each containing 100 randomly selected samples (`test_100_shard0.json` to `test_100_shard9.json`).

```bash
python - <<'PY'
import json, random, math

random.seed(3916)
all_data = json.load(open("data/test.json"))
used_100 = set(x["id"] for x in json.load(open("data/test_100.json")))

remaining = [x for x in all_data if x["id"] not in used_100]
random.shuffle(remaining)

shard_size = 100
num_shards = math.ceil(len(remaining)/shard_size)
for i in range(min(10, num_shards)):  # Generate 10 shards of 100 samples each
    shard = remaining[i*shard_size:(i+1)*shard_size]
    with open(f"data/test_100_shard{i}.json","w") as f:
        json.dump(shard,f,indent=2)
print("done:", [f"data/test_100_shard{i}.json" for i in range(min(10,num_shards))])
PY
```

### B. Parallel Test SLURM Script (`sbatch_src/run_test_shards.sbatch`)

Create the following SLURM script. It uses the `$SLURM_ARRAY_TASK_ID` variable to dynamically select and process the corresponding shard file.

```bash
#!/bin/bash
#SBATCH --job-name=finqa-shards
#SBATCH --account=csci_ga_3033_09-2025fa
#SBATCH --partition=c12m85-a100-1   # Assuming A100 node partition
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00             # Time limit
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --array=0-9                 # Task Array from 0 to 9 (corresponds to shard0 through shard9)
#SBATCH --output=logs/%x-%A_%a.out  # Output file: logs/finqa-shards-JobID_ArrayTaskID.out
#SBATCH --error=logs/%x-%A_%a.err   # Error file

# Ensure logs and outputs directories exist
mkdir -p logs outputs
# Change to the project root directory
cd /scratch/<netid>/FinQA-with-QSTAR
source .venv/bin/activate

# Environment Variables
export HF_HOME=/scratch/<netid>/.cache/huggingface # Adjust path as necessary
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(pwd) # Ensure Python can import src/*

# Dynamically set input split and output filename based on array task ID
SPLIT="test_100_shard${SLURM_ARRAY_TASK_ID}"
OUT="outputs/finr1-${SPLIT}.json"

python scripts/run_baseline.py \
  --dataset-dir data \
  --split "${SPLIT}" \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --output "${OUT}"
```


## 4\. ▶️ Run the Baseline Script

### A. Run a Single Shard for Verification

Before running all tasks, it is highly recommended to run one task (e.g., array index 1) to verify the pipeline works correctly.

```bash
# Run Array Task ID 1 (which processes test_100_shard1.json)
sbatch --array=1 sbatch_src/run_test_shards.sbatch
```

### B. Run All 10 Shards (Parallel Testing)

Once verified, submit the SLURM script without the `--array` flag to launch all 10 tasks in parallel, as defined by the `#SBATCH --array=0-9` line in the script.

```bash
sbatch sbatch_src/run_test_shards.sbatch
```

### C. Run the Full Test Split (Non-Parallel)

If you only need to run the full `test.json` (without sharding), submit the following job (ensure it's run on a GPU node via an `sbatch` script):

```bash
export PYTHONPATH=$(pwd)
python scripts/run_baseline.py \
  --dataset-dir data \
  --split test \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --output outputs/finr1-test.json
```

**Note:** The first run downloads \~14 GB of weights. Reserve a GPU node with $\ge 24$ GB of VRAM.
