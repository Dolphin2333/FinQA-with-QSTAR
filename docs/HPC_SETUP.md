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

Download `train.json`, `dev.json`, and `test.json` from the official FinQA repository (<https://github.com/czyssrs/FinQA>) and upload/copied them into `data/FinQA/` on the HPC node.

```bash
mkdir -p data/FinQA
scp <local_path>/FinQA/*.json <netid>@hpc.nyu.edu:/scratch/<netid>/FinQA-with-QSTAR/data/FinQA/
```

If you cloned the FinQA repo directly on the HPC, simply copy the JSON files from `FinQA/dataset/` into `data/FinQA/`.

## 4. Run the baseline script

```bash
export PYTHONPATH=$(pwd)               # ensure Python can import src/*
python scripts/run_baseline.py \
  --dataset-dir data/FinQA \
  --split test \
  --model-name SUFE-AIFLM-Lab/Fin-R1 \
  --max-new-tokens 4000 \
  --limit 100 \
  --output outputs/finr1-test.json
```

If you hit `ModuleNotFoundError: No module named 'src'`, re-run the `export PYTHONPATH=$(pwd)` step and launch the script again.

Remove `--limit` once you verify the pipeline works. The first run downloads ~14 GB of weights, so reserve a GPU node with ≥24 GB of VRAM (or tweak `torch_dtype`/`device_map` in `src/load_model.py`).

## 5. Optional: serve Fin-R1 with vLLM

To expose the model as an inference endpoint instead of running the baseline script:

```bash
module load anaconda3
conda create -n finr1-serve python=3.10 -y
conda activate finr1-serve
pip install vllm

git lfs install
git clone https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1
vllm serve "/scratch/<netid>/Fin-R1" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --tensor-parallel-size 2 \
  --served-model-name "Fin-R1"
```

Example client (run on a login/compute node):

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://0.0.0.0:8000/v1")
prompt = "请判断下列描述是否符合金融与信息合规性..."

resp = client.chat.completions.create(
    model="Fin-R1",
    messages=[
        {"role": "system", "content": "You are a helpful AI Assistant..."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=4000,
    extra_body={"repetition_penalty": 1.05},
)
print(resp)
```

Terminate the vLLM service when finished (`Ctrl+C` in the terminal running `vllm serve`).
