# HPC Baseline Quickstart

This note assumes you have already finished the baseline code locally and pushed it to GitHub.

## 1. Push from local
- `git add .`
- `git commit -m "Baseline scaffold"`
- `git push origin main`

## 2. Clone on NYU HPC
```bash
ssh <netid>@hpc.nyu.edu
cd /scratch/<netid>              # or your project space
git clone https://github.com/Dolphin2333/FinQA-with-QSTAR.git
cd FinQA-with-QSTAR
```

If HTTPS is blocked, set up SSH keys and clone via `git@github.com:...`.

## 3. Create Python environment
```bash
module load anaconda3            # or the cluster’s preferred Python module
conda create -n finqa python=3.10 -y
conda activate finqa
pip install -r requirements.txt
```

Optional: set `HF_HOME=/scratch/<netid>/hf_cache` to control the Hugging Face cache location.

## 4. Prepare FinQA dataset
```bash
mkdir -p data/FinQA
# copy train.json / dev.json / test.json from the official repo
scp <local_path>/FinQA/*.json <netid>@hpc.nyu.edu:/scratch/<netid>/FinQA-with-QSTAR/data/FinQA/
```

## 5. Run the baseline script
```bash
python scripts/run_baseline.py \
  --dataset-dir data/FinQA \
  --split dev \
  --model-name Zyphra/FinR1-7B \
  --max-new-tokens 64 \
  --limit 50 \
  --output outputs/finr1-dev.json
```

Remove `--limit` once you verify everything works. The first run will download ~14 GB model weights; ensure you reserved a GPU node with ≥24 GB VRAM or adjust `torch_dtype`/`device_map` in `src/load_model.py`.

## 6. Alternative: serve Fin-R1 with vLLM
If you need to expose an inference endpoint instead of running the baseline script:
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

Example client (from a login or compute node):
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

Remember to shut down the service once you finish (`Ctrl+C` on the serving process).
