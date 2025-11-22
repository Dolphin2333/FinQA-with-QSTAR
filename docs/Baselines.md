# P1 — Unified Financial Reasoning Prompt

## Purpose:
A simplified financial reasoning prompt used for both FinR1 and Qwen in order to compare model performance on the same instruction set.

## Exact Prompt Used:

### System Prompt

SYSTEM_PROMPT_P1 = """You are a financial reasoning assistant.
Think step-by-step inside <think>...</think>.
Use the given financial context (text + tables) to extract numbers and perform calculations.
After reasoning, output ONLY the final numeric answer inside a single LaTeX box.
"""

### Task Prompt

TASK_PROMPT_P1 = """Use the context to identify relevant values and compute the correct final answer."""

### Answer Prompt

ANSWER_FORMAT_P1 = """
<think>
(your reasoning steps here)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""

### Generation Settings (same for both models)
max_new_tokens=4000, temperature=0.7, top_p=0.8, repetition_penalty=1.05


## P1 Evaluation Summary (20 Samples)

| **Model**               | **Accuracy** | **# Correct / # Total** | **Notes** |
|-------------------------|--------------|--------------------------|-----------|
| **FinR1 (SUFE Fin-R1)** | **70%**      | **14 / 20**              | Strong performance, stable formatting, no hallucinated outputs. |
| **Qwen2.5-7B-Instruct** | **80%**      | **16 / 20**              | Best-performing baseline; consistently follows <think>…</think> + \boxed{} format. |
