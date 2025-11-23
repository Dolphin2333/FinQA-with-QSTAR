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

## Generation Settings (same for both models)
max_new_tokens=4000, temperature=0.7, top_p=0.8, repetition_penalty=1.05


## P1 Evaluation Summary (20 Samples)

| **Model**               | **Accuracy** | **# Correct / # Total** | **Notes** |
|-------------------------|--------------|--------------------------|-----------|
| **FinR1 (SUFE Fin-R1)** | **70%**      | **14 / 20**              | Strong performance, stable formatting, no hallucinated outputs. |
| **Qwen2.5-7B-Instruct** | **60%**      | **16 / 20**              | Performs well but lower than earlier runs; prefers clear tag structure. |

## Interpretation of P1 Results

The P1 prompt continues to be a strong baseline for both models. FinR1 remains highly stable at 70%, showing it handles clean reasoning prompts extremely well and rarely drifts outside the expected answer.

Qwen achieved 60% on this controlled run, which is lower than previous exploratory runs but still solid. Qwen clearly prefers structured tag-based prompts, and P1 aligns well with that preference. Its numerical extraction is generally good, and formatting is consistent.

Overall, P1 remains a reliable baseline prompt to compare other variants against.


# P2 — Structured Financial Extraction Prompt

## Purpose:

A more explicit, structured prompt designed to enforce concise reasoning + strict numeric extraction, inspired by:

FinR1 style instructions (GitHub)

Aiera FinQA evaluation guidelines

Programmer.ie FinR1 prompt analysis

This prompt removes tags like <think> and instead emphasizes clear extraction → short reasoning → boxed answer.

## Exact Prompt Used:

### System Prompt

SYSTEM_PROMPT_P2 = """You are a financial reasoning assistant.
Carefully extract all relevant numbers from the provided text and tables.
Perform the necessary calculations step-by-step.
Then return ONLY the final numeric answer inside a single LaTeX \\boxed{}.
Keep your reasoning concise and strictly tied to the context."""

### Task Prompt

TASK_PROMPT_P2 = """Using the financial context, identify the required values and compute the correct final answer."""

### Answer Prompt

ANSWER_FORMAT_P2 = """Explain briefly how you arrived at the answer.
Then provide only the final result:
\\boxed{FINAL_ANSWER}
"""

## Generation Settings (same for both models)

max_new_tokens = 4000
temperature = 0.7
top_p = 0.8
repetition_penalty = 1.05

## P2 Evaluation Summary (20 Samples)

| **Model**               | **Accuracy** | **# Correct / # Total** | **Notes** |
|-------------------------|--------------|--------------------------|-----------|
| **FinR1 (SUFE Fin-R1)** | **70%**      | **14 / 20**              | Very stable; responds well to structured extraction instructions. |
| **Qwen2.5-7B-Instruct** | **50%**      | **10 / 20**              | Accuracy drops vs P1; Qwen prefers explicit tags and CoT scaffolding. |

## Interpretation

FinR1 stays very stable at 70% → robust to prompt P2.

Qwen dropped from 70% → 50% → P2 harms Qwen because P2 forces a more structured CoT + explanation + formatted answer block, which Qwen doesn’t follow consistently.

# P3 — Structured Extractive Reasoning Prompt

## Purpose

A more structured “extract → calculate → answer” prompt inspired by:

FinR1 GitHub’s original instruction style

Aiera FinQA benchmarking article

Programmer.ie explanations on extract–operate–answer reasoning

The goal of P3 is to force the model into a more deterministic, step-organized reasoning pattern while still returning a final boxed answer.

## Exact Prompt Used (inside infer.py)

### System Prompt

SYSTEM_PROMPT_P3 = """You are a financial QA assistant.
Break down the problem into structured steps:
1. Extract all relevant numbers from the context.
2. Identify required operations.
3. Perform calculations carefully.
After that, return ONLY the final result in one LaTeX box."""


TASK_PROMPT_P3 = """Extract numbers, outline the steps, compute carefully, and ensure a consistent numeric answer."""

ANSWER_FORMAT_P3 = """
<think>
Step 1: Extract key numbers.
Step 2: Identify the correct operations.
Step 3: Perform calculations.
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""

## Generation Settings

max_new_tokens=4000
temperature=0.7
top_p=0.8
repetition_penalty=1.05

## P3 Evaluation Summary (20 Samples)


| **Model**               | **Accuracy** | **# Correct / # Total** | **Notes** |
|-------------------------|--------------|--------------------------|-----------|
| **FinR1 (SUFE Fin-R1)** | **70%**      | **14 / 20**              | Stable across P1, P2, P3. Consistent reasoning style; prompt structure does not significantly affect performance. |
| **Qwen2.5-7B-Instruct** | **60%**      | **12 / 20**              | Better than P2 (50%), worse than P1 (80%). Handles structured prompts moderately well but prefers flexible CoT formats. |

## Interpretation

FinR1 again remains extremely stable at 70% across P1, P2, and P3 — confirming that its training style is deeply aligned with structured financial reasoning.

Qwen improves from 50% (P2) to 60% (P3). The structured steps in P3 help it focus better than the free-form extraction in P2, but it still performs best with the more lightweight structure of P1.

Qwen clearly prefers prompts that give it flexibility in reasoning, while FinR1 remains remarkably robust across all tested prompt formats.