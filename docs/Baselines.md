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
| **Qwen2.5-7B-Instruct** | **80%**      | **16 / 20**              | Best-performing baseline; consistently follows <think>…</think> + \boxed{} format. |

## Interpretation of P1 Results

The P1 prompt turned out to be a very solid baseline for both models.
FinR1 reached 70% accuracy, which shows that it performs well when the instructions are simple, structured, and not overloaded with too many constraints. The model clearly benefits from the clean <think> and <answer> format, and it follows the boxed-answer requirement without drifting or hallucinating.

Qwen, on the other hand, hit 80%, which is the strongest performance we’ve seen so far. Qwen seems to really like explicit tags and clear reasoning scaffolding. It follows the format almost perfectly and tends to give clean numerical answers without extra explanation. The combination of structured instructions + LaTeX boxed answer seems to align well with its training style.

Overall, P1 shows that a single unified prompt can work well for both models, with Qwen taking a small lead in accuracy. It also gives us a clean reference point as we explore more advanced prompts like P2 and beyond.



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

Qwen dropped from 80% → 50% → P2 harms Qwen because P2 forces a more structured CoT + explanation + formatted answer block, which Qwen doesn’t follow consistently.