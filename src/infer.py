"""Baseline inference utilities with switchable prompt variants."""

from __future__ import annotations
from typing import List, Sequence
from tqdm import tqdm
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .load_data import FinQASample
from .table_utils import table_to_text


# ============================================================
# CHOOSE FINR1 PROMPT VERSION HERE
# ============================================================

FINR1_PROMPT_VERSION = "P1"  
# Options: "P0" (original 70% baseline), "P1" (your improved version)


# ============================================================
# FINR1 PROMPTS — MULTIPLE VARIANTS
# ============================================================

# ---- P0 (Original High-Accuracy Prompt) ----
FINR1_SYSTEM_P0 = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format:
<think>
...
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""

FINR1_TASK_P0 = "Please answer the given financial question based on the context."

FINR1_ANSWER_P0 = """Show your reasoning step by step, then output only the final numeric result in the form \\boxed{value}.
End your response immediately after the boxed answer — no extra text.
"""


# ---- P1 (Improved Reasoning Prompt Variant) ----
FINR1_SYSTEM_P1 = """You are a financial analysis assistant.
Think step-by-step inside <think>...</think> using precise numeric reasoning.
Then provide ONLY the final number inside one LaTeX box.
"""

FINR1_TASK_P1 = """Use the financial context (narrative + table) to extract the relevant numbers and compute the answer."""

FINR1_ANSWER_P1 = """
<think>
(hidden reasoning)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# Map for easy switching
FINR1_PROMPTS = {
    "P0": (FINR1_SYSTEM_P0, FINR1_TASK_P0, FINR1_ANSWER_P0),
    "P1": (FINR1_SYSTEM_P1, FINR1_TASK_P1, FINR1_ANSWER_P1),
}


# ============================================================
# QWEN PROMPTS (UNCHANGED — 80% ACCURACY)
# ============================================================

SYSTEM_PROMPT_QWEN = """You are a financial reasoning assistant.
Think step-by-step inside <think>...</think>.
After reasoning, produce ONLY the final numeric answer inside a single LaTeX box.
"""

TASK_PROMPT_QWEN = """Use the financial context to extract numbers and compute the final numeric result."""

ANSWER_FORMAT_QWEN = """
<think>
(your hidden reasoning)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# ============================================================
# BUILD PROMPT (CHAT FORMAT FOR BOTH MODELS)
# ============================================================

def build_user_prompt(task_prompt: str, answer_format: str, sample: FinQASample) -> str:

    pre = sample.pre_text.strip()
    post = sample.post_text.strip()
    table_text = table_to_text(sample.table)

    parts = [task_prompt + "\nContext:"]
    if pre:
        parts.append(pre)
    if table_text:
        parts.append("Table:\n" + table_text)
    if post:
        parts.append(post)

    context_block = "\n\n".join(parts)

    return (
        f"{context_block}\n\nGiven the context, {sample.question}\n\n{answer_format}"
    )


# ============================================================
# STOPPING CRITERIA
# ============================================================

class BoxedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=12):
        self.trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        self.trigger_len = len(self.trigger_ids)
        self.close_id = tokenizer.encode(close, add_special_tokens=False)[-1]
        self.seen = False
        self.after = 0
        self.min_after = min_after
        self.max_after = max_after

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]

        if not self.seen:
            if seq[-self.trigger_len:] == self.trigger_ids:
                self.seen = True
        else:
            self.after += 1
            if seq[-1] == self.close_id and self.after >= self.min_after:
                return True
            if self.after >= self.max_after:
                return True

        return False


# ============================================================
# RUN INFERENCE (MODEL-SWITCHING)
# ============================================================

@torch.inference_mode()
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[FinQASample],
    model_name: str,
    **gen_kwargs,
):

    device = next(model.parameters()).device
    predictions = []

    # Select FINR1 prompt version
    fin_system, fin_task, fin_answer = FINR1_PROMPTS[FINR1_PROMPT_VERSION]

    for sample in tqdm(samples):

        # ----------------------------------------------------
        # FINR1 → Chat-based prompt (NOT plain text)
        # ----------------------------------------------------
        if "Fin-R1" in model_name or "SUFE-AIFLM-Lab/Fin-R1" in model_name:

            user_msg = build_user_prompt(fin_task, fin_answer, sample)

            messages = [
                {"role": "system", "content": fin_system},
                {"role": "user", "content": user_msg},
            ]

            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(encoded, return_tensors="pt").to(device)

        # ----------------------------------------------------
        # QWEN → Unmodified P1 Chat Prompt
        # ----------------------------------------------------
        else:
            user_msg = build_user_prompt(TASK_PROMPT_QWEN, ANSWER_FORMAT_QWEN, sample)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_QWEN},
                {"role": "user", "content": user_msg},
            ]

            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(encoded, return_tensors="pt").to(device)

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------
        stopping = BoxedStoppingCriteria(tokenizer)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

        output_ids = model.generate(**inputs, **gen_kwargs)

        gen_text = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        predictions.append(gen_text)

    return predictions
