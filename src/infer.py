"""Baseline inference utilities."""

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
# ORIGINAL PROMPTS
# ============================================================

SYSTEM_PROMPT_ORIG = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format:
<think>
...
</think>
<answer>
\boxed{FINAL_ANSWER}
</answer>
"""

TASK_PROMPT_ORIG = "Please answer the given financial question based on the context."

ANSWER_FORMAT_ORIG = """Provide your detailed reasoning inside <think>...</think>.
Then output ONLY the final numeric result inside one \boxed{} inside <answer>...</answer>.
"""


# ============================================================
# NEW PROMPTS (P1 / P2)
# ============================================================

# ---- P1 ----
SYSTEM_PROMPT_P1 = """You are a financial reasoning assistant.
You think step-by-step inside <think>...</think>.
After reasoning, produce ONLY the final numeric answer inside a single LaTeX box inside <answer>...</answer>.
"""

TASK_PROMPT_P1 = """Use the financial context (text + tables) to extract relevant numbers,
perform multi-step calculations, and derive the final numeric answer."""

ANSWER_FORMAT_P1 = """
<think>
(your hidden reasoning)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# ---- P2 ----
SYSTEM_PROMPT_P2 = """You are a helpful AI Assistant.
You think step-by-step inside <think>...</think> and then provide the final answer.
"""

TASK_PROMPT_P2 = """Use the financial document to extract relevant quantities and compute the final answer."""

ANSWER_FORMAT_P2 = """Output the final answer ONLY inside one LaTeX box: \\boxed{FINAL_ANSWER}."""


# ============================================================
# PROMPT SELECTION
# ============================================================

USE_PROMPT = "P1"   # OPTIONS: "ORIGINAL", "P1", "P2"

if USE_PROMPT == "P1":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P1
    TASK_PROMPT = TASK_PROMPT_P1
    ANSWER_FORMAT = ANSWER_FORMAT_P1
elif USE_PROMPT == "P2":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P2
    TASK_PROMPT = TASK_PROMPT_P2
    ANSWER_FORMAT = ANSWER_FORMAT_P2
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_ORIG
    TASK_PROMPT = TASK_PROMPT_ORIG
    ANSWER_FORMAT = ANSWER_FORMAT_ORIG


# ============================================================
# BUILD PROMPT
# FinR1 → plain-text prompt
# Qwen  → chat messages handled later
# ============================================================

def build_plain_prompt(sample: FinQASample) -> str:
    """
    Used ONLY for FinR1.
    Construct a FULL plain-text prompt with:
    - SYSTEM_PROMPT
    - TASK_PROMPT
    - CONTEXT
    - QUESTION
    - ANSWER FORMAT
    """
    pre = sample.pre_text.strip()
    post = sample.post_text.strip()
    table_text = table_to_text(sample.table)

    context_parts = [TASK_PROMPT, "Context:"]
    if pre:
        context_parts.append(pre)
    if table_text:
        context_parts.append("Table:\n" + table_text)
    if post:
        context_parts.append(post)

    context_block = "\n\n".join(context_parts)

    final_prompt = (
        SYSTEM_PROMPT
        + "\n"
        + context_block
        + "\n\nGiven the context, " + sample.question + "\n\n"
        + ANSWER_FORMAT
    )

    return final_prompt.strip()


def build_chat_prompt(sample: FinQASample) -> str:
    """
    Used for Qwen chat template.
    Only returns USER message content.
    """
    pre = sample.pre_text.strip()
    post = sample.post_text.strip()
    table_text = table_to_text(sample.table)

    context_parts = [TASK_PROMPT + "\nContext:"]
    if pre:
        context_parts.append(pre)
    if table_text:
        context_parts.append("Table:\n" + table_text)
    if post:
        context_parts.append(post)

    context_block = "\n\n".join(context_parts)
    return f"{context_block}\n\nGiven the context, {sample.question}\n\n{ANSWER_FORMAT}"


# ============================================================
# STOPPING CRITERIA
# ============================================================

class BoxedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=12):
        self.trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        self.len_trigger_ids = len(self.trigger_ids)
        self.close_id = tokenizer.encode(close, add_special_tokens=False)[-1]
        self.seen_trigger = False
        self.after = 0
        self.min_after = min_after
        self.max_after = max_after

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        if not self.seen_trigger:
            if seq[-self.len_trigger_ids:] == self.trigger_ids:
                self.seen_trigger = True
        else:
            self.after += 1
            if seq[-1] == self.close_id and self.after >= self.min_after:
                return True
            if self.after >= self.max_after:
                return True
        return False


# ============================================================
# RUN INFERENCE (WORKS FOR BOTH MODELS)
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

    for sample in tqdm(samples):

        # ----------------------------------------------------
        # BRANCH 1 — FinR1 (plain text)
        # ----------------------------------------------------
        if "SUFE-AIFLM-Lab/Fin-R1" in model_name or "Fin-R1" in model_name:
            prompt = build_plain_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ----------------------------------------------------
        # BRANCH 2 — Qwen (chat template)
        # ----------------------------------------------------
        else:
            user_msg = build_chat_prompt(sample)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(encoded, return_tensors="pt").to(device)

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------
        stopping = BoxedStoppingCriteria(tokenizer)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

        output = model.generate(**inputs, **gen_kwargs)

        gen_text = tokenizer.decode(
            output[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        predictions.append(gen_text)

    return predictions
