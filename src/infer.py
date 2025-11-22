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
# ORIGINAL FINR1-NATIVE PROMPTS (HIGH ACCURACY)
# ============================================================

FINR1_SYSTEM_PROMPT = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the final answer.
Respond in the following format:
<think>
...
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""

FINR1_TASK_PROMPT = """Please answer the financial question based on the provided context."""

FINR1_ANSWER_FORMAT = """
<think>
(step-by-step internal reasoning here)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# ============================================================
# QWEN P1 PROMPTS (UNCHANGED — 80% ACCURACY)
# ============================================================

SYSTEM_PROMPT_QWEN = """You are a financial reasoning assistant.
You think step-by-step inside <think>...</think>.
After reasoning, produce ONLY the final numeric answer inside one LaTeX box inside <answer>...</answer>.
"""

TASK_PROMPT_QWEN = """Use the financial context (text + tables) to extract relevant numbers,
perform multi-step calculations, and derive the final numeric answer."""

ANSWER_FORMAT_QWEN = """
<think>
(your hidden reasoning)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# ============================================================
# BUILD PLAIN PROMPT (FINR1 — PLAIN TEXT, NO CHAT)
# ============================================================

def build_plain_prompt(sample: FinQASample) -> str:
    """Strict FinR1-style formatting."""

    system_prompt = FINR1_SYSTEM_PROMPT
    task_prompt   = FINR1_TASK_PROMPT
    answer_format = FINR1_ANSWER_FORMAT

    pre = sample.pre_text.strip()
    post = sample.post_text.strip()
    table_text = table_to_text(sample.table)

    parts = [task_prompt, "Context:"]
    if pre:
        parts.append(pre)
    if table_text:
        parts.append("Table:\n" + table_text)
    if post:
        parts.append(post)

    context_block = "\n\n".join(parts)

    full = (
        system_prompt
        + "\n"
        + context_block
        + "\n\nGiven the context, "
        + sample.question
        + "\n\n"
        + answer_format
    )

    return full.strip()


# ============================================================
# BUILD CHAT PROMPT (QWEN)
# ============================================================

def build_chat_prompt(sample: FinQASample) -> str:
    pre = sample.pre_text.strip()
    post = sample.post_text.strip()
    table_text = table_to_text(sample.table)

    parts = [TASK_PROMPT_QWEN + "\nContext:"]
    if pre:
        parts.append(pre)
    if table_text:
        parts.append("Table:\n" + table_text)
    if post:
        parts.append(post)

    context_block = "\n\n".join(parts)
    return f"{context_block}\n\nGiven the context, {sample.question}\n\n{ANSWER_FORMAT_QWEN}"


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
# RUN INFERENCE (MODEL SWITCHING LOGIC)
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
        # FINR1 BRANCH (PLAIN PROMPT)
        # ----------------------------------------------------
        if "SUFE-AIFLM-Lab/Fin-R1" in model_name or "Fin-R1" in model_name:
            prompt = build_plain_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ----------------------------------------------------
        # QWEN BRANCH (CHAT PROMPT — UNTOUCHED)
        # ----------------------------------------------------
        else:
            user_msg = build_chat_prompt(sample)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_QWEN},
                {"role": "user",  "content": user_msg},
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

        output_ids = model.generate(**inputs, **gen_kwargs)

        gen_text = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        predictions.append(gen_text)

    return predictions
