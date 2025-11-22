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
# MODIFIED P1 PROMPT FOR FinR1 (PLAIN TEXT, HIGH-STABILITY)
# ============================================================

FINR1_SYSTEM_PROMPT_P1 = """You are a financial quantitative reasoning assistant.
Read the financial context carefully, extract the required values from the text and tables,
perform step-by-step calculations, and produce a single numeric answer.
Show your reasoning only inside <think>...</think>.
Give only the final answer inside \\boxed{} inside <answer>...</answer>.
Do not provide explanations outside these tags.
"""

FINR1_TASK_PROMPT_P1 = """Task: Use the provided narrative and table to compute the answer."""

FINR1_ANSWER_FORMAT_P1 = """
<think>
(step-by-step reasoning here)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# ============================================================
# QWEN P1 PROMPT (UNCHANGED)
# ============================================================

SYSTEM_PROMPT_QWEN = """You are a financial reasoning assistant.
You think step-by-step inside <think>...</think>.
After reasoning, output ONLY the final numeric answer inside one LaTeX box.
"""

TASK_PROMPT_QWEN = """Use the financial context (text + tables) to extract relevant numbers
and compute the final numeric answer."""

ANSWER_FORMAT_QWEN = """
<think>
(hidden reasoning)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


# ============================================================
# BUILD PLAIN PROMPT (FinR1)
# ============================================================

def build_plain_prompt(sample: FinQASample) -> str:
    system_prompt = FINR1_SYSTEM_PROMPT_P1
    task_prompt   = FINR1_TASK_PROMPT_P1
    answer_format = FINR1_ANSWER_FORMAT_P1

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

    full_prompt = (
        system_prompt
        + "\n"
        + context_block
        + "\n\nQuestion: " + sample.question
        + "\n\n"
        + answer_format
    )

    return full_prompt.strip()


# ============================================================
# BUILD CHAT PROMPT (Qwen)
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

    block = "\n\n".join(parts)

    return f"{block}\n\nGiven the context, {sample.question}\n\n{ANSWER_FORMAT_QWEN}"


# ============================================================
# STOPPING CRITERIA
# ============================================================

class BoxedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=10):
        self.trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        self.trigger_len = len(self.trigger_ids)
        self.close_id = tokenizer.encode(close, add_special_tokens=False)[-1]
        self.seen = False
        self.after = 0
        self.min_after = min_after
        self.max_after = max_after

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]

        # detect start of \boxed{
        if not self.seen:
            if seq[-self.trigger_len:] == self.trigger_ids:
                self.seen = True
        else:
            self.after += 1

            # stop when closing brace appears
            if seq[-1] == self.close_id and self.after >= self.min_after:
                return True

            # safety cutoff
            if self.after >= self.max_after:
                return True

        return False


# ============================================================
# RUN INFERENCE
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
    preds = []

    for sample in tqdm(samples):

        # ----- FinR1 branch -----
        if "Fin-R1" in model_name or "SUFE-AIFLM-Lab/Fin-R1" in model_name:
            prompt = build_plain_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ----- Qwen branch -----
        else:
            user_msg = build_chat_prompt(sample)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_QWEN},
                {"role": "user", "content": user_msg},
            ]
            enc = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(enc, return_tensors="pt").to(device)

        # generation settings
        stopping = BoxedStoppingCriteria(tokenizer)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

        out = model.generate(**inputs, **gen_kwargs)

        gen_text = tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        preds.append(gen_text)

    return preds
