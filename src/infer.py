"""Baseline inference utilities.

This module formats prompts for FinQA samples, runs autoregressive
generation with a Hugging Face causal LM, and stop generation when the
boxed numeric answer is complete.
"""

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


# =========================
#  PROMPT VERSIONS
# =========================

SYSTEM_PROMPT_old = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
"""

TASK_PROMPT_old = """Please answer the given financial question based on the context."""

ANSWER_FORMAT_old = """Show reasoning then output \\boxed{value}.\n"""

SYSTEM_PROMPT_P1 = """You are a financial reasoning assistant.
You will think step-by-step inside <think>...</think>.
Produce only the final numeric answer inside \\boxed{}."""

TASK_PROMPT_P1 = """Use the financial context to extract numbers and compute the final answer."""

ANSWER_FORMAT_P1 = """<answer>\\boxed{FINAL_ANSWER}</answer>"""

SYSTEM_PROMPT_P2 = """You are a helpful AI Assistant that reasons inside <think> and then answers."""

TASK_PROMPT_P2 = """Use the document to compute the requested numeric value."""

ANSWER_FORMAT_P2 = """Provide ONLY the final answer inside \\boxed{FINAL_ANSWER}."""

# CURRENT ACTIVE = P1
SYSTEM_PROMPT = SYSTEM_PROMPT_P1
TASK_PROMPT = TASK_PROMPT_P1
ANSWER_FORMAT = ANSWER_FORMAT_P1


# =========================
#  PROMPT BUILDER
# =========================

def build_prompt(sample: FinQASample) -> str:
    context_parts: List[str] = [TASK_PROMPT + "\nContext:"]

    pre = sample.pre_text.strip()
    post = sample.post_text.strip()
    table_text = table_to_text(sample.table)

    if pre:
        context_parts.append(pre)
    if table_text:
        context_parts.append("Table:\n" + table_text)
    if post:
        context_parts.append(post)

    context_block = "\n\n".join(context_parts).strip()
    if context_block:
        context_block += "\n\n"

    return f"{context_block}Given the context, {sample.question}"


# =========================
#  STOPPING CRITERIA
# =========================

class BoxedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=8):
        self.trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        self.len_trigger_ids = len(self.trigger_ids)
        self.close_id = tokenizer.encode(close, add_special_tokens=False)[-1]
        self.min_after = min_after
        self.max_after = max_after
        self.seen_trigger = False
        self.after_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]

        if not self.seen_trigger:
            if seq[-self.len_trigger_ids:] == self.trigger_ids:
                self.seen_trigger = True
        else:
            self.after_count += 1
            if seq[-1] == self.close_id and self.after_count >= self.min_after:
                return True
            if self.after_count >= self.max_after:
                return True

        return False


# =========================
#  MAIN INFERENCE FUNCTION
# =========================

@torch.inference_mode()
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[FinQASample],
    *,
    max_new_tokens: int = 4000,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.05,
) -> List[str]:

    device = next(model.parameters()).device
    predictions: List[str] = []

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature is not None or top_p is not None,
        repetition_penalty=repetition_penalty,
    )
    if temperature is not None:
        generation_kwargs["temperature"] = temperature
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    # =========================
    #  MODEL NAME SELECTOR
    # =========================
    model_name = getattr(model.config, "_name_or_path", "").lower()
    use_chat_format = (
        "qwen" in model_name
        or "chat" in model_name
        or "instruct" in model_name
    )
    # FinR1 MUST NOT use chat format

    # =========================
    #  LOOP THROUGH SAMPLES
    # =========================
    for sample in tqdm(samples):
        prompt = build_prompt(sample)

        if use_chat_format:
            # QWEN / Chat LLMs
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt + "\n\n" + ANSWER_FORMAT},
            ]
            str_messages = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(str_messages, return_tensors="pt").to(device)

        else:
            # FINR1 → Plain text ONLY
            plain_prompt = (
                SYSTEM_PROMPT
                + "\n\n"
                + prompt
                + "\n\n"
                + ANSWER_FORMAT
            )
            inputs = tokenizer(plain_prompt, return_tensors="pt").to(device)

        criteria = BoxedStoppingCriteria(tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=8)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(prediction)

    return predictions
