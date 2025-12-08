"""Baseline inference utilities.

This module formats prompts for FinQA samples, runs autoregressive
generation with a Hugging Face causal LM, and stop generation when the
boxed numeric answer is complete.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Any, Tuple
from tqdm import tqdm
import pandas as pd

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from .table_utils import table_to_text
from .eval_finqa import _extract_boxed_prediction, _normalize_magnitude


PROMPT_FILE = "data/prompts.csv"


# ============================================================
# BUILD PROMPT — SAME FOR BOTH MODELS
# ============================================================

def build_prompt(sample, TASK_PROMPT, ANSWER_FORMAT) -> str:
    """Format a FinQA prompt for autoregressive inference."""
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

    return f"{context_block}Given the context, {sample.question}\n\n{ANSWER_FORMAT}\n\n"


# ============================================================
# STOPPING CRITERIA
# ============================================================

class BoxedStoppingCriteria(StoppingCriteria):
    """Stop generation after a boxed answer is complete."""

    def __init__(self, tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=8):
        self.trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        self.len_trigger_ids = len(self.trigger_ids)
        self.close_id = tokenizer.encode(close, add_special_tokens=False)[-1]
        self.min_after = min_after
        self.max_after = max_after
        self._seen_trigger: List[bool] = []
        self._after_counts: List[int] = []

    def _ensure_state(self, batch_size: int) -> None:
        if len(self._seen_trigger) != batch_size:
            self._seen_trigger = [False for _ in range(batch_size)]
            self._after_counts = [0 for _ in range(batch_size)]

    def __call__(self, input_ids, scores, **kwargs):
        batch_size = input_ids.shape[0]
        self._ensure_state(batch_size)

        stop_mask = []
        for idx in range(batch_size):
            seq = input_ids[idx]

            if not self._seen_trigger[idx]:
                if seq[-self.len_trigger_ids:] == self.trigger_ids:
                    self._seen_trigger[idx] = True
                stop_mask.append(False)
                continue

            self._after_counts[idx] += 1
            at_close = seq[-1] == self.close_id and self._after_counts[idx] >= self.min_after
            at_limit = self._after_counts[idx] >= self.max_after
            stop_mask.append(at_close or at_limit)

        return all(stop_mask) if stop_mask else False


# ============================================================
# SELF-CONSISTENCY HELPERS
# ============================================================

def extract_boxed_answers(responses: Sequence[str]) -> List[Any]:
    """Parse a list of raw generations into boxed answers."""

    return [_extract_boxed_prediction(resp) for resp in responses]


def _canonicalize_vote_value(value: Any, anchor: float | None) -> Tuple[Tuple[str, str], Any, float | None]:
    """Normalize answers so majority voting can tolerate scale mismatches."""

    if isinstance(value, (int, float)) and math.isfinite(value):
        numeric_value = float(value)
        if anchor is None:
            anchor = numeric_value
            normalized = numeric_value
        else:
            normalized, _ = _normalize_magnitude(numeric_value, anchor)
        key = ("num", f"{normalized:.6g}")
        return key, normalized, anchor

    text_value = "" if value is None else str(value)
    normalized_text = text_value.strip()
    key = ("text", normalized_text.lower())
    return key, normalized_text, anchor


def majority_vote_boxed_answers(values: Sequence[Any]) -> List[int]:
    """Return the winning normalized answer and indices of members supporting it."""

    if not values:
        return None, []

    counts: Counter = Counter()
    representatives: dict[Tuple[str, str], Any] = {}
    first_seen: dict[Tuple[str, str], int] = {}
    memberships: dict[Tuple[str, str], List[int]] = {}
    anchor: float | None = None

    for idx, value in enumerate(values):
        if value is None:
            continue
        key, canonical_value, anchor = _canonicalize_vote_value(value, anchor)
        counts[key] += 1
        representatives.setdefault(key, canonical_value)
        first_seen.setdefault(key, idx)
        memberships.setdefault(key, []).append(idx)

    if not counts:
        return None, list(range(len(values)))

    # determine highest vote count
    max_count = max(counts.values())

    # collect all keys that have the max_count, order by first_seen (earlier seen first)
    winning_keys = sorted((k for k, c in counts.items() if c == max_count), key=lambda k: first_seen[k])

    # build winners and their memberships in the same order
    winner_memberships = [idx for k in winning_keys for idx in memberships[k]]

    return winner_memberships


def _format_numeric_answer(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def format_boxed_answer(value: Any) -> str:
    """Wrap the selected value in a boxed answer string."""

    if value is None:
        return "\\boxed{}"
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("\\boxed"):
            return cleaned
        return f"\\boxed{{{cleaned}}}"
    return f"\\boxed{{{_format_numeric_answer(float(value))}}}"


def get_prompts(prompt_id: int) -> List[str]:
    prompts = pd.read_csv(PROMPT_FILE)
    prompts = prompts.loc[prompts.ID == f"P{prompt_id}",:].reset_index()
    SYSTEM_PROMPT = prompts.SYSTEM_PROMPT[0]
    TASK_PROMPT = prompts.TASK_PROMPT[0]
    ANSWER_FORMAT = prompts.ANSWER_FORMAT[0]
    return SYSTEM_PROMPT, TASK_PROMPT, ANSWER_FORMAT


# ============================================================
# RUN INFERENCE (UNCHANGED LOGIC)
# ============================================================

@torch.inference_mode()
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[Any],
    *,
    max_new_tokens: int = 4000,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.05,
    prompt_id: int = 1,
) -> List[str]:
    """Generate answers for a list of FinQASample inputs."""
    SYSTEM_PROMPT, TASK_PROMPT, ANSWER_FORMAT = get_prompts(prompt_id)
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

    for sample in tqdm(samples):
        prompt = build_prompt(sample, TASK_PROMPT, ANSWER_FORMAT)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        str_messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(str_messages, return_tensors="pt").to(device)

        criteria = BoxedStoppingCriteria(
            tokenizer,
            trigger="\\boxed{",
            close="}",
            min_after=1,
            max_after=8,
        )
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predictions.append(prediction)

    return predictions


@torch.inference_mode()
def run_self_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[Any],
    *,
    num_sequences: int = 10,
    max_new_tokens: int = 4000,
    temperature: float | None = 0.7,
    top_p: float | None = 0.8,
    repetition_penalty: float = 1.05,
    prompt_id: int = 1,
    return_all_generations: bool = False,
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Generate multiple reasoning paths per sample and aggregate by majority vote."""

    if num_sequences < 1:
        raise ValueError("num_sequences must be >= 1")

    SYSTEM_PROMPT, TASK_PROMPT, ANSWER_FORMAT = get_prompts(prompt_id)
    device = next(model.parameters()).device
    all_generations: List[List[str]] = []
    winning_generations: List[List[str]] = []

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=num_sequences,
        repetition_penalty=repetition_penalty,
    )
    if temperature is not None:
        generation_kwargs["temperature"] = temperature
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    for sample in tqdm(samples):
        prompt = build_prompt(sample, TASK_PROMPT, ANSWER_FORMAT)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        str_messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(str_messages, return_tensors="pt").to(device)

        criteria = BoxedStoppingCriteria(
            tokenizer,
            trigger="\\boxed{",
            close="}",
            min_after=1,
            max_after=8,
        )
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])

        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
        sample_generations = [
            tokenizer.decode(seq, skip_special_tokens=True).strip()
            for seq in generated_ids
        ]

        answers = extract_boxed_answers(sample_generations)
        winning_indices = majority_vote_boxed_answers(answers)
        winning_generations.append([sample_generations[i] for i in winning_indices])

        if return_all_generations:
            all_generations.append(sample_generations)

    if return_all_generations:
        return all_generations, winning_generations
    return [], winning_generations
