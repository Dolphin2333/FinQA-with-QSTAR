"""Baseline inference utilities.

This module formats prompts for FinQA samples, runs autoregressive
generation with a Hugging Face causal LM, and stop generation when the
boxed numeric answer is complete.

Highlights:
- ``build_prompt`` assembles context from narrative text and table content
  (converted to sentences) and instructs the model to return the final
  result in ``\\boxed{...}``.
- ``BoxedStoppingCriteria`` halts generation shortly after ``\\boxed{``
  is opened and the closing brace ``}`` is produced, preventing trailing text.
- ``run_inference`` performs batched generation with temperature, top-p, and
  repetition penalty controls, returning decoded predictions.
"""

from __future__ import annotations

from typing import List, Sequence
from tqdm import tqdm

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from .load_data import FinQASample
from .table_utils import table_to_text


SYSTEM_PROMPT_old = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
You first think about the reasoning process as an internal monologue and then provide the user with the answer. 
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>
Please use \\boxed{} to wrap the final answer\n\n"""

TASK_PROMPT_old = """Please answer the given financial question based on the context."""

ANSWER_FORMAT_old = """Show your reasoning step by step, then output only the final numeric result in the form \\boxed{value}. 
End your response immediately after the boxed answer — do not add any explanation, summary, or extra text.\n\n"""


SYSTEM_PROMPT_P1 = """You are a financial reasoning assistant.
You will think step-by-step using hidden deliberate reasoning inside <think>...</think>.
After reasoning, produce the final numeric answer inside a single LaTeX box."""

TASK_PROMPT_P1 = """Use the financial context (text + tables) to extract relevant numbers,
perform multi-step calculations, and derive the final result."""

ANSWER_FORMAT_P1 = """Respond in the format:
<think>
(Your step-by-step reasoning here)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


SYSTEM_PROMPT_P2 = """You are a helpful AI Assistant.
You first think about the reasoning process as an internal monologue,
and then provide the user with the final answer.

Respond in the following format:
<think>
(step-by-step reasoning here)
</think>
<answer>
...
</answer>
"""

TASK_PROMPT_P2 = """Use the financial document (text + tables) to extract all relevant numbers,
perform the necessary reasoning and calculations, and obtain the final numeric result."""

ANSWER_FORMAT_P2 = """Provide ONLY the final answer inside a single LaTeX box: \\boxed{FINAL_ANSWER}.
End your response immediately after the box with no extra text."""


SYSTEM_PROMPT = SYSTEM_PROMPT_P1
TASK_PROMPT = TASK_PROMPT_P1
ANSWER_FORMAT = ANSWER_FORMAT_P1


def build_prompt(sample: FinQASample) -> str:
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

    # FIX: Removed ANSWER_FORMAT from prompt body
    return f"{context_block}Given the context, {sample.question}"


class BoxedStoppingCriteria(StoppingCriteria):
    """Stop generation after a boxed answer is complete."""
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
    """Generate answers for a list of :class:`FinQASample` inputs."""
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
        prompt = build_prompt(sample)
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        str_messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(str_messages, return_tensors="pt").to(device)

        criteria = BoxedStoppingCriteria(tokenizer, trigger="\\boxed{", close="}", min_after=1, max_after=8)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])
        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(prediction)

    return predictions
