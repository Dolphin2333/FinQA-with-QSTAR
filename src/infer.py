"""Baseline inference utilities."""

from __future__ import annotations

from typing import List, Sequence
from tqdm import tqdm

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from .load_data import FinQASample
from .table_utils import table_to_text


def build_prompt(sample: FinQASample) -> str:
    """Format a FinQA prompt for autoregressive inference."""
    context_parts: List[str] = ["""Please answer the given financial question based on the context.
Context:"""]

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

    answer_format = """
Show your reasoning step by step, then output only the final numeric result in the form \\boxed{value}. 
End your response immediately after the boxed answer — do not add any explanation, summary, or extra text.\n\n"""

    return f"{context_block}Given the context, {sample.question}\n\n{answer_format}\n\n"


class BoxedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, trigger="\\boxed{", extra_tokens=10):
        self.tokenizer = tokenizer
        self.trigger = trigger
        self.extra_tokens = extra_tokens
        self.trigger_found = False
        self.trigger_token_len = len(tokenizer.encode(trigger, add_special_tokens=False))
        self.counter = 0

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0])
        if not self.trigger_found:
            if self.trigger == text[-len(self.trigger):]:
                self.trigger_found = True
                self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.extra_tokens:
                return True
        return False


@torch.inference_mode()
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[FinQASample],
    *,
    max_new_tokens: int = 64,
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

    system_prompt = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
You first think about the reasoning process as an internal monologue and then provide the user with the answer. 
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>
Please use \\boxed{} to wrap the final answer\n\n"""

    for sample in tqdm(samples):
        prompt = build_prompt(sample)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        str_messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(str_messages, return_tensors="pt").to(device)

        criteria = BoxedStoppingCriteria(tokenizer, trigger="\\boxed{", extra_tokens=10)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])
        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(prediction)

    return predictions
