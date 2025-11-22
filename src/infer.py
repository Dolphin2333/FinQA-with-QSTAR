"""Baseline inference utilities."""

from __future__ import annotations
from typing import List, Sequence
from tqdm import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
from .load_data import FinQASample
from .table_utils import table_to_text

# ============================================================
# ORIGINAL PROMPTS (kept exactly as in main branch)
# ============================================================

SYSTEM_PROMPT_ORIG = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
You first think about the reasoning process as an internal monologue and then provide the user with the answer. 
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>
Please use \\boxed{} to wrap the final answer\n\n"""

TASK_PROMPT_ORIG = """Please answer the given financial question based on the context."""

ANSWER_FORMAT_ORIG = """Show your reasoning step by step, then output only the final numeric result in the form \\boxed{value}. 
End your response immediately after the boxed answer — do not add any explanation, summary, or extra text.\n\n"""


# ============================================================
#  NEW PROMPTS
# ============================================================

# ----- P1 -----
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


# ----- P2 -----
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


# ============================================================
# SELECT WHICH PROMPT TO USE (change ONLY THIS)
# ============================================================

USE_PROMPT = "P1"   # OPTIONS: "ORIGINAL", "P1", "P2"

if USE_PROMPT == "P1":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P1
    TASK_PROMPT = TASK_PROMPT_P1
    # do NOT inject ANSWER_FORMAT into prompt body
elif USE_PROMPT == "P2":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P2
    TASK_PROMPT = TASK_PROMPT_P2
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_ORIG
    TASK_PROMPT = TASK_PROMPT_ORIG


# ============================================================
# FIXED build_prompt (compatible with all prompt styles)
# ============================================================

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

    context_block = "\n\n".join(context_parts).strip() + "\n\n"
    return f"{context_block}Given the context, {sample.question}"


# ============================================================
# Stopping Criteria and run_inference (unchanged)
# ============================================================

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


@torch.inference_mode()
def run_inference(model, tokenizer, samples, **gen_kwargs):
    device = next(model.parameters()).device
    predictions = []

    for sample in tqdm(samples):
        prompt = build_prompt(sample)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        encoded = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(encoded, return_tensors="pt").to(device)

        stopping = BoxedStoppingCriteria(tokenizer)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

        output_ids = model.generate(**inputs, **gen_kwargs)
        gen = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        predictions.append(gen)

    return predictions
