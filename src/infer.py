"""Baseline inference utilities.

This module formats prompts for FinQA samples, runs autoregressive
generation with a Hugging Face causal LM, and stop generation when the
boxed numeric answer is complete.
"""

from __future__ import annotations

from typing import List, Sequence
from tqdm import tqdm

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from .load_data import FinQASample
from .table_utils import table_to_text

# ============================================================
# PROMPT VERSION SELECTION
# ============================================================

USE_PROMPT = "P3"     # OPTIONS: "ORIGINAL", "P1"

# ------------ ORIGINAL (from main branch) -------------------

SYSTEM_PROMPT_ORIG = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
You first think about the reasoning process as an internal monologue and then provide the user with the answer. 
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>
Please use \\boxed{} to wrap the final answer\n\n"""

TASK_PROMPT_ORIG = """Please answer the given financial question based on the context."""

ANSWER_FORMAT_ORIG = """Show your reasoning step by step, then output only the final numeric result in the form \\boxed{value}. 
End your response immediately after the boxed answer — do not add any explanation, summary, or extra text.\n\n"""

# ------------ NEW P1 PROMPT (UNIFIED FOR BOTH MODELS) -------------

SYSTEM_PROMPT_P1 = """You are a financial reasoning assistant.
Think step-by-step inside <think>...</think>.
Use the given financial context (text + tables) to extract numbers and perform calculations.
After reasoning, output ONLY the final numeric answer inside a single LaTeX box.
"""

TASK_PROMPT_P1 = """Use the context to identify relevant values and compute the correct final answer."""

ANSWER_FORMAT_P1 = """
<think>
(your reasoning steps here)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""


SYSTEM_PROMPT_P2 = """You are a financial computation assistant.
You must carefully extract numbers from the context, perform accurate step-by-step calculations,
and return ONLY a final numeric answer formatted inside a single LaTeX box.

Always follow this format:
<think>
(detailed reasoning steps, computations, intermediate values)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>

The final answer must be a pure number without units."""


TASK_PROMPT_P2 = """Use the provided financial narrative and table to extract the correct values,
perform multi-step reasoning, and compute the final numeric answer.
Be careful: the context may contain distractor numbers or irrelevant lines."""


ANSWER_FORMAT_P2 = """
<think>
(show all reasoning steps and calculations)
</think>
<answer>
\\boxed{FINAL_ANSWER}
</answer>
"""

# ------------ NEW P3 PROMPT (Program-of-Thought + Precision) -------------

SYSTEM_PROMPT_P3 = """You are a financial reasoning assistant.
You must analyze the financial context and extract a sequence of explicit mathematical operations.
Always think inside <think>...</think> using the following rules:

1. Identify ALL relevant numbers from the text and tables.
2. Represent intermediate steps using explicit operators:
   ADD(a,b), SUB(a,b), MUL(a,b), DIV(a,b), MAX(), MIN(), AVG(), TBL_SUM().
3. Show each operation in order inside <think>.
4. After completing the reasoning, compute the final numeric value.
5. Output ONLY the final numeric result with two-decimal precision inside \\boxed{}.

Do NOT output words, sentences, or explanations in the final answer.
"""

TASK_PROMPT_P3 = """Extract all relevant financial values, express intermediate steps using program-style operations, compute the final result, and return ONLY the final numeric output."""

ANSWER_FORMAT_P3 = """
<think>
Extract numbers → write program-style operations (ADD(), SUB(), MUL(), DIV(), etc.) → compute result.
</think>
<answer>
\\boxed{FINAL_NUMERIC_ANSWER}
</answer>
"""



# ------------ SELECT ACTIVE PROMPT ----------------------------

if USE_PROMPT == "P3":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P1
    TASK_PROMPT = TASK_PROMPT_P1
    ANSWER_FORMAT = ANSWER_FORMAT_P1
elif USE_PROMPT == "P2":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P2
    TASK_PROMPT = TASK_PROMPT_P2
    ANSWER_FORMAT = ANSWER_FORMAT_P2  
elif USE_PROMPT == "P3":
    SYSTEM_PROMPT = SYSTEM_PROMPT_P3
    TASK_PROMPT = TASK_PROMPT_P3
    ANSWER_FORMAT = ANSWER_FORMAT_P3      
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_ORIG
    TASK_PROMPT = TASK_PROMPT_ORIG
    ANSWER_FORMAT = ANSWER_FORMAT_ORIG


# ============================================================
# BUILD PROMPT — SAME FOR BOTH MODELS
# ============================================================

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


# ============================================================
# RUN INFERENCE (UNCHANGED LOGIC)
# ============================================================

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
    """Generate answers for a list of FinQASample inputs."""
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
