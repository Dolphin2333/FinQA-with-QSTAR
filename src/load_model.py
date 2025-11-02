"""Helper utilities to initialize baseline language models."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Zyphra/FinR1-7B"  # official FinR1 checkpoint on Hugging Face


def load_baseline(
    model_name: str = DEFAULT_MODEL_ID,
    *,
    device_map: str | dict | None = "auto",
    torch_dtype: torch.dtype | None = None,
    use_auth_token: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load a causal language model and tokenizer for the FinQA baseline.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier. Defaults to the public FinR1 checkpoint.
    device_map:
        Passed to ``from_pretrained`` to control device placement. ``\"auto\"`` will
        shard across available GPUs if possible.
    torch_dtype:
        Optional dtype override, e.g. ``torch.float16`` to reduce memory usage.
    use_auth_token:
        Personal access token for private models (if required).

    Returns
    -------
    (model, tokenizer):
        Tuple containing the causal LM and tokenizer instances.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_auth_token=use_auth_token,
    )
    model.eval()
    return model, tokenizer
