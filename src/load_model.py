"""Model and tokenizer initialization for the baseline.

Loads a causal language model and tokenizer (defaults to the Fin-R1
checkpoint) and applies small compatibility tweaks for padding behavior.

Primary API:
- ``load_baseline``: returns ``(model, tokenizer)`` ready for inference,
  with optional ``device_map`` and ``torch_dtype`` configuration.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling_qwen2_qstar import Qwen2ForCausalLM


DEFAULT_MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"  # official FinR1 checkpoint on Hugging Face


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
        Passed to ``from_pretrained`` to control device placement. ``"auto"`` will
        shard across available GPUs if possible.
    torch_dtype:
        Optional dtype override, e.g. ``torch.bfloat16`` to reduce memory usage.
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
        dtype=torch_dtype,
        device_map=device_map,
        use_auth_token=use_auth_token,
    )
    model.eval()
    return model, tokenizer

def load_qstar_for_training(
    model_name: str = DEFAULT_MODEL_ID,
    params: dict | None = None,
    *,
    device_map: str | dict | None = "auto",
    torch_dtype: torch.dtype | None = None,
    use_auth_token: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load a causal language model and tokenizer and add QSTAR-specific settings.
    Modified from repository quiet-star/quiet-star-train.py

    Parameters
    ----------
    model_name:
        Hugging Face model identifier. Defaults to the public FinR1 checkpoint.
    device_map:
        Passed to ``from_pretrained`` to control device placement. ``"auto"`` will
        shard across available GPUs if possible.
    torch_dtype:
        Optional dtype override, e.g. ``torch.bfloat16`` to reduce memory usage.
    use_auth_token:
        Personal access token for private models (if required).

    Returns
    -------
    (model, tokenizer):
        Tuple containing the causal LM and tokenizer instances.
    """

    if params is None:
        params = {}
    else:
        params = vars(params)
    
    # Get parameter values
    root_prefix = params.get("root_prefix", "..")
    n_ahead = params.get("n_ahead", 1)
    n_ahead_talk = params.get("n_ahead_talk", 1)
    n_passes = params.get("n_passes", 1)
    gumbel_temperature = params.get("gumbel_temperature", 1.0)

    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print("Loading model")
    if "Qwen" in model_name:
        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='auto',
            cache_dir=root_prefix + "/.cache",
        )
    else:
        raise NotImplementedError(f"QSTAR not implemented for {model_name}")
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add QSTAR-specific settings
    special_tokens_to_add = []
    special_tokens_to_add.append("<|startthought|>")
    special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.gumbel_temperature = gumbel_temperature

    model.train()
    return model, tokenizer