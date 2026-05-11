from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    use_safetensors: bool


def _torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def load_ocr2(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModel.from_pretrained(
        cfg.name,
        trust_remote_code=cfg.trust_remote_code,
        use_safetensors=cfg.use_safetensors,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=_torch_dtype(cfg.dtype),
    )
    _configure_generation_tokens(model, tokenizer)
    model = model.eval().to(cfg.device)
    return model, tokenizer


def _configure_generation_tokens(model, tokenizer) -> None:
    token_fields = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    for field, value in token_fields.items():
        if value is None:
            continue
        setattr(model.config, field, value)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, field, value)
