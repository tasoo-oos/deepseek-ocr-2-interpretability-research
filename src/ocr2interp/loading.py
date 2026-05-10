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
    model = model.eval().to(cfg.device)
    return model, tokenizer
