"""Hugging Face inference wrapper for the original DeepSeek-OCR model."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .pipeline import clean_prediction


DEEPSEEK_OCR_V1_MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
DEEPSEEK_OCR_V1_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a dtype CLI string to a PyTorch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class DeepSeekOCRV1Pipeline:
    """Thin wrapper around the official upstream HF ``model.infer`` API.

    Defaults follow ``DeepSeek-OCR-hf/run_dpsk_ocr.py``:
    ``base_size=1024``, ``image_size=640``, ``crop_mode=True``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        test_compress: bool = True,
        prompt: str = DEEPSEEK_OCR_V1_PROMPT,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.test_compress = test_compress
        self.prompt = prompt

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEEPSEEK_OCR_V1_MODEL_PATH,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ) -> "DeepSeekOCRV1Pipeline":
        """Load the upstream DeepSeek-OCR v1 model.

        The official example uses FlashAttention 2. This wrapper falls back
        to eager attention when ``flash_attn`` is unavailable.
        """
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        implementations = [attn_implementation]
        if attn_implementation != "eager":
            implementations.append("eager")

        last_error: Exception | None = None
        for implementation in implementations:
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    _attn_implementation=implementation,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
                break
            except (ImportError, ValueError) as exc:
                last_error = exc
                if implementation == implementations[-1]:
                    raise
                print(f"Falling back from {implementation!r} to 'eager': {exc}")
        else:
            raise RuntimeError("Failed to load DeepSeek-OCR v1 model") from last_error

        model = model.eval().to(device).to(dtype)
        return cls(model, tokenizer, **kwargs)

    def __call__(
        self,
        image: Image.Image | str,
        *,
        prompt: Optional[str] = None,
        raw: bool = False,
    ) -> str:
        """Run OCR on one image and return markdown."""
        if isinstance(image, str):
            image_file = image
            cleanup_path = None
        else:
            fd, image_file = tempfile.mkstemp(suffix=".png")
            cleanup_path = image_file
            try:
                image.convert("RGB").save(image_file)
            finally:
                os.close(fd)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self.model.infer(
                    self.tokenizer,
                    prompt=prompt or self.prompt,
                    image_file=image_file,
                    output_path=tmpdir,
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=self.crop_mode,
                    save_results=True,
                    test_compress=self.test_compress,
                )
                result = Path(tmpdir) / "result.mmd"
                text = result.read_text(encoding="utf-8") if result.exists() else ""
        finally:
            if cleanup_path is not None:
                Path(cleanup_path).unlink(missing_ok=True)

        return text if raw else clean_prediction(text)
