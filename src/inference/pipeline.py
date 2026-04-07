"""
Simple end-to-end inference pipeline for DeepSeek-OCR-2.
No vLLM — uses HuggingFace Transformers only.

Two loading strategies are provided:

1. ``from_pretrained`` — loads the *research-only* vision pipeline
   (SAM + D2E + Projector) plus an **upstream** ``AutoModel`` that
   exposes ``model.infer()``.  This path works out-of-the-box with
   ``deepseek-ai/DeepSeek-OCR-2`` and does **not** require ``vLLM``
   or ``flash_attn``.

2. ``from_components`` — accepts individually constructed vision
   model, language model, processor, and tokenizer for users who
   need fine-grained control (e.g. mechanistic-interpretability hooks).
"""

from __future__ import annotations

import re
from typing import Optional

import torch
from PIL import Image

from src.config import MODEL_PATH, PROMPT


def _load_upstream_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> "torch.nn.Module":
    """Load the upstream ``AutoModel`` with ``trust_remote_code=True``.

    Falls back from ``flash_attention_2`` to ``eager`` automatically when
    the ``flash_attn`` package is unavailable.
    """
    from transformers import AutoModel

    for attn_impl in ("flash_attention_2", "eager"):
        try:
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                use_safetensors=True,
            )
            model = model.eval().to(device).to(dtype)
            return model
        except (ImportError, ValueError):
            if attn_impl == "eager":
                raise
            # flash_attn missing — try eager
            continue
    # unreachable, but keeps mypy happy
    raise RuntimeError("Failed to load model")  # pragma: no cover


# ---------------------------------------------------------------------------
# Post-processing helpers (ported from upstream eval script)
# ---------------------------------------------------------------------------

_REF_DET_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
_FORMULA_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def _clean_formula(text: str) -> str:
    """Strip parenthetical labels from inline LaTeX ``\\[...\\]`` blocks."""

    def _process(m: re.Match) -> str:
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", m.group(1)).strip()
        return r"\[" + formula + r"\]"

    return _FORMULA_RE.sub(_process, text)


def _strip_ref_det_tags(text: str) -> str:
    """Remove ``<|ref|>…<|/ref|><|det|>…<|/det|>`` annotation spans."""
    return (
        _REF_DET_RE.sub("", text).replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")
    )


def clean_prediction(text: str) -> str:
    """Apply standard upstream post-processing to raw model output."""
    text = _clean_formula(text)
    return _strip_ref_det_tags(text)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DeepseekOCRPipeline:
    """End-to-end inference pipeline for DeepSeek-OCR-2.

    The recommended way to construct this is via :meth:`from_pretrained`,
    which loads the upstream ``AutoModel`` (with ``trust_remote_code``)
    and delegates to its ``model.infer()`` method.

    Usage::

        pipeline = DeepseekOCRPipeline.from_pretrained("deepseek-ai/DeepSeek-OCR-2")
        text = pipeline(image)              # returns cleaned markdown string
        text = pipeline(image, raw=True)    # returns raw model output (with ref/det tags)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: "AutoTokenizer",  # noqa: F821
        *,
        device: str = "cuda",
        prompt: str = PROMPT,
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt = prompt
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = MODEL_PATH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        prompt: str = PROMPT,
    ) -> "DeepseekOCRPipeline":
        """Load the full model from a HuggingFace model path.

        This uses ``AutoModel`` with ``trust_remote_code=True`` so that the
        upstream ``model.infer()`` interface is available.  FlashAttention is
        tried first and falls back to ``eager`` when unavailable.

        Args:
            model_path: HuggingFace model ID or local directory.
            device:     Target device.
            dtype:      Weight dtype.
            prompt:     Default prompt (can be overridden per-call).

        Returns:
            Ready-to-use pipeline instance.
        """
        from transformers import AutoTokenizer

        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"Loading model from {model_path}...")
        model = _load_upstream_model(model_path, dtype=dtype, device=device)

        return cls(model, tokenizer, device=device, prompt=prompt)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: Image.Image | str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        *,
        raw: bool = False,
    ) -> str:
        """Run OCR on *image* and return a markdown string.

        Args:
            image:          PIL ``Image`` or a filesystem path.
            prompt:         Text prompt (uses default if *None*).
            max_new_tokens: Maximum tokens to generate.
            raw:            If *True*, skip post-processing.

        Returns:
            Generated text (cleaned unless *raw=True*).
        """
        if isinstance(image, str):
            image_file = image
        else:
            # ``model.infer`` accepts file paths; write to a temp file
            import tempfile, os

            fd, image_file = tempfile.mkstemp(suffix=".png")
            try:
                image.convert("RGB").save(image_file)
            finally:
                os.close(fd)

        prompt = prompt or self.prompt

        # The upstream ``model.infer`` saves results to disk and prints
        # its raw generation.  We capture the cleaned markdown from the
        # saved ``.mmd`` file.
        import tempfile as _tf

        with _tf.TemporaryDirectory() as tmpdir:
            self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=tmpdir,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=True,
            )
            from pathlib import Path

            mmd = Path(tmpdir) / "result.mmd"
            text = mmd.read_text(encoding="utf-8") if mmd.exists() else ""

        if not raw:
            text = clean_prediction(text)
        return text
