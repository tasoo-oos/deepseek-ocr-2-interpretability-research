"""
Batch inference without vLLM — simple loop over images.
"""

from pathlib import Path
from typing import List, Optional, Union

from .pipeline import DeepseekOCRPipeline


def run_batch(
    pipeline: DeepseekOCRPipeline,
    image_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 2048,
) -> List[str]:
    """
    Run inference on a list of image paths.

    Args:
        pipeline:       Initialized DeepseekOCRPipeline.
        image_paths:    List of paths to input images.
        output_dir:     If provided, save .md results here.
        prompt:         Custom prompt (uses default if None).
        max_new_tokens: Max tokens per generation.

    Returns:
        List of generated text strings, one per image.
    """
    results = []
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        print(f"[{i + 1}/{len(image_paths)}] Processing {img_path.name}...")

        # Pass file path directly to pipeline — avoids temp-file round-trip.
        text = pipeline(str(img_path), prompt=prompt, max_new_tokens=max_new_tokens)
        results.append(text)

        if output_dir:
            out_file = output_dir / img_path.with_suffix(".md").name
            out_file.write_text(text, encoding="utf-8")

    return results
