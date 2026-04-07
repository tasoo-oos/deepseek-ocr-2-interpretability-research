#!/usr/bin/env python3
"""
Simple image-to-text inference with DeepSeek-OCR-2 (no vLLM).

Usage:
    uv run python scripts/simple_inference.py \
        --image_path input/example.jpg \
        --prompt "<image>\nConvert the document to markdown."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 simple inference")
    parser.add_argument("--image_path", required=True, help="Input image path")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument(
        "--prompt", default="<image>\n<|grounding|>Convert the document to markdown."
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Save output text to this file (optional)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from src.inference.pipeline import DeepseekOCRPipeline

    print(f"Loading pipeline from {args.model_path}...")
    pipeline = DeepseekOCRPipeline.from_pretrained(
        model_path=args.model_path,
        device=args.device,
    )

    print(f"Processing {args.image_path}...")
    # Pass file path directly — avoids redundant temp-file round-trip.
    result = pipeline(
        args.image_path, prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )

    print("\n" + "=" * 60)
    print("OUTPUT:")
    print("=" * 60)
    print(result)

    if args.output_path:
        Path(args.output_path).write_text(result, encoding="utf-8")
        print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
