#!/usr/bin/env python3
"""
Run ablation experiments on DeepSeek-OCR-2.

Usage:
    # Ablate a specific attention head in D2E layer 12:
    uv run python scripts/run_interventions.py \
        --image_path input/example.jpg \
        --ablate_head 12,7

    # Zero out query token outputs:
    uv run python scripts/run_interventions.py \
        --image_path input/example.jpg \
        --ablate_queries
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Run causal ablation experiments")
    parser.add_argument("--image_path", required=True, help="Input image path")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--ablate_head", type=str, default=None,
                        help="Ablate D2E head: format 'layer,head' (e.g. '12,7')")
    parser.add_argument("--ablate_queries", action="store_true",
                        help="Zero out all D2E query token outputs")
    parser.add_argument("--ablate_sam_head", type=str, default=None,
                        help="Ablate SAM attention head: format 'layer,head'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.models.deepseek_ocr import DeepseekOCRModel
    from src.analysis.interventions import InterventionManager
    from src.preprocessing.image_transforms import ImageProcessor

    print(f"Loading model from {args.model_path}...")
    model = DeepseekOCRModel.from_pretrained(args.model_path, device=args.device)
    model.eval()

    processor = ImageProcessor()
    image = Image.open(args.image_path).convert("RGB")
    inputs = processor.process_image(image)
    pixel_values = inputs["pixel_values"].to(args.device)
    images_crop = inputs["images_crop"].to(args.device)
    images_spatial_crop = inputs["images_spatial_crop"].to(args.device)

    # Baseline (no intervention)
    with torch.no_grad():
        baseline = model.get_multimodal_embeddings(pixel_values, images_crop, images_spatial_crop)
    baseline_norm = sum(e.norm().item() for e in baseline)
    print(f"Baseline embedding norm: {baseline_norm:.4f}")

    # Apply interventions
    with InterventionManager(model) as mgr:
        if args.ablate_head:
            layer, head = map(int, args.ablate_head.split(","))
            print(f"Ablating D2E layer {layer}, head {head}...")
            mgr.ablate_attention_head(layer=layer, head=head, component="d2e")

        if args.ablate_sam_head:
            layer, head = map(int, args.ablate_sam_head.split(","))
            print(f"Ablating SAM layer {layer}, head {head}...")
            mgr.ablate_attention_head(layer=layer, head=head, component="sam")

        if args.ablate_queries:
            print("Ablating all D2E query token outputs...")
            mgr.ablate_query_tokens()

        with torch.no_grad():
            ablated = model.get_multimodal_embeddings(pixel_values, images_crop, images_spatial_crop)
        ablated_norm = sum(e.norm().item() for e in ablated)

    print(f"Ablated embedding norm: {ablated_norm:.4f}")
    print(f"Norm change: {ablated_norm - baseline_norm:+.4f} ({(ablated_norm / baseline_norm - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
