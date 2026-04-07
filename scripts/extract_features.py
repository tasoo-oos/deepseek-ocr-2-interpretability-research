#!/usr/bin/env python3
"""
Extract all intermediate activations from a DeepSeek-OCR-2 forward pass.

Usage:
    uv run python scripts/extract_features.py \
        --image_path input/example.jpg \
        --output_path features.pt \
        --sam_layers 0,5,11 \
        --d2e_layers 0,5,11,17,23

Then inspect:
    import torch
    f = torch.load("features.pt")
    print(list(f.keys()))
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Extract intermediate activations")
    parser.add_argument("--image_path", required=True, help="Input image path")
    parser.add_argument("--output_path", default="features.pt",
                        help="Output file for saved activations")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2",
                        help="HuggingFace model path or local directory")
    parser.add_argument("--sam_layers", default="0,5,11",
                        help="Comma-separated SAM layer indices to hook")
    parser.add_argument("--d2e_layers", default="0,5,11,17,23",
                        help="Comma-separated D2E layer indices to hook")
    parser.add_argument("--no_projector", action="store_true",
                        help="Skip projector hook")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    sam_layers = [int(x) for x in args.sam_layers.split(",")]
    d2e_layers = [int(x) for x in args.d2e_layers.split(",")]

    from src.models.deepseek_ocr import DeepseekOCRModel
    from src.analysis.feature_extractor import FeatureExtractor
    from src.preprocessing.image_transforms import ImageProcessor

    print(f"Loading model from {args.model_path}...")
    model = DeepseekOCRModel.from_pretrained(args.model_path, device=args.device)
    model.eval()

    print(f"Processing image: {args.image_path}")
    processor = ImageProcessor()
    image = Image.open(args.image_path).convert("RGB")
    inputs = processor.process_image(image)
    pixel_values = inputs["pixel_values"].to(args.device)
    images_crop = inputs["images_crop"].to(args.device)
    images_spatial_crop = inputs["images_spatial_crop"].to(args.device)

    print(f"Registering hooks — SAM layers: {sam_layers}, D2E layers: {d2e_layers}")
    extractor = FeatureExtractor(model)
    extractor.register_hooks(
        sam_layers=sam_layers,
        d2e_layers=d2e_layers,
        projector=not args.no_projector,
    )

    print("Running forward pass...")
    activations = extractor.extract(pixel_values, images_crop, images_spatial_crop)
    extractor.clear_hooks()

    print(f"Extracted activations:")
    for name, tensor in activations.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    torch.save(activations, args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
