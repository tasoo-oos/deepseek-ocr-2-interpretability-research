#!/usr/bin/env python3
"""Run official DeepSeek-OCR v1 HF inference on OmniDocBench pages.

This follows DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py:
  model.infer(..., base_size=1024, image_size=640, crop_mode=True)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.omnidocbench import OmniDocBenchDataset, OmniDocBenchRunner
from src.inference.deepseek_ocr_v1 import (
    DEEPSEEK_OCR_V1_MODEL_PATH,
    DEEPSEEK_OCR_V1_PROMPT,
    DeepSeekOCRV1Pipeline,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path("input/OmniDocBench"))
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", default=DEEPSEEK_OCR_V1_MODEL_PATH)
    parser.add_argument("--prompt", default=DEEPSEEK_OCR_V1_PROMPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--crop_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_compress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_dataset(args: argparse.Namespace) -> OmniDocBenchDataset:
    if args.manifest_path is not None:
        return OmniDocBenchDataset.from_manifest(
            args.manifest_path,
            image_root=args.image_root,
            limit=args.limit,
            offset=args.offset,
        )
    return OmniDocBenchDataset.from_dataset_root(
        args.dataset_root,
        limit=args.limit,
        offset=args.offset,
    )


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(args)
    runner = OmniDocBenchRunner(dataset)

    if args.dry_run:
        records = runner.run(
            lambda image, prompt=None: "",
            args.output_dir,
            prompt=args.prompt,
            overwrite=args.overwrite,
            dry_run=True,
        )
        print(json.dumps({"pages": len(records), "output_dir": str(args.output_dir)}, indent=2))
        return

    pipeline = DeepSeekOCRV1Pipeline.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        test_compress=args.test_compress,
        prompt=args.prompt,
    )
    records = runner.run(
        lambda image, prompt=None: pipeline(image, prompt=prompt),
        args.output_dir,
        prompt=args.prompt,
        overwrite=args.overwrite,
        dry_run=False,
    )
    print(json.dumps({"pages": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
