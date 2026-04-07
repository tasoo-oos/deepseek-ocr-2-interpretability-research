#!/usr/bin/env python3
"""Bulk inference runner for OmniDocBench-formatted datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.omnidocbench import OmniDocBenchDataset, OmniDocBenchRunner
from src.config import MODEL_PATH, PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bulk DeepSeek-OCR-2 inference on OmniDocBench")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Folder with OmniDocBench.json and images/")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Path to OmniDocBench.json")
    parser.add_argument("--image_root", type=Path, default=None, help="Optional image root override")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for markdown outputs")
    parser.add_argument("--model_path", default=MODEL_PATH, help="HF model id or local checkpoint")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt used for every page")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Max number of pages to run")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many pages before running")
    parser.add_argument("--filter", action="append", default=[], help="Page filter in key=value form; repeatable")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing markdown outputs")
    parser.add_argument("--dry_run", action="store_true", help="Inspect dataset and planned outputs without loading the model")
    return parser.parse_args()


def parse_filters(items: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid filter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        filters[key] = value
    return filters


def resolve_dataset(args: argparse.Namespace) -> OmniDocBenchDataset:
    filters = parse_filters(args.filter)
    if args.dataset_root is not None:
        return OmniDocBenchDataset.from_dataset_root(
            args.dataset_root,
            filters=filters or None,
            limit=args.limit,
            offset=args.offset,
        )
    if args.manifest_path is None:
        raise ValueError("Provide either --dataset_root or --manifest_path")
    return OmniDocBenchDataset.from_manifest(
        args.manifest_path,
        image_root=args.image_root,
        filters=filters or None,
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
        summary = {
            "pages": len(dataset),
            "output_dir": str(args.output_dir),
            "first_outputs": [record["output_path"] for record in records[:5]],
        }
        print(json.dumps(summary, indent=2))
        return

    from src.inference.pipeline import DeepseekOCRPipeline

    pipeline = DeepseekOCRPipeline.from_pretrained(
        model_path=args.model_path,
        device=args.device,
    )
    records = runner.run(
        lambda image, prompt=None: pipeline(
            image,
            prompt=prompt or args.prompt,
            max_new_tokens=args.max_new_tokens,
        ),
        args.output_dir,
        prompt=args.prompt,
        overwrite=args.overwrite,
        dry_run=False,
    )
    print(json.dumps({"pages": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
