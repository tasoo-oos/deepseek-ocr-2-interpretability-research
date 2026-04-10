#!/usr/bin/env python3
"""
Map D2E circuits on real OmniDocBench pages using annotated target regions.

This first-pass implementation intentionally uses the global view only. That
keeps the region-to-token geometry simple and makes the resulting circuit scores
easier to interpret.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image, ImageFilter

from src.analysis.interventions import InterventionManager
from src.analysis.real_circuit_mapping import (
    aggregate_circuit_results,
    bbox_to_grid_mask,
    extract_region_targets,
    map_bbox_to_padded_square,
    score_query_alignment,
    select_target_queries,
)
from src.benchmarks.omnidocbench import OmniDocBenchDataset
from src.models.deepseek_ocr import DeepseekOCRModel
from src.preprocessing.image_transforms import ImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-document circuit mapping on OmniDocBench pages")
    parser.add_argument("--dataset_root", required=True, help="Path to OmniDocBench root")
    parser.add_argument("--output_dir", default="output/real_circuit_mapping", help="Where to write JSON results")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--region_type", required=True, help="Substring match against layout label, e.g. table or formula")
    parser.add_argument("--filters", action="append", default=[], help="Dataset filter key=value, repeatable")
    parser.add_argument("--limit", type=int, default=25, help="Max pages to process")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_regions_per_page", type=int, default=1)
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,23")
    parser.add_argument("--attention_layer", type=int, default=23, help="Layer used to select target-focused queries")
    parser.add_argument("--top_queries", type=int, default=8, help="How many query slots to patch per region")
    parser.add_argument("--corruption_mode", choices=["mean_fill", "white_fill", "blur"], default="mean_fill")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def parse_filters(values: List[str]) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid filter {value!r}; expected key=value")
        key, raw = value.split("=", 1)
        filters[key] = raw
    return filters


def parse_layers(raw: str) -> List[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def corrupt_region(image: Image.Image, bbox, mode: str) -> Image.Image:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return image.copy()

    corrupted = image.copy()
    patch = corrupted.crop((x0, y0, x1, y1))

    if mode == "white_fill":
        patch = Image.new("RGB", patch.size, color=(255, 255, 255))
    elif mode == "blur":
        patch = patch.filter(ImageFilter.GaussianBlur(radius=12.0))
    else:
        pixels = list(patch.getdata())
        if pixels:
            mean = tuple(int(sum(channel) / len(channel)) for channel in zip(*pixels))
        else:
            mean = (255, 255, 255)
        patch = Image.new("RGB", patch.size, color=mean)

    corrupted.paste(patch, (x0, y0))
    return corrupted


def run_global_d2e(
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    image: Image.Image,
    *,
    device: str,
) -> Dict[str, object]:
    inputs = processor.process_image(image)
    pixel_values = inputs["pixel_values"][0].to(
        device=device,
        dtype=next(model.sam_model.parameters()).dtype,
    )

    with torch.no_grad():
        sam_features = model.sam_model(pixel_values)
        query_outputs, attentions, hidden_states, _token_type_ids = model.qwen2_model(
            sam_features,
            output_attentions=True,
            output_hidden_states=True,
        )

    return {
        "query_outputs": query_outputs[0].detach(),
        "attentions": [layer[0].detach() for layer in attentions],
        "hidden_states": [state[0].detach() for state in hidden_states],
        "spatial_size": sam_features.shape[-1],
    }


def run_patched_global_d2e(
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    image: Image.Image,
    *,
    device: str,
    layer: int,
    position: int,
    patch_value: torch.Tensor,
) -> torch.Tensor:
    inputs = processor.process_image(image)
    pixel_values = inputs["pixel_values"][0].to(
        device=device,
        dtype=next(model.sam_model.parameters()).dtype,
    )

    with InterventionManager(model) as manager:
        manager.patch_activation(
            layer=layer,
            position=position,
            new_value=patch_value.unsqueeze(0),
            component="d2e",
        )
        with torch.no_grad():
            sam_features = model.sam_model(pixel_values)
            query_outputs = model.qwen2_model(
                sam_features,
                output_attentions=False,
                output_hidden_states=False,
            )
            if isinstance(query_outputs, tuple):
                query_outputs = query_outputs[0]

    return query_outputs[0].detach()


def process_sample(
    *,
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    sample,
    args: argparse.Namespace,
    layers: List[int],
) -> List[Dict[str, object]]:
    with Image.open(sample.image_path) as image:
        clean_image = image.convert("RGB")

    regions = extract_region_targets(
        sample.raw_entry,
        image_size=clean_image.size,
        page_size=(sample.width, sample.height),
        region_type=args.region_type,
        max_regions=args.max_regions_per_page,
    )
    if not regions:
        return []

    clean_run = run_global_d2e(model, processor, clean_image, device=args.device)
    spatial_size = int(clean_run["spatial_size"])
    n_image_tokens = spatial_size * spatial_size

    records: List[Dict[str, object]] = []
    for region in regions:
        corrupted_image = corrupt_region(clean_image, region.bbox, args.corruption_mode)
        corrupted_run = run_global_d2e(model, processor, corrupted_image, device=args.device)

        square_bbox = map_bbox_to_padded_square(
            region.bbox,
            image_size=clean_image.size,
            square_size=args.base_size,
        )
        box_mask = bbox_to_grid_mask(
            square_bbox,
            square_size=args.base_size,
            spatial_size=spatial_size,
        )
        query_indices, query_scores = select_target_queries(
            clean_run["attentions"][args.attention_layer],
            box_mask=box_mask,
            top_k=args.top_queries,
            n_image_tokens=n_image_tokens,
        )

        clean_queries = clean_run["query_outputs"]
        corrupted_queries = corrupted_run["query_outputs"]
        baseline_score = score_query_alignment(clean_queries, corrupted_queries, query_indices)

        results = []
        for layer in layers:
            clean_hidden = clean_run["hidden_states"][layer + 1]
            for query_idx in query_indices:
                full_position = n_image_tokens + query_idx
                patched_queries = run_patched_global_d2e(
                    model,
                    processor,
                    corrupted_image,
                    device=args.device,
                    layer=layer,
                    position=full_position,
                    patch_value=clean_hidden[full_position],
                )
                patched_score = score_query_alignment(clean_queries, patched_queries, query_indices)
                results.append(
                    {
                        "layer": layer,
                        "query_idx": query_idx,
                        "full_position": full_position,
                        "baseline_score": baseline_score,
                        "patched_score": patched_score,
                        "impact": patched_score - baseline_score,
                    }
                )

        results.sort(key=lambda item: item["impact"], reverse=True)
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "page_attributes": sample.page_attributes,
                "region_label": region.label,
                "region_bbox": [round(v, 2) for v in region.bbox],
                "region_det_index": region.det_index,
                "attention_layer": args.attention_layer,
                "query_indices": query_indices,
                "query_selection_scores": [float(x) for x in query_scores.tolist()],
                "baseline_score": baseline_score,
                "results": results,
            }
        )

    return records


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    filters = parse_filters(args.filters)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OmniDocBenchDataset.from_dataset_root(
        args.dataset_root,
        filters=filters or None,
        limit=args.limit,
        offset=args.offset,
    )

    dtype = resolve_dtype(args.dtype)
    print(f"Loading model from {args.model_path}...")
    model = DeepseekOCRModel.from_pretrained(
        args.model_path,
        use_language_model=False,
        output_attentions=True,
        output_hidden_states=True,
        device=args.device,
        dtype=dtype,
    )
    model.eval()

    processor = ImageProcessor(base_size=args.base_size, crop_mode=False)

    all_records: List[Dict[str, object]] = []
    for index, sample in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] {sample.sample_id}")
        sample_records = process_sample(
            model=model,
            processor=processor,
            sample=sample,
            args=args,
            layers=layers,
        )
        all_records.extend(sample_records)

    summary = {
        "region_type": args.region_type,
        "dataset_root": args.dataset_root,
        "num_samples_requested": len(dataset),
        "num_regions_processed": len(all_records),
        "layers": layers,
        "top_queries": args.top_queries,
        "attention_layer": args.attention_layer,
        "corruption_mode": args.corruption_mode,
        "aggregated_results": aggregate_circuit_results(all_records),
        "records": all_records,
    }

    json_path = output_dir / "circuit_mapping_summary.json"
    md_path = output_dir / "circuit_mapping_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_summary(summary), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def render_markdown_summary(summary: Dict[str, object]) -> str:
    lines = [
        "# Real Circuit Mapping Summary",
        "",
        f"- Region type: `{summary['region_type']}`",
        f"- Regions processed: `{summary['num_regions_processed']}`",
        f"- Layers searched: `{','.join(str(x) for x in summary['layers'])}`",
        f"- Attention layer for query selection: `{summary['attention_layer']}`",
        f"- Corruption mode: `{summary['corruption_mode']}`",
        "",
        "## Top Aggregated Sites",
        "",
        "| layer | query | count | mean impact | mean baseline | mean patched |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in summary["aggregated_results"][:20]:
        lines.append(
            "| {layer} | {query_idx} | {count} | {mean_impact:.4f} | {mean_baseline_score:.4f} | {mean_patched_score:.4f} |".format(
                **item
            )
        )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
