#!/usr/bin/env python3
"""
Run real-document reading-order experiments on OmniDocBench pages.

Experiment A:
- final-layer spatial probe on real pages for local/global D2E paths
- per-page query-index correlations against raster-order page coordinates
- mid-layer query ablation on a 20-page subset

Experiment B:
- layer-by-layer spatial probe across the same 50 pages
- raw query-embedding probe before any forward pass
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.interventions import InterventionManager
from src.analysis.spatial_analysis import LinearSpatialProbe
from src.benchmarks.omnidocbench import OmniDocBenchDataset, OmniDocBenchSample
from src.models.deepseek_ocr import DeepseekOCRModel
from src.preprocessing.image_transforms import ImageProcessor


EXPERIMENT_A_DIR = Path("output/real_doc_reading_order")
EXPERIMENT_B_DIR = Path("output/query_ordering_emergence")

FINAL_LAYER = 23
ABLATION_LAYER = 12
ABLATION_START = 96
ABLATION_END = 128


@dataclass(frozen=True)
class SelectedPage:
    sample: OmniDocBenchSample
    bucket: str
    language: str
    data_source: str
    layout_label: str
    table_count: int
    table_area_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-document D2E ordering experiments")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("input/OmniDocBench"),
        help="OmniDocBench root containing OmniDocBench.json and images/",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR-2",
        help="Hugging Face model id or local snapshot directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--num_pages",
        type=int,
        default=50,
        help="Total number of real pages to analyze",
    )
    parser.add_argument(
        "--ablation_pages",
        type=int,
        default=20,
        help="Subset size for the layer-12 query ablation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Sampling seed for deterministic page selection",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def bbox_area(poly: Sequence[float]) -> float:
    if len(poly) < 8:
        return 0.0
    xs = poly[0::2]
    ys = poly[1::2]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def table_stats(entry: Mapping[str, object]) -> tuple[int, float]:
    page_info = entry.get("page_info", {}) if isinstance(entry, Mapping) else {}
    width = float(page_info.get("width") or 1.0)
    height = float(page_info.get("height") or 1.0)
    page_area = max(width * height, 1.0)

    count = 0
    area = 0.0
    for det in entry.get("layout_dets", []):
        if det.get("category_type") == "table":
            count += 1
            area += bbox_area(det.get("poly") or [])
    return count, area / page_area


def assign_bucket(sample: OmniDocBenchSample) -> str | None:
    attrs = sample.page_attributes or {}
    layout = attrs.get("layout")
    table_count, table_fraction = table_stats(sample.raw_entry)

    if table_count >= 1 and table_fraction >= 0.12:
        return "table_heavy"
    if layout in {"double_column", "three_column", "1andmore_column"}:
        return "multi_column"
    if layout == "single_column":
        return "single_column"
    return None


def build_selection_targets(num_pages: int) -> Dict[str, int]:
    base = {
        "single_column": num_pages // 3,
        "multi_column": num_pages // 3,
        "table_heavy": num_pages // 3,
    }
    for bucket in ("single_column", "multi_column", "table_heavy")[: num_pages - sum(base.values())]:
        base[bucket] += 1
    return base


def select_pages(dataset_root: Path, num_pages: int, seed: int) -> List[SelectedPage]:
    dataset = OmniDocBenchDataset.from_dataset_root(dataset_root)
    targets = build_selection_targets(num_pages)
    rng = random.Random(seed)

    buckets: Dict[str, List[SelectedPage]] = defaultdict(list)
    for sample in dataset:
        page_info = sample.raw_entry.get("page_info", {})
        width = int(page_info.get("width") or 0)
        height = int(page_info.get("height") or 0)
        if max(width, height) <= 768:
            continue

        bucket = assign_bucket(sample)
        if bucket is None:
            continue

        table_count, table_fraction = table_stats(sample.raw_entry)
        buckets[bucket].append(
            SelectedPage(
                sample=sample,
                bucket=bucket,
                language=str(sample.page_attributes.get("language", "unknown")),
                data_source=str(sample.page_attributes.get("data_source", "unknown")),
                layout_label=str(sample.page_attributes.get("layout", "unknown")),
                table_count=table_count,
                table_area_fraction=float(table_fraction),
            )
        )

    selected: List[SelectedPage] = []
    seen_ids: set[str] = set()
    for bucket, target in targets.items():
        candidates = list(buckets[bucket])
        rng.shuffle(candidates)

        bucket_selected: List[SelectedPage] = []
        for page in candidates:
            if page.sample.sample_id in seen_ids:
                continue
            bucket_selected.append(page)
            seen_ids.add(page.sample.sample_id)
            if len(bucket_selected) == target:
                break

        if len(bucket_selected) != target:
            raise RuntimeError(
                f"Could not satisfy selection target for {bucket}: "
                f"wanted {target}, found {len(bucket_selected)}"
            )
        selected.extend(bucket_selected)

    selected.sort(key=lambda page: (page.bucket, page.sample.image_name))
    return selected


def select_ablation_pages(pages: Sequence[SelectedPage], ablation_pages: int) -> List[SelectedPage]:
    targets = build_selection_targets(ablation_pages)
    by_bucket: Dict[str, List[SelectedPage]] = defaultdict(list)
    for page in pages:
        by_bucket[page.bucket].append(page)

    picked: List[SelectedPage] = []
    for bucket in ("single_column", "multi_column", "table_heavy"):
        picked.extend(by_bucket[bucket][: targets[bucket]])
    return picked


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x_centered * y_centered).sum().item() / denom.item())


def raster_coords(width_steps: int, height_steps: int) -> torch.Tensor:
    if width_steps <= 0 or height_steps <= 0:
        raise ValueError("width_steps and height_steps must be positive")
    xs = torch.linspace(0.0, 1.0, width_steps)
    ys = torch.linspace(0.0, 1.0, height_steps)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)


def make_global_targets() -> torch.Tensor:
    return raster_coords(16, 16)


def make_local_targets(num_width_tiles: int, num_height_tiles: int) -> torch.Tensor:
    full_width = max(num_width_tiles * 12, 1)
    full_height = max(num_height_tiles * 12, 1)
    xs = torch.linspace(0.0, 1.0, full_width)
    ys = torch.linspace(0.0, 1.0, full_height)

    coords: List[torch.Tensor] = []
    for patch_y in range(num_height_tiles):
        for patch_x in range(num_width_tiles):
            patch_xs = xs[patch_x * 12 : (patch_x + 1) * 12]
            patch_ys = ys[patch_y * 12 : (patch_y + 1) * 12]
            grid_y, grid_x = torch.meshgrid(patch_ys, patch_xs, indexing="ij")
            coords.append(torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1))
    return torch.cat(coords, dim=0)


def move_inputs_to_device(
    inputs: Mapping[str, torch.Tensor],
    *,
    device: str,
    sam_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    return {
        "pixel_values": inputs["pixel_values"].to(device=device, dtype=sam_dtype),
        "images_crop": inputs["images_crop"].to(device=device, dtype=sam_dtype),
        "images_spatial_crop": inputs["images_spatial_crop"].to(device=device),
    }


def split_query_half(full_hidden: torch.Tensor) -> torch.Tensor:
    seq_len = full_hidden.shape[1]
    n_image = seq_len // 2
    return full_hidden[:, n_image:, :]


def extract_path_queries(
    extractor: FeatureExtractor,
    *,
    layer: int,
) -> Dict[str, torch.Tensor]:
    sequence = extractor.get_activation_sequence(f"d2e_layer_{layer}")
    if not sequence:
        raise RuntimeError(f"No activations captured for d2e_layer_{layer}")

    if len(sequence) == 1:
        return {
            "global": split_query_half(sequence[0])[0].cpu(),
        }

    if len(sequence) != 2:
        raise RuntimeError(f"Expected 1 or 2 D2E calls, got {len(sequence)}")

    local_full, global_full = sequence
    return {
        "local": split_query_half(local_full).reshape(-1, local_full.shape[-1]).cpu(),
        "global": split_query_half(global_full)[0].cpu(),
    }


def compute_page_correlations(targets: torch.Tensor) -> Dict[str, float]:
    query_index = torch.arange(targets.shape[0], dtype=torch.float32)
    return {
        "corr_x": pearson(query_index, targets[:, 0]),
        "corr_y": pearson(query_index, targets[:, 1]),
    }


def cosine_drop(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    cosine = F.cosine_similarity(reference.float(), candidate.float(), dim=-1)
    return float((1.0 - cosine.mean()).item())


def summarize_directionality(clean: torch.Tensor, ablated: torch.Tensor) -> Dict[str, float]:
    if clean.ndim == 2:
        clean = clean.unsqueeze(0)
        ablated = ablated.unsqueeze(0)

    prefix_drops: List[float] = []
    suffix_drops: List[float] = []
    ratios: List[float] = []
    for clean_seq, ablated_seq in zip(clean, ablated):
        prefix_drop = cosine_drop(clean_seq[:ABLATION_START], ablated_seq[:ABLATION_START])
        suffix_drop = cosine_drop(clean_seq[ABLATION_END:], ablated_seq[ABLATION_END:])
        prefix_drops.append(prefix_drop)
        suffix_drops.append(suffix_drop)
        ratios.append(suffix_drop / max(prefix_drop, 1e-9))

    mean_prefix = sum(prefix_drops) / len(prefix_drops)
    mean_suffix = sum(suffix_drops) / len(suffix_drops)
    return {
        "prefix_drop": mean_prefix,
        "suffix_drop": mean_suffix,
        "suffix_over_prefix": mean_suffix / max(mean_prefix, 1e-9),
        "suffix_gt_prefix": sum(float(s > p) for p, s in zip(prefix_drops, suffix_drops)) / len(prefix_drops),
    }


def run_path_d2e(
    model: DeepseekOCRModel,
    path_tensor: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        sam_features = model.sam_model(path_tensor)
        query_outputs = model.qwen2_model(
            sam_features,
            output_attentions=False,
            output_hidden_states=False,
        )
        if isinstance(query_outputs, tuple):
            query_outputs = query_outputs[0]
    return query_outputs.detach().cpu()


def run_path_d2e_ablation(
    model: DeepseekOCRModel,
    path_tensor: torch.Tensor,
) -> torch.Tensor:
    with InterventionManager(model) as manager:
        manager.ablate_query_states_in_layer(
            layer=ABLATION_LAYER,
            start_idx=ABLATION_START,
            end_idx=ABLATION_END,
        )
        with torch.no_grad():
            sam_features = model.sam_model(path_tensor)
            query_outputs = model.qwen2_model(
                sam_features,
                output_attentions=False,
                output_hidden_states=False,
            )
            if isinstance(query_outputs, tuple):
                query_outputs = query_outputs[0]
    return query_outputs.detach().cpu()


def fit_probe(features: torch.Tensor, targets: torch.Tensor) -> LinearSpatialProbe:
    return LinearSpatialProbe(l2_penalty=1e-4).fit(features, targets)


def compute_probe_metrics(features: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probe = fit_probe(features, targets)
    metrics = probe.evaluate(features, targets)
    return {
        "mse": float(metrics.mse),
        "r2_x": float(metrics.r2[0].item()),
        "r2_y": float(metrics.r2[1].item()),
    }


def compute_probe_metrics_holdout(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int = 0,
    train_fraction: float = 0.8,
) -> Dict[str, float]:
    if features.shape[0] < 8:
        return compute_probe_metrics(features, targets)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    train_count = max(1, min(features.shape[0] - 1, int(round(features.shape[0] * train_fraction))))
    train_idx = permutation[:train_count]
    test_idx = permutation[train_count:]
    probe = fit_probe(features[train_idx], targets[train_idx])
    metrics = probe.evaluate(features[test_idx], targets[test_idx])
    return {
        "mse": float(metrics.mse),
        "r2_x": float(metrics.r2[0].item()),
        "r2_y": float(metrics.r2[1].item()),
    }


def aggregate_correlations(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    corr_x = [float(row["corr_x"]) for row in rows]
    corr_y = [float(row["corr_y"]) for row in rows]
    return {
        "pages": len(rows),
        "mean_corr_x": sum(corr_x) / max(len(corr_x), 1),
        "mean_corr_y": sum(corr_y) / max(len(corr_y), 1),
        "dominant_y_fraction": (
            sum(float(abs(y) > abs(x)) for x, y in zip(corr_x, corr_y)) / max(len(corr_x), 1)
        ),
        "positive_y_fraction": sum(float(y > 0.0) for y in corr_y) / max(len(corr_y), 1),
    }


def aggregate_ablations(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    prefix = [float(row["prefix_drop"]) for row in rows]
    suffix = [float(row["suffix_drop"]) for row in rows]
    dominant = [float(row["suffix_gt_prefix"]) for row in rows]
    mean_prefix = sum(prefix) / max(len(prefix), 1)
    mean_suffix = sum(suffix) / max(len(suffix), 1)
    return {
        "pages": len(rows),
        "mean_prefix_drop": mean_prefix,
        "mean_suffix_drop": mean_suffix,
        "suffix_over_prefix": mean_suffix / max(mean_prefix, 1e-9),
        "suffix_gt_prefix_fraction": sum(dominant) / max(len(dominant), 1),
    }


def run_experiment_a(
    *,
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    pages: Sequence[SelectedPage],
    ablation_pages: Sequence[SelectedPage],
    dataset_root: Path,
    device: str,
) -> Dict[str, object]:
    extractor = FeatureExtractor(model)
    extractor.register_hooks(d2e_layers=[FINAL_LAYER], projector=False)

    probe_features: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    probe_targets: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    probe_pages: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    page_features: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    correlations: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    ablations: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    dataset_meta: List[Dict[str, object]] = []
    ablation_ids = {page.sample.sample_id for page in ablation_pages}

    for index, page in enumerate(pages, start=1):
        print(f"[A {index}/{len(pages)}] {page.sample.sample_id}")
        with Image.open(page.sample.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = processor.process_image(image)

        grid_w, grid_h = [int(x) for x in inputs["images_spatial_crop"][0, 0].tolist()]
        if grid_w * grid_h <= 1:
            raise RuntimeError(f"Selected page did not yield local crops: {page.sample.sample_id}")

        device_inputs = move_inputs_to_device(
            inputs,
            device=device,
            sam_dtype=next(model.sam_model.parameters()).dtype,
        )
        extractor.extract(
            device_inputs["pixel_values"],
            device_inputs["images_crop"],
            device_inputs["images_spatial_crop"],
        )
        path_queries = extract_path_queries(extractor, layer=FINAL_LAYER)

        path_targets = {
            "global": make_global_targets(),
            "local": make_local_targets(grid_w, grid_h),
        }

        dataset_meta.append(
            {
                "sample_id": page.sample.sample_id,
                "image_path": str(page.sample.image_path),
                "bucket": page.bucket,
                "layout_label": page.layout_label,
                "language": page.language,
                "data_source": page.data_source,
                "grid_tiles": [grid_w, grid_h],
                "table_count": page.table_count,
                "table_area_fraction": round(page.table_area_fraction, 6),
            }
        )

        for path_name, features in path_queries.items():
            targets = path_targets[path_name]
            if features.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    f"Token/target mismatch on {page.sample.sample_id} {path_name}: "
                    f"{features.shape[0]} vs {targets.shape[0]}"
                )
            probe_features[path_name]["overall"].append(features)
            probe_features[path_name][page.bucket].append(features)
            probe_targets[path_name]["overall"].append(targets)
            probe_targets[path_name][page.bucket].append(targets)
            probe_pages[path_name]["overall"] += 1
            probe_pages[path_name][page.bucket] += 1
            page_features[path_name].append(
                {
                    "sample_id": page.sample.sample_id,
                    "bucket": page.bucket,
                    "features": features,
                }
            )

        if page.sample.sample_id in ablation_ids:
            local_tensor = device_inputs["images_crop"][0, 0]
            global_tensor = device_inputs["pixel_values"][0]
            clean_global = run_path_d2e(model, global_tensor)
            ablated_global = run_path_d2e_ablation(model, global_tensor)
            global_metrics = summarize_directionality(clean_global[0], ablated_global[0])
            ablations["global"].append(
                {
                    "sample_id": page.sample.sample_id,
                    "bucket": page.bucket,
                    **global_metrics,
                }
            )

            clean_local = run_path_d2e(model, local_tensor)
            ablated_local = run_path_d2e_ablation(model, local_tensor)
            local_metrics = summarize_directionality(clean_local, ablated_local)
            ablations["local"].append(
                {
                    "sample_id": page.sample.sample_id,
                    "bucket": page.bucket,
                    **local_metrics,
                }
            )

        torch.cuda.empty_cache()

    extractor.clear_hooks()

    probe_summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    overall_probes: Dict[str, LinearSpatialProbe] = {}
    for path_name, bucket_features in probe_features.items():
        for bucket, feature_list in bucket_features.items():
            features = torch.cat(feature_list, dim=0)
            targets = torch.cat(probe_targets[path_name][bucket], dim=0)
            metrics = compute_probe_metrics(features, targets)
            metrics["pages"] = probe_pages[path_name][bucket]
            metrics["tokens"] = int(features.shape[0])
            probe_summary[path_name][bucket] = metrics
            if bucket == "overall":
                overall_probes[path_name] = fit_probe(features, targets)

    for path_name, rows in page_features.items():
        probe = overall_probes[path_name]
        for row in rows:
            predicted = probe.predict(row["features"])
            corr = compute_page_correlations(predicted)
            correlations[path_name].append(
                {
                    "sample_id": row["sample_id"],
                    "bucket": row["bucket"],
                    "corr_x": corr["corr_x"],
                    "corr_y": corr["corr_y"],
                }
            )

    correlation_summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for path_name, rows in correlations.items():
        correlation_summary[path_name]["overall"] = aggregate_correlations(rows)
        for bucket in ("single_column", "multi_column", "table_heavy"):
            bucket_rows = [row for row in rows if row["bucket"] == bucket]
            correlation_summary[path_name][bucket] = aggregate_correlations(bucket_rows)

    ablation_summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for path_name, rows in ablations.items():
        ablation_summary[path_name]["overall"] = aggregate_ablations(rows)
        for bucket in ("single_column", "multi_column", "table_heavy"):
            bucket_rows = [row for row in rows if row["bucket"] == bucket]
            ablation_summary[path_name][bucket] = aggregate_ablations(bucket_rows)

    return {
        "dataset_root": str(dataset_root),
        "selected_pages": dataset_meta,
        "probe_summary": probe_summary,
        "correlation_summary": correlation_summary,
        "ablation_summary": ablation_summary,
        "correlation_pages": correlations,
        "ablation_pages": ablations,
    }


def write_experiment_a_summary(output_dir: Path, result: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    selected_pages = result["selected_pages"]
    bucket_counts = Counter(page["bucket"] for page in selected_pages)
    language_counts = Counter(page["language"] for page in selected_pages)
    source_counts = Counter(page["data_source"] for page in selected_pages)
    probe_summary = result["probe_summary"]
    correlation_summary = result["correlation_summary"]
    ablation_summary = result["ablation_summary"]

    lines = [
        "# Real-Document Reading Order Summary",
        "",
        "## Dataset",
        "",
        f"- Source: `{result['dataset_root']}`",
        f"- Pages analyzed: `{len(selected_pages)}`",
        f"- Layout buckets: `single_column={bucket_counts['single_column']}`, `multi_column={bucket_counts['multi_column']}`, `table_heavy={bucket_counts['table_heavy']}`",
        f"- Languages: `{dict(language_counts)}`",
        f"- Top data sources: `{dict(source_counts.most_common(5))}`",
        "- Bucket rule: `table_heavy` if any table covers >= 12% of page area; else `multi_column` for double/three/1+ columns; else `single_column`.",
        "",
        "## Probe Metrics",
        "",
        "| path | bucket | pages | tokens | mse | r2_x | r2_y |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for path_name in ("global", "local"):
        for bucket in ("overall", "single_column", "multi_column", "table_heavy"):
            row = probe_summary[path_name][bucket]
            lines.append(
                f"| {path_name} | {bucket} | {int(row['pages'])} | {int(row['tokens'])} | "
                f"{row['mse']:.6f} | {row['r2_x']:.4f} | {row['r2_y']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Query-Index Correlations",
            "",
            "- Correlations are computed on coordinates predicted by the overall path probe fitted on raster-order targets.",
            "",
            "| path | bucket | pages | mean corr(query,x) | mean corr(query,y) | |y|>|x| frac | y>0 frac |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for path_name in ("global", "local"):
        for bucket in ("overall", "single_column", "multi_column", "table_heavy"):
            row = correlation_summary[path_name][bucket]
            lines.append(
                f"| {path_name} | {bucket} | {int(row['pages'])} | {row['mean_corr_x']:.3f} | "
                f"{row['mean_corr_y']:.3f} | {row['dominant_y_fraction']:.2f} | {row['positive_y_fraction']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Layer-12 Query Ablation (96:128 on 20 pages)",
            "",
            "| path | bucket | pages | prefix drop | suffix drop | suffix/prefix | suffix>prefix frac |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for path_name in ("global", "local"):
        for bucket in ("overall", "single_column", "multi_column", "table_heavy"):
            row = ablation_summary[path_name][bucket]
            lines.append(
                f"| {path_name} | {bucket} | {int(row['pages'])} | {row['mean_prefix_drop']:.4f} | "
                f"{row['mean_suffix_drop']:.4f} | {row['suffix_over_prefix']:.2f} | {row['suffix_gt_prefix_fraction']:.2f} |"
            )

    global_probe = probe_summary["global"]["overall"]
    global_corr = correlation_summary["global"]["overall"]
    global_multi = correlation_summary["global"]["multi_column"]
    lines.extend(
        [
            "",
            "## What Changed Vs Synthetic",
            "",
            f"- Synthetic reference from `output/causal_token_research`: `mse=0.001018`, `r2_x=0.9567`, `r2_y=0.9981`, `corr_y` in `[0.162, 0.343]`, `corr_x` in `[-0.214, -0.069]`.",
            f"- Global-path real pages: `mse={global_probe['mse']:.6f}`, `r2_x={global_probe['r2_x']:.4f}`, `r2_y={global_probe['r2_y']:.4f}`.",
            f"- Global-path real correlations: overall `corr_x={global_corr['mean_corr_x']:.3f}`, `corr_y={global_corr['mean_corr_y']:.3f}`; multi-column `corr_x={global_multi['mean_corr_x']:.3f}`, `corr_y={global_multi['mean_corr_y']:.3f}`.",
            "- Probe targets here are raster-order SAM grid coordinates on real pages, so the comparison to the synthetic attention-center probe should be read as directional rather than perfectly like-for-like.",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment_b(
    *,
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    pages: Sequence[SelectedPage],
    device: str,
) -> Dict[str, object]:
    layers = [0, 3, 6, 9, 12, 15, 18, 21, 23]
    extractor = FeatureExtractor(model)
    extractor.register_hooks(d2e_layers=layers, projector=False)

    features_by_layer: Dict[str, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    targets_by_path: Dict[str, List[torch.Tensor]] = defaultdict(list)

    for index, page in enumerate(pages, start=1):
        print(f"[B {index}/{len(pages)}] {page.sample.sample_id}")
        with Image.open(page.sample.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = processor.process_image(image)

        grid_w, grid_h = [int(x) for x in inputs["images_spatial_crop"][0, 0].tolist()]
        if grid_w * grid_h <= 1:
            raise RuntimeError(f"Selected page did not yield local crops: {page.sample.sample_id}")
        device_inputs = move_inputs_to_device(
            inputs,
            device=device,
            sam_dtype=next(model.sam_model.parameters()).dtype,
        )
        extractor.extract(
            device_inputs["pixel_values"],
            device_inputs["images_crop"],
            device_inputs["images_spatial_crop"],
        )

        local_targets = make_local_targets(grid_w, grid_h)
        global_targets = make_global_targets()
        targets_by_path["local"].append(local_targets)
        targets_by_path["global"].append(global_targets)

        for layer in layers:
            path_queries = extract_path_queries(extractor, layer=layer)
            features_by_layer["global"][layer].append(path_queries["global"])
            features_by_layer["local"][layer].append(path_queries["local"])

        torch.cuda.empty_cache()

    extractor.clear_hooks()

    layer_metrics: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for path_name in ("global", "local"):
        targets = torch.cat(targets_by_path[path_name], dim=0)
        for layer in layers:
            features = torch.cat(features_by_layer[path_name][layer], dim=0)
            layer_metrics[path_name][layer] = compute_probe_metrics(features, targets)

    embedding_metrics = {
        "global": compute_probe_metrics_holdout(
            model.qwen2_model.query_1024.weight.detach().cpu(),
            make_global_targets(),
            seed=11,
        ),
        "local": compute_probe_metrics_holdout(
            model.qwen2_model.query_768.weight.detach().cpu(),
            raster_coords(12, 12),
            seed=11,
        ),
    }

    final_mean = 0.5 * (
        layer_metrics["global"][23]["r2_y"] + layer_metrics["local"][23]["r2_y"]
    )
    raw_mean = 0.5 * (
        embedding_metrics["global"]["r2_y"] + embedding_metrics["local"]["r2_y"]
    )
    if raw_mean > 0.9:
        verdict = "embedded"
    elif raw_mean < 0.5 and final_mean > 0.9:
        verdict = "emergent"
    else:
        verdict = "mixed"

    return {
        "layers": layers,
        "layer_metrics": layer_metrics,
        "embedding_metrics": embedding_metrics,
        "verdict": verdict,
    }


def write_experiment_b_summary(output_dir: Path, result: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Query Ordering Emergence Summary",
        "",
        "## Layer Sweep",
        "",
        "| path | layer | mse | r2_x | r2_y |",
        "|---|---:|---:|---:|---:|",
    ]
    for path_name in ("global", "local"):
        for layer in result["layers"]:
            row = result["layer_metrics"][path_name][layer]
            lines.append(
                f"| {path_name} | {layer} | {row['mse']:.6f} | {row['r2_x']:.4f} | {row['r2_y']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Raw Query Embeddings",
            "",
            "- Raw embedding metrics use a deterministic 80/20 holdout split to avoid trivial interpolation from the 896-d embedding space.",
            "",
            "| path | mse | r2_x | r2_y |",
            "|---|---:|---:|---:|",
        ]
    )
    for path_name in ("global", "local"):
        row = result["embedding_metrics"][path_name]
        lines.append(f"| {path_name} | {row['mse']:.6f} | {row['r2_x']:.4f} | {row['r2_y']:.4f} |")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Classification: `{result['verdict']}`",
            f"- Global path: raw `r2_y={result['embedding_metrics']['global']['r2_y']:.4f}` -> layer 23 `r2_y={result['layer_metrics']['global'][23]['r2_y']:.4f}`.",
            f"- Local path: raw `r2_y={result['embedding_metrics']['local']['r2_y']:.4f}` -> layer 23 `r2_y={result['layer_metrics']['local'][23]['r2_y']:.4f}`.",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    pages = select_pages(dataset_root, num_pages=args.num_pages, seed=args.seed)
    ablation_pages = select_ablation_pages(pages, args.ablation_pages)

    print(f"Loading model from {args.model_path}")
    model = DeepseekOCRModel.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation="sdpa",
        output_attentions=False,
        output_hidden_states=False,
    )
    model.eval()
    processor = ImageProcessor(crop_mode=True)

    experiment_a = run_experiment_a(
        model=model,
        processor=processor,
        pages=pages,
        ablation_pages=ablation_pages,
        dataset_root=dataset_root,
        device=args.device,
    )
    write_experiment_a_summary(EXPERIMENT_A_DIR, experiment_a)
    print(f"Wrote {EXPERIMENT_A_DIR / 'summary.md'}")

    experiment_b = run_experiment_b(
        model=model,
        processor=processor,
        pages=pages,
        device=args.device,
    )
    write_experiment_b_summary(EXPERIMENT_B_DIR, experiment_b)
    print(f"Wrote {EXPERIMENT_B_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
