#!/usr/bin/env python3
"""Run query-to-layout tracing and D2E mask ablation experiments.

This script targets the first two mechanistic-interpretability experiments:

1. Causal query reading-order tracing.
2. Attention-mask/query-order ablations.

It uses OmniDocBench layout annotations to map D2E causal queries back to
document elements through query-to-visual attention.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from src.benchmarks.omnidocbench import OmniDocBenchDataset, OmniDocBenchSample
from src.models.deepseek_ocr import DeepseekOCRModel
from src.preprocessing.image_transforms import ImageProcessor


DEFAULT_OUTPUT_DIR = Path("output/query_trace_mask_ablation")


@dataclass(frozen=True)
class LayoutElement:
    order: int
    category: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]


@dataclass(frozen=True)
class QueryTrace:
    path: str
    query_index: int
    x: float
    y: float
    entropy: float
    topk_mass: float
    max_attention: float
    element_order: Optional[int]
    element_category: Optional[str]
    assignment: str
    element_center_distance: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path("input/OmniDocBench"))
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=None, help="Limit pages; omit for the full dataset")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample_strategy", choices=["manifest", "stratified"], default="manifest")
    parser.add_argument("--topk", type=int, default=12, help="Top visual tokens used for query source centers")
    parser.add_argument(
        "--ablation_pages",
        type=int,
        default=64,
        help="Number of pages used for expensive mask ablations; use -1 for all selected pages",
    )
    parser.add_argument(
        "--ablation_layer",
        type=int,
        default=-1,
        help="Attention layer used for ablation trace metrics; -1 uses final layer",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def poly_to_bbox(poly: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(poly) < 8:
        return None
    xs = [float(x) for x in poly[0::2]]
    ys = [float(y) for y in poly[1::2]]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def load_layout_elements(sample: OmniDocBenchSample) -> List[LayoutElement]:
    elements: List[LayoutElement] = []
    for det in sample.raw_entry.get("layout_dets", []):
        if det.get("ignore"):
            continue
        order = det.get("order")
        if order is None:
            continue
        bbox = poly_to_bbox(det.get("poly") or [])
        if bbox is None:
            continue
        elements.append(
            LayoutElement(
                order=int(order),
                category=str(det.get("category_type", "unknown")),
                bbox=bbox,
                center=bbox_center(bbox),
            )
        )
    elements.sort(key=lambda item: item.order)
    return elements


def select_samples(dataset: OmniDocBenchDataset, *, strategy: str, limit: Optional[int], seed: int) -> List[OmniDocBenchSample]:
    samples = list(dataset)
    if strategy == "manifest":
        return samples if limit is None else samples[:limit]

    buckets: Dict[str, List[OmniDocBenchSample]] = defaultdict(list)
    for sample in samples:
        attrs = sample.page_attributes or {}
        bucket = str(attrs.get("layout") or attrs.get("data_source") or "unknown")
        buckets[bucket].append(sample)

    if limit is None:
        return samples

    rng = random.Random(seed)
    for bucket_samples in buckets.values():
        rng.shuffle(bucket_samples)

    selected: List[OmniDocBenchSample] = []
    bucket_names = sorted(buckets)
    while len(selected) < limit and bucket_names:
        next_bucket_names: List[str] = []
        for bucket in bucket_names:
            if buckets[bucket] and len(selected) < limit:
                selected.append(buckets[bucket].pop())
            if buckets[bucket]:
                next_bucket_names.append(bucket)
        bucket_names = next_bucket_names
    return selected


def image_content_box(orig_w: int, orig_h: int, target: int) -> Tuple[float, float, float, float]:
    scale = min(target / orig_w, target / orig_h)
    resized_w = orig_w * scale
    resized_h = orig_h * scale
    pad_x = (target - resized_w) / 2.0
    pad_y = (target - resized_h) / 2.0
    return pad_x, pad_y, resized_w, resized_h


def global_token_centers(orig_w: int, orig_h: int, grid: int = 16, target: int = 1024) -> torch.Tensor:
    pad_x, pad_y, resized_w, resized_h = image_content_box(orig_w, orig_h, target)
    centers: List[Tuple[float, float]] = []
    cell = target / grid
    for row in range(grid):
        for col in range(grid):
            x = (col + 0.5) * cell
            y = (row + 0.5) * cell
            src_x = min(max((x - pad_x) / max(resized_w, 1e-6), 0.0), 1.0) * orig_w
            src_y = min(max((y - pad_y) / max(resized_h, 1e-6), 0.0), 1.0) * orig_h
            centers.append((src_x, src_y))
    return torch.tensor(centers, dtype=torch.float32)


def local_token_centers(orig_w: int, orig_h: int, tiles_w: int, tiles_h: int, grid: int = 12) -> torch.Tensor:
    centers: List[Tuple[float, float]] = []
    for tile_y in range(tiles_h):
        for tile_x in range(tiles_w):
            for row in range(grid):
                for col in range(grid):
                    x = ((tile_x + (col + 0.5) / grid) / max(tiles_w, 1)) * orig_w
                    y = ((tile_y + (row + 0.5) / grid) / max(tiles_h, 1)) * orig_h
                    centers.append((x, y))
    return torch.tensor(centers, dtype=torch.float32)


def assign_element(x: float, y: float, elements: Sequence[LayoutElement]) -> Dict[str, Any]:
    if not elements:
        return {
            "element_order": None,
            "element_category": None,
            "assignment": "none",
            "element_center_distance": None,
        }

    containing: List[LayoutElement] = []
    for element in elements:
        x0, y0, x1, y1 = element.bbox
        if x0 <= x <= x1 and y0 <= y <= y1:
            containing.append(element)
    if containing:
        containing.sort(key=lambda element: (element.bbox[2] - element.bbox[0]) * (element.bbox[3] - element.bbox[1]))
        element = containing[0]
        cx, cy = element.center
        return {
            "element_order": element.order,
            "element_category": element.category,
            "assignment": "inside",
            "element_center_distance": math.sqrt((x - cx) ** 2 + (y - cy) ** 2),
        }

    def dist2(element: LayoutElement) -> float:
        cx, cy = element.center
        return (x - cx) ** 2 + (y - cy) ** 2

    element = min(elements, key=dist2)
    return {
        "element_order": element.order,
        "element_category": element.category,
        "assignment": "nearest",
        "element_center_distance": math.sqrt(dist2(element)),
    }


def normalized_entropy(weights: torch.Tensor) -> float:
    weights = weights.float()
    weights = weights / weights.sum().clamp_min(1e-12)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
    return float((entropy / math.log(max(int(weights.numel()), 2))).item())


def weighted_query_centers(
    attentions: Sequence[torch.Tensor],
    centers: torch.Tensor,
    *,
    topk: int,
    layer: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = attentions[layer].detach().float().cpu()
    if attn.ndim != 4:
        raise RuntimeError(f"Expected attention tensor [B,H,S,S], got {tuple(attn.shape)}")

    batch, _, seq_len, _ = attn.shape
    n_image = seq_len // 2
    if centers.shape[0] != batch * n_image:
        raise RuntimeError(f"Center count mismatch: centers={centers.shape[0]} batch*n_image={batch * n_image}")

    query_to_image = attn[:, :, n_image:, :n_image].mean(dim=1)
    flat_centers = centers.reshape(batch, n_image, 2)
    all_query_centers: List[torch.Tensor] = []
    all_entropies: List[torch.Tensor] = []
    all_topk_mass: List[torch.Tensor] = []
    all_max_attention: List[torch.Tensor] = []
    k = min(topk, n_image)

    for batch_idx in range(batch):
        weights = query_to_image[batch_idx]
        top_weights, top_indices = torch.topk(weights, k=k, dim=-1)
        topk_mass = top_weights.sum(dim=-1)
        normalized_top_weights = top_weights / topk_mass.unsqueeze(-1).clamp_min(1e-12)
        selected_centers = flat_centers[batch_idx][top_indices]
        query_centers = (selected_centers * normalized_top_weights.unsqueeze(-1)).sum(dim=1)
        all_query_centers.append(query_centers)
        all_entropies.append(torch.tensor([normalized_entropy(row) for row in weights]))
        all_topk_mass.append(topk_mass)
        all_max_attention.append(weights.max(dim=-1).values)

    return (
        torch.cat(all_query_centers, dim=0),
        torch.cat(all_entropies, dim=0),
        torch.cat(all_topk_mass, dim=0),
        torch.cat(all_max_attention, dim=0),
    )


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return None
    return float((x * y).sum().item() / denom.item())


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def inversion_rate(values: Sequence[int]) -> Optional[float]:
    if len(values) < 2:
        return None
    inversions = 0
    total = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] == values[j]:
                continue
            total += 1
            if values[i] > values[j]:
                inversions += 1
    if total == 0:
        return None
    return inversions / total


def _order_summary(traces: Sequence[QueryTrace], element_count: int) -> Dict[str, Any]:
    assigned = [trace for trace in traces if trace.element_order is not None]
    orders = [int(trace.element_order) for trace in assigned if trace.element_order is not None]
    query_indices = [trace.query_index for trace in assigned]
    covered = set(orders)
    duplicate_fraction = 0.0
    if orders:
        duplicate_fraction = 1.0 - len(covered) / len(orders)
    return {
        "assigned_queries": len(assigned),
        "coverage": len(covered) / max(element_count, 1),
        "duplicate_fraction": duplicate_fraction,
        "pearson_query_order": pearson(query_indices, orders),
        "spearman_query_order": spearman(query_indices, orders),
        "inversion_rate": inversion_rate(orders),
    }


def summarize_traces(traces: Sequence[QueryTrace], element_count: int) -> Dict[str, Any]:
    inside = [trace for trace in traces if trace.assignment == "inside"]
    nearest = [trace for trace in traces if trace.assignment == "nearest"]
    distances = [
        trace.element_center_distance
        for trace in traces
        if trace.element_center_distance is not None
    ]
    category_counts = Counter(
        trace.element_category or "unassigned"
        for trace in traces
        if trace.element_order is not None
    )
    return {
        "queries": len(traces),
        "inside_queries": len(inside),
        "nearest_queries": len(nearest),
        "inside_query_fraction": len(inside) / max(len(traces), 1),
        "mean_entropy": sum(trace.entropy for trace in traces) / max(len(traces), 1),
        "mean_topk_mass": sum(trace.topk_mass for trace in traces) / max(len(traces), 1),
        "mean_max_attention": sum(trace.max_attention for trace in traces) / max(len(traces), 1),
        "mean_element_center_distance": mean_optional(distances),
        "category_counts": dict(category_counts),
        "any_assignment": _order_summary(traces, element_count),
        "inside_assignment": _order_summary(inside, element_count),
    }


def reorder_traces(traces: Sequence[QueryTrace], mode: str, *, seed: int) -> List[QueryTrace]:
    reordered = list(traces)
    if mode == "reverse_final_queries":
        reordered = list(reversed(reordered))
    elif mode == "shuffle_final_queries":
        rng = random.Random(seed)
        rng.shuffle(reordered)
    else:
        raise ValueError(f"Unknown trace reorder mode: {mode}")

    return [
        QueryTrace(
            path=trace.path,
            query_index=new_index,
            x=trace.x,
            y=trace.y,
            entropy=trace.entropy,
            topk_mass=trace.topk_mass,
            max_attention=trace.max_attention,
            element_order=trace.element_order,
            element_category=trace.element_category,
            assignment=trace.assignment,
            element_center_distance=trace.element_center_distance,
        )
        for new_index, trace in enumerate(reordered)
    ]


def move_inputs(inputs: Mapping[str, torch.Tensor], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "pixel_values": inputs["pixel_values"].to(device=device, dtype=dtype),
        "images_crop": inputs["images_crop"].to(device=device, dtype=dtype),
        "images_spatial_crop": inputs["images_spatial_crop"].to(device=device),
    }


@contextmanager
def patched_mask_mode(model: DeepseekOCRModel, mode: str) -> Iterator[None]:
    """Temporarily replace the D2E custom mask builder."""

    qwen_model = model.qwen2_model.model.model
    original = qwen_model._create_custom_4d_mask

    def replacement(sequence_length, dtype, device, batch_size, token_type_ids):
        min_dtype = torch.finfo(dtype).min
        masks = []
        for batch_idx in range(batch_size):
            mask = torch.full((sequence_length, sequence_length), min_dtype, dtype=dtype, device=device)
            type_ids = token_type_ids[batch_idx]
            image_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
            query_positions = (type_ids == 1).nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                mask[image_positions[:, None], image_positions] = 0.0
            for i, query_pos in enumerate(query_positions):
                if len(image_positions) > 0 and mode != "blocked_visual":
                    mask[query_pos, image_positions] = 0.0
                if mode == "bidirectional_queries":
                    mask[query_pos, query_positions] = 0.0
                elif mode == "no_query_to_query":
                    mask[query_pos, query_pos] = 0.0
                elif mode == "blocked_visual":
                    mask[query_pos, query_positions[: i + 1]] = 0.0
                else:
                    raise ValueError(f"Unknown mask mode: {mode}")
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)

    qwen_model._create_custom_4d_mask = replacement
    try:
        yield
    finally:
        qwen_model._create_custom_4d_mask = original


def run_d2e_path(
    model: DeepseekOCRModel,
    images: torch.Tensor,
    *,
    output_attentions: bool,
) -> Tuple[torch.Tensor, Optional[Sequence[torch.Tensor]]]:
    with torch.no_grad():
        sam = model.sam_model(images)
        out = model.qwen2_model(
            sam,
            output_attentions=output_attentions,
            output_hidden_states=False,
        )
    if output_attentions:
        query, attentions, _, _ = out
        return query.detach(), attentions
    if isinstance(out, tuple):
        return out[0].detach(), None
    return out.detach(), None


def trace_path(
    *,
    path: str,
    attentions: Sequence[torch.Tensor],
    centers: torch.Tensor,
    elements: Sequence[LayoutElement],
    topk: int,
    layer: int,
) -> List[QueryTrace]:
    query_centers, entropies, topk_masses, max_attentions = weighted_query_centers(
        attentions,
        centers,
        topk=topk,
        layer=layer,
    )
    traces: List[QueryTrace] = []
    for idx, center in enumerate(query_centers):
        x, y = float(center[0].item()), float(center[1].item())
        assignment = assign_element(x, y, elements)
        traces.append(
            QueryTrace(
                path=path,
                query_index=idx,
                x=x,
                y=y,
                entropy=float(entropies[idx].item()),
                topk_mass=float(topk_masses[idx].item()),
                max_attention=float(max_attentions[idx].item()),
                element_order=assignment["element_order"],
                element_category=assignment["element_category"],
                assignment=assignment["assignment"],
                element_center_distance=assignment["element_center_distance"],
            )
        )
    return traces


def cosine_shift(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.reshape(-1, reference.shape[-1]).float().cpu()
    cand = candidate.reshape(-1, candidate.shape[-1]).float().cpu()
    return float((1.0 - F.cosine_similarity(ref, cand, dim=-1).mean()).item())


def l2_shift(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.reshape(-1, reference.shape[-1]).float().cpu()
    cand = candidate.reshape(-1, candidate.shape[-1]).float().cpu()
    denom = ref.norm(dim=-1).mean().clamp_min(1e-12)
    return float(((ref - cand).norm(dim=-1).mean() / denom).item())


def nearest_neighbor_cosine_shift(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Order-insensitive representation shift using nearest cosine matches."""
    ref = F.normalize(reference.reshape(-1, reference.shape[-1]).float().cpu(), dim=-1)
    cand = F.normalize(candidate.reshape(-1, candidate.shape[-1]).float().cpu(), dim=-1)
    similarity = ref @ cand.transpose(0, 1)
    ref_to_cand = 1.0 - similarity.max(dim=1).values.mean()
    cand_to_ref = 1.0 - similarity.max(dim=0).values.mean()
    return float(((ref_to_cand + cand_to_ref) / 2.0).item())


def center_path_length(traces: Sequence[QueryTrace]) -> Optional[float]:
    if len(traces) < 2:
        return None
    total = 0.0
    for prev, curr in zip(traces, traces[1:]):
        total += math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)
    return total / (len(traces) - 1)


def apply_order_ablation(query: torch.Tensor, mode: str, *, seed: int) -> torch.Tensor:
    if mode == "reverse_final_queries":
        return torch.flip(query, dims=[1])
    if mode == "shuffle_final_queries":
        generator = torch.Generator(device=query.device).manual_seed(seed)
        permutation = torch.randperm(query.shape[1], generator=generator, device=query.device)
        return query[:, permutation, :]
    raise ValueError(f"Unknown order ablation: {mode}")


def run_sample(
    *,
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    sample: OmniDocBenchSample,
    device: str,
    dtype: torch.dtype,
    topk: int,
    ablation_layer: int,
    run_ablations: bool,
    seed: int,
) -> Dict[str, Any]:
    elements = load_layout_elements(sample)
    with Image.open(sample.image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        orig_w, orig_h = image.size
        inputs = processor.process_image(image)

    device_inputs = move_inputs(inputs, device=device, dtype=dtype)
    grid_w, grid_h = [int(v) for v in inputs["images_spatial_crop"][0, 0].tolist()]
    layer = ablation_layer

    path_records: Dict[str, Dict[str, Any]] = {}
    baseline_queries: Dict[str, torch.Tensor] = {}
    baseline_traces: Dict[str, List[QueryTrace]] = {}
    path_inputs = {
        "global": (device_inputs["pixel_values"][0], global_token_centers(orig_w, orig_h)),
        "local": (device_inputs["images_crop"][0, 0], local_token_centers(orig_w, orig_h, grid_w, grid_h)),
    }
    for path_name, (images, centers) in path_inputs.items():
        if path_name == "local" and torch.sum(images).item() == 0:
            continue
        query, attentions = run_d2e_path(model, images, output_attentions=True)
        if attentions is None:
            raise RuntimeError("Attention output was not returned")
        baseline_queries[path_name] = query
        traces = trace_path(
            path=path_name,
            attentions=attentions,
            centers=centers,
            elements=elements,
            topk=topk,
            layer=layer,
        )
        baseline_traces[path_name] = traces
        path_records[path_name] = {
            "trace_summary": summarize_traces(traces, len(elements)),
            "center_path_length": center_path_length(traces),
            "traces": [trace.__dict__ for trace in traces],
        }

    ablations: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    if run_ablations:
        for path_name, (images, centers) in path_inputs.items():
            if path_name not in baseline_queries:
                continue
            baseline = baseline_queries[path_name]
            for mode in ("reverse_final_queries", "shuffle_final_queries"):
                ablated = apply_order_ablation(baseline, mode, seed=seed)
                reordered_traces = reorder_traces(baseline_traces[path_name], mode, seed=seed)
                ablations[path_name][mode] = {
                    "intervention_type": "final_query_order",
                    "index_aligned_cosine_shift": cosine_shift(baseline, ablated),
                    "index_aligned_relative_l2_shift": l2_shift(baseline, ablated),
                    "unordered_nn_cosine_shift": nearest_neighbor_cosine_shift(baseline, ablated),
                    "baseline_center_path_length": center_path_length(baseline_traces[path_name]),
                    "ablated_center_path_length": center_path_length(reordered_traces),
                    "trace_summary": summarize_traces(reordered_traces, len(elements)),
                }
            for mode in ("bidirectional_queries", "no_query_to_query", "blocked_visual"):
                with patched_mask_mode(model, mode):
                    ablated, ablated_attentions = run_d2e_path(model, images, output_attentions=True)
                if ablated_attentions is None:
                    raise RuntimeError("Attention output was not returned for ablation")
                ablated_traces = trace_path(
                    path=path_name,
                    attentions=ablated_attentions,
                    centers=centers,
                    elements=elements,
                    topk=topk,
                    layer=layer,
                )
                ablations[path_name][mode] = {
                    "intervention_type": "attention_mask",
                    "index_aligned_cosine_shift": cosine_shift(baseline, ablated),
                    "index_aligned_relative_l2_shift": l2_shift(baseline, ablated),
                    "unordered_nn_cosine_shift": nearest_neighbor_cosine_shift(baseline, ablated),
                    "baseline_center_path_length": center_path_length(baseline_traces[path_name]),
                    "ablated_center_path_length": center_path_length(ablated_traces),
                    "trace_summary": summarize_traces(ablated_traces, len(elements)),
                }

    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "image_size": [orig_w, orig_h],
        "page_attributes": sample.page_attributes,
        "layout_elements": len(elements),
        "paths": path_records,
        "ablations": ablations,
    }


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def aggregate_results(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path_summary: Dict[str, Dict[str, Any]] = {}
    trace_by_layout: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for path_name in ("global", "local"):
        rows = [
            record["paths"][path_name]["trace_summary"]
            for record in records
            if path_name in record["paths"]
        ]
        if not rows:
            continue
        path_summary[path_name] = {
            "pages": len(rows),
            "mean_queries": sum(row["queries"] for row in rows) / len(rows),
            "mean_inside_query_fraction": mean_optional(row["inside_query_fraction"] for row in rows),
            "mean_inside_coverage": mean_optional(row["inside_assignment"]["coverage"] for row in rows),
            "mean_any_coverage": mean_optional(row["any_assignment"]["coverage"] for row in rows),
            "mean_inside_duplicate_fraction": mean_optional(row["inside_assignment"]["duplicate_fraction"] for row in rows),
            "mean_any_duplicate_fraction": mean_optional(row["any_assignment"]["duplicate_fraction"] for row in rows),
            "mean_entropy": mean_optional(row["mean_entropy"] for row in rows),
            "mean_topk_mass": mean_optional(row["mean_topk_mass"] for row in rows),
            "mean_max_attention": mean_optional(row["mean_max_attention"] for row in rows),
            "mean_inside_spearman_query_order": mean_optional(row["inside_assignment"]["spearman_query_order"] for row in rows),
            "mean_any_spearman_query_order": mean_optional(row["any_assignment"]["spearman_query_order"] for row in rows),
            "mean_inside_inversion_rate": mean_optional(row["inside_assignment"]["inversion_rate"] for row in rows),
            "mean_any_inversion_rate": mean_optional(row["any_assignment"]["inversion_rate"] for row in rows),
        }
        layouts_for_path = sorted(
            {
                str(record.get("page_attributes", {}).get("layout", "unknown"))
                for record in records
                if path_name in record["paths"]
            }
        )
        for layout in layouts_for_path:
            layout_rows = [
                record["paths"][path_name]["trace_summary"]
                for record in records
                if path_name in record["paths"]
                and str(record.get("page_attributes", {}).get("layout", "unknown")) == layout
            ]
            trace_by_layout[path_name][layout] = {
                "pages": len(layout_rows),
                "mean_inside_query_fraction": mean_optional(row["inside_query_fraction"] for row in layout_rows),
                "mean_inside_coverage": mean_optional(row["inside_assignment"]["coverage"] for row in layout_rows),
                "mean_any_coverage": mean_optional(row["any_assignment"]["coverage"] for row in layout_rows),
                "mean_inside_spearman_query_order": mean_optional(row["inside_assignment"]["spearman_query_order"] for row in layout_rows),
                "mean_any_spearman_query_order": mean_optional(row["any_assignment"]["spearman_query_order"] for row in layout_rows),
                "mean_inside_inversion_rate": mean_optional(row["inside_assignment"]["inversion_rate"] for row in layout_rows),
                "mean_any_inversion_rate": mean_optional(row["any_assignment"]["inversion_rate"] for row in layout_rows),
            }

    ablation_summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for path_name in ("global", "local"):
        modes = sorted(
            {
                mode
                for record in records
                for mode in record.get("ablations", {}).get(path_name, {})
            }
        )
        for mode in modes:
            rows = [
                record["ablations"][path_name][mode]
                for record in records
                if mode in record.get("ablations", {}).get(path_name, {})
            ]
            ablation_summary[path_name][mode] = {
                "pages": len(rows),
                "intervention_type": rows[0].get("intervention_type"),
                "mean_index_aligned_cosine_shift": mean_optional(row["index_aligned_cosine_shift"] for row in rows),
                "mean_index_aligned_relative_l2_shift": mean_optional(row["index_aligned_relative_l2_shift"] for row in rows),
                "mean_unordered_nn_cosine_shift": mean_optional(row["unordered_nn_cosine_shift"] for row in rows),
                "mean_baseline_center_path_length": mean_optional(row["baseline_center_path_length"] for row in rows),
                "mean_ablated_center_path_length": mean_optional(row["ablated_center_path_length"] for row in rows),
                "mean_inside_coverage": mean_optional(row["trace_summary"]["inside_assignment"]["coverage"] for row in rows),
                "mean_any_coverage": mean_optional(row["trace_summary"]["any_assignment"]["coverage"] for row in rows),
                "mean_inside_inversion_rate": mean_optional(row["trace_summary"]["inside_assignment"]["inversion_rate"] for row in rows),
                "mean_any_inversion_rate": mean_optional(row["trace_summary"]["any_assignment"]["inversion_rate"] for row in rows),
            }

    layouts = Counter(str(record.get("page_attributes", {}).get("layout", "unknown")) for record in records)
    languages = Counter(str(record.get("page_attributes", {}).get("language", "unknown")) for record in records)
    return {
        "pages": len(records),
        "layouts": dict(layouts),
        "languages": dict(languages),
        "trace_summary": path_summary,
        "trace_by_layout": trace_by_layout,
        "ablation_summary": ablation_summary,
    }


def write_outputs(output_dir: Path, config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_results(records)
    payload = {"config": config, "summary": summary, "records": records}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    lines = [
        "# Query Trace And Mask Ablation",
        "",
        "## Dataset",
        "",
        f"- Pages: `{summary['pages']}`",
        f"- Layouts: `{summary['layouts']}`",
        f"- Languages: `{summary['languages']}`",
        f"- Dataset root: `{config['dataset_root']}`",
        "",
        "## Query-To-Layout Trace",
        "",
        "| path | pages | queries/page | inside query frac | inside coverage | any coverage | inside dup frac | any dup frac | entropy | top-k mass | max attn | inside spearman | any spearman | inside inversion | any inversion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path_name, row in summary["trace_summary"].items():
        lines.append(
            f"| {path_name} | {row['pages']} | {row['mean_queries']:.1f} | "
            f"{fmt(row['mean_inside_query_fraction'])} | {fmt(row['mean_inside_coverage'])} | "
            f"{fmt(row['mean_any_coverage'])} | {fmt(row['mean_inside_duplicate_fraction'])} | "
            f"{fmt(row['mean_any_duplicate_fraction'])} | {fmt(row['mean_entropy'])} | "
            f"{fmt(row['mean_topk_mass'])} | {fmt(row['mean_max_attention'])} | "
            f"{fmt(row['mean_inside_spearman_query_order'])} | {fmt(row['mean_any_spearman_query_order'])} | "
            f"{fmt(row['mean_inside_inversion_rate'])} | {fmt(row['mean_any_inversion_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Trace By Layout",
            "",
            "| path | layout | pages | inside query frac | inside coverage | any coverage | inside spearman | any spearman | inside inversion | any inversion |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for path_name, layouts in summary["trace_by_layout"].items():
        for layout, row in layouts.items():
            lines.append(
                f"| {path_name} | {layout} | {row['pages']} | "
                f"{fmt(row['mean_inside_query_fraction'])} | {fmt(row['mean_inside_coverage'])} | "
                f"{fmt(row['mean_any_coverage'])} | {fmt(row['mean_inside_spearman_query_order'])} | "
                f"{fmt(row['mean_any_spearman_query_order'])} | {fmt(row['mean_inside_inversion_rate'])} | "
                f"{fmt(row['mean_any_inversion_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| path | ablation | type | pages | index cosine shift | unordered cosine shift | rel l2 shift | path length base | path length ablated | inside coverage | inside inversion | any inversion |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for path_name, modes in summary["ablation_summary"].items():
        for mode, row in modes.items():
            lines.append(
                f"| {path_name} | {mode} | {row['intervention_type']} | {row['pages']} | "
                f"{fmt(row['mean_index_aligned_cosine_shift'], precision=6)} | "
                f"{fmt(row['mean_unordered_nn_cosine_shift'], precision=6)} | "
                f"{fmt(row['mean_index_aligned_relative_l2_shift'], precision=6)} | "
                f"{fmt(row['mean_baseline_center_path_length'], precision=2)} | "
                f"{fmt(row['mean_ablated_center_path_length'], precision=2)} | "
                f"{fmt(row['mean_inside_coverage'])} | "
                f"{fmt(row['mean_inside_inversion_rate'])} | {fmt(row['mean_any_inversion_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `inside coverage` only counts query centers that land inside an annotated layout box.",
            "- `any coverage` also includes nearest-element fallback assignments and should be treated as a weak diagnostic.",
            "- `duplicate frac` is high when many queries map to already-covered elements.",
            "- `inversion rate` near 0 means query order follows annotation reading order; near 0.5 is close to random pair order.",
            "- `index cosine shift` is position-aligned; `unordered cosine shift` ignores query order via nearest-neighbor matching.",
            "- Final-query reverse/shuffle rows are proxy order tests unless connected to decoder OCR outputs.",
            "- Mask ablation shifts are measured on D2E query representations before the language decoder.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Optional[float], *, precision: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}"


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    dataset_root = args.dataset_root.resolve()
    dataset = OmniDocBenchDataset.from_dataset_root(
        dataset_root,
        limit=None,
        offset=args.offset,
    )
    samples = select_samples(dataset, strategy=args.sample_strategy, limit=args.limit, seed=args.seed)
    if args.ablation_pages < 0:
        ablation_ids = {sample.sample_id for sample in samples}
    else:
        ablation_ids = {sample.sample_id for sample in samples[: args.ablation_pages]}

    print(f"Loading model from {args.model_path} on {args.device}")
    model = DeepseekOCRModel.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=dtype,
        attn_implementation="eager",
        output_attentions=True,
        output_hidden_states=False,
    )
    model.eval()
    processor = ImageProcessor(crop_mode=True)

    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}")
        record = run_sample(
            model=model,
            processor=processor,
            sample=sample,
            device=args.device,
            dtype=dtype,
            topk=args.topk,
            ablation_layer=args.ablation_layer,
            run_ablations=sample.sample_id in ablation_ids,
            seed=args.seed + index,
        )
        records.append(record)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    config = {
        "dataset_root": str(dataset_root),
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "limit": args.limit,
        "offset": args.offset,
        "sample_strategy": args.sample_strategy,
        "topk": args.topk,
        "ablation_pages": args.ablation_pages,
        "ablation_layer": args.ablation_layer,
    }
    write_outputs(args.output_dir, config, records)
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
