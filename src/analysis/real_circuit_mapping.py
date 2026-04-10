"""Helpers for circuit mapping on real OmniDocBench pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RegionTarget:
    """Resolved target region on a page."""

    label: str
    bbox: BBox
    area: float
    det_index: int
    raw_det: Mapping[str, Any]


def normalize_label(label: Any) -> str:
    """Normalize a region label for filtering."""
    if label is None:
        return ""
    text = str(label).strip().lower()
    for old, new in (("-", "_"), (" ", "_"), ("/", "_")):
        text = text.replace(old, new)
    return text


def get_region_label(det: Mapping[str, Any]) -> str:
    """Extract the best-effort label from a layout annotation."""
    for key in ("category_type", "category", "type", "label", "cls", "class", "name"):
        if key in det and det[key] is not None:
            return normalize_label(det[key])
    return ""


def extract_box_coordinates(det: Mapping[str, Any]) -> Optional[List[float]]:
    """Extract a flat list of coordinates from a layout annotation."""
    for key in ("bbox", "box", "polygon", "poly", "quad", "points"):
        value = det.get(key)
        if value is None:
            continue

        flat = _flatten_numbers(value)
        if len(flat) >= 4:
            return flat
    return None


def resolve_region_bbox(
    det: Mapping[str, Any],
    *,
    image_size: Tuple[int, int],
    page_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Optional[BBox]:
    """Resolve a region bbox into image pixel coordinates."""
    coords = extract_box_coordinates(det)
    if coords is None:
        return None

    x_values = coords[0::2]
    y_values = coords[1::2]
    if not x_values or not y_values:
        return None

    x0 = min(x_values)
    x1 = max(x_values)
    y0 = min(y_values)
    y1 = max(y_values)
    if x1 <= x0 or y1 <= y0:
        return None

    width, height = image_size
    page_width = page_size[0] if page_size is not None else None
    page_height = page_size[1] if page_size is not None else None
    max_coord = max(coords)

    if max_coord <= 1.5:
        scale_x = width
        scale_y = height
    elif (
        page_width
        and page_height
        and max_coord <= max(float(page_width), float(page_height)) * 1.5
    ):
        scale_x = width / max(float(page_width), 1.0)
        scale_y = height / max(float(page_height), 1.0)
    elif max_coord <= 1000.0:
        scale_x = width / 1000.0
        scale_y = height / 1000.0
    else:
        scale_x = 1.0
        scale_y = 1.0

    bbox = (
        float(max(0.0, min(width, x0 * scale_x))),
        float(max(0.0, min(height, y0 * scale_y))),
        float(max(0.0, min(width, x1 * scale_x))),
        float(max(0.0, min(height, y1 * scale_y))),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def extract_region_targets(
    raw_entry: Mapping[str, Any],
    *,
    image_size: Tuple[int, int],
    page_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    region_type: Optional[str] = None,
    max_regions: Optional[int] = None,
    min_area_ratio: float = 1e-4,
) -> List[RegionTarget]:
    """Extract filtered target regions from an OmniDocBench entry."""
    detections = raw_entry.get("layout_dets", [])
    desired = normalize_label(region_type) if region_type else ""
    min_area = image_size[0] * image_size[1] * min_area_ratio

    targets: List[RegionTarget] = []
    for det_index, det in enumerate(detections):
        if not isinstance(det, Mapping):
            continue
        label = get_region_label(det)
        if desired and desired not in label:
            continue

        bbox = resolve_region_bbox(
            det,
            image_size=image_size,
            page_size=page_size,
        )
        if bbox is None:
            continue

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_area:
            continue

        targets.append(
            RegionTarget(
                label=label or "unknown",
                bbox=bbox,
                area=float(area),
                det_index=det_index,
                raw_det=det,
            )
        )

    targets.sort(key=lambda item: item.area, reverse=True)
    if max_regions is not None:
        targets = targets[:max_regions]
    return targets


def map_bbox_to_padded_square(
    bbox: BBox,
    *,
    image_size: Tuple[int, int],
    square_size: int,
) -> BBox:
    """Map an image-space bbox into the padded square used for the global view."""
    width, height = image_size
    scale = min(square_size / width, square_size / height)
    scaled_width = width * scale
    scaled_height = height * scale
    offset_x = (square_size - scaled_width) / 2.0
    offset_y = (square_size - scaled_height) / 2.0

    return (
        bbox[0] * scale + offset_x,
        bbox[1] * scale + offset_y,
        bbox[2] * scale + offset_x,
        bbox[3] * scale + offset_y,
    )


def bbox_to_grid_mask(
    bbox: BBox,
    *,
    square_size: int,
    spatial_size: int,
) -> torch.Tensor:
    """Convert a square-space bbox to a fractional overlap mask on the token grid."""
    x0, y0, x1, y1 = bbox
    cell_size = square_size / spatial_size
    mask = torch.zeros(spatial_size, spatial_size, dtype=torch.float32)

    for row in range(spatial_size):
        cell_y0 = row * cell_size
        cell_y1 = (row + 1) * cell_size
        overlap_y = max(0.0, min(y1, cell_y1) - max(y0, cell_y0))
        if overlap_y <= 0.0:
            continue

        for col in range(spatial_size):
            cell_x0 = col * cell_size
            cell_x1 = (col + 1) * cell_size
            overlap_x = max(0.0, min(x1, cell_x1) - max(x0, cell_x0))
            if overlap_x <= 0.0:
                continue
            mask[row, col] = float((overlap_x * overlap_y) / (cell_size * cell_size))

    return mask


def select_target_queries(
    attention: torch.Tensor,
    *,
    box_mask: torch.Tensor,
    top_k: int,
    n_image_tokens: Optional[int] = None,
) -> Tuple[List[int], torch.Tensor]:
    """Rank queries by how much query-to-image attention lands in the target box."""
    if attention.dim() != 3:
        raise ValueError("attention must have shape [n_heads, seq, seq].")

    spatial_tokens = box_mask.numel()
    n_image = n_image_tokens or spatial_tokens
    q2i = attention[:, n_image:, :n_image].float().mean(dim=0)
    weights = box_mask.reshape(-1).to(q2i.device, dtype=q2i.dtype)
    scores = q2i @ weights

    k = min(top_k, scores.shape[0])
    top_scores, top_indices = torch.topk(scores, k=k, dim=0)
    return top_indices.tolist(), top_scores.cpu()


def score_query_alignment(
    clean_queries: torch.Tensor,
    candidate_queries: torch.Tensor,
    query_indices: Sequence[int],
) -> float:
    """Measure mean cosine alignment on a selected set of query slots."""
    if not query_indices:
        return 0.0

    index = torch.as_tensor(list(query_indices), dtype=torch.long, device=clean_queries.device)
    clean = clean_queries.index_select(0, index).float()
    candidate = candidate_queries.index_select(0, index).float()
    cosine = torch.nn.functional.cosine_similarity(clean, candidate, dim=-1)
    return float(cosine.mean().item())


def aggregate_circuit_results(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate patching results by `(layer, query_idx)` across many records."""
    buckets: Dict[Tuple[int, int], Dict[str, float]] = {}
    for record in records:
        for result in record.get("results", []):
            key = (int(result["layer"]), int(result["query_idx"]))
            bucket = buckets.setdefault(
                key,
                {
                    "layer": float(result["layer"]),
                    "query_idx": float(result["query_idx"]),
                    "count": 0.0,
                    "impact_sum": 0.0,
                    "baseline_sum": 0.0,
                    "patched_sum": 0.0,
                },
            )
            bucket["count"] += 1.0
            bucket["impact_sum"] += float(result["impact"])
            bucket["baseline_sum"] += float(result["baseline_score"])
            bucket["patched_sum"] += float(result["patched_score"])

    aggregated: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        count = max(bucket["count"], 1.0)
        aggregated.append(
            {
                "layer": int(bucket["layer"]),
                "query_idx": int(bucket["query_idx"]),
                "count": int(bucket["count"]),
                "mean_impact": bucket["impact_sum"] / count,
                "mean_baseline_score": bucket["baseline_sum"] / count,
                "mean_patched_score": bucket["patched_sum"] / count,
            }
        )

    aggregated.sort(key=lambda item: item["mean_impact"], reverse=True)
    return aggregated


def _flatten_numbers(value: Any) -> List[float]:
    """Flatten nested coordinate containers into a numeric list."""
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        for key_order in (
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("x0", "y0", "x1", "y1"),
        ):
            if all(key in value for key in key_order):
                return [float(value[key]) for key in key_order]
        flat: List[float] = []
        for item in value.values():
            flat.extend(_flatten_numbers(item))
        return flat
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flat = []
        for item in value:
            flat.extend(_flatten_numbers(item))
        return flat
    return []
