#!/usr/bin/env python3
"""Run feature-level SAE ablations inside a D2E layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from src.analysis.interventions import InterventionManager
from src.analysis.sparse_autoencoder import SparseAutoencoder
from src.models.deepseek_ocr import DeepseekOCRModel
from src.preprocessing.image_transforms import ImageProcessor

from scripts.research_causal_tokens import build_stimuli, compute_attention_centers, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate learned SAE features in a D2E layer")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--sae_checkpoint", required=True)
    parser.add_argument("--sae_summary", required=True)
    parser.add_argument("--output_dir", default="output/sae_feature_ablation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--feature_ids", default=None, help="Comma-separated explicit feature ids")
    parser.add_argument("--features_per_view", type=int, default=3)
    parser.add_argument("--group_strategy", default="none", choices=["none", "localized_bands"])
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--mode", default="subtract_decoder", choices=["subtract_decoder", "reconstruct"])
    parser.add_argument("--top_k_queries", type=int, default=8)
    return parser.parse_args()


def load_sae_checkpoint(path: str, device: str) -> tuple[SparseAutoencoder, Dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload["state_dict"]
    encoder_weight = state_dict["encoder.weight"]
    n_features, input_dim = encoder_weight.shape
    config = payload.get("config", {})
    sae = SparseAutoencoder(
        input_dim=input_dim,
        n_features=n_features,
        activation_mode=config.get("activation_mode", "relu"),
        top_k=config.get("top_k"),
    )
    sae.load_state_dict(state_dict)
    sae = sae.to(device=device, dtype=torch.float32)
    sae.eval()
    return sae, config


def select_features(summary_path: str, feature_ids: str | None, features_per_view: int) -> List[Dict[str, object]]:
    data = json.loads(Path(summary_path).read_text())
    all_features = {row["feature_index"]: row for row in data.get("all_features", [])}

    if feature_ids:
        selected_ids = [int(item) for item in feature_ids.split(",") if item.strip()]
    else:
        selected_ids: List[int] = []
        for view_name in ("top_query_localized_features", "top_spatially_localized_features"):
            for row in data["feature_views"][view_name]:
                if row["feature_index"] not in selected_ids:
                    selected_ids.append(row["feature_index"])
                if len(selected_ids) >= features_per_view * 2:
                    break
            if len(selected_ids) >= features_per_view * 2:
                break

    rows = []
    for feature_idx in selected_ids:
        if feature_idx in all_features:
            rows.append(all_features[feature_idx])
        else:
            rows.append({"feature_index": feature_idx})
    return rows


def _localized_feature_score(row: Dict[str, object]) -> float:
    q_std = float(row.get("weighted_query_index_std") or 1e9)
    x_std = float(row.get("weighted_attention_x_std") or 1e9)
    y_std = float(row.get("weighted_attention_y_std") or 1e9)
    mean_activation = float(row.get("mean_activation") or 0.0)
    activation_frequency = float(row.get("activation_frequency") or 0.0)
    return -(q_std + x_std * 100.0 + y_std * 100.0) + mean_activation * 5.0 + activation_frequency * 2.0


def select_feature_groups(
    summary_path: str,
    *,
    group_strategy: str,
    group_size: int,
) -> List[Dict[str, object]]:
    if group_strategy == "none":
        return []

    data = json.loads(Path(summary_path).read_text())
    all_rows = data.get("all_features", [])
    if not all_rows:
        return []

    def choose(label: str, predicate) -> Optional[Dict[str, object]]:
        candidates = [row for row in all_rows if predicate(row)]
        candidates.sort(key=_localized_feature_score, reverse=True)
        selected = candidates[:group_size]
        if not selected:
            return None
        return {
            "group_label": label,
            "feature_indices": [int(row["feature_index"]) for row in selected],
            "tag": label,
            "group_members": selected,
        }

    groups: List[Dict[str, object]] = []
    if group_strategy == "localized_bands":
        definitions = [
            (
                "early_top_band",
                lambda row: (
                    row.get("weighted_query_index_mean") is not None
                    and row.get("weighted_attention_y_mean") is not None
                    and float(row["weighted_query_index_mean"]) < 40.0
                    and float(row["weighted_attention_y_mean"]) < 0.25
                    and float(row.get("activation_frequency") or 0.0) >= 0.05
                ),
            ),
            (
                "mid_lower_band",
                lambda row: (
                    row.get("weighted_query_index_mean") is not None
                    and row.get("weighted_attention_y_mean") is not None
                    and 100.0 <= float(row["weighted_query_index_mean"]) <= 190.0
                    and float(row["weighted_attention_y_mean"]) > 0.5
                    and float(row.get("activation_frequency") or 0.0) >= 0.05
                ),
            ),
            (
                "late_top_band",
                lambda row: (
                    row.get("weighted_query_index_mean") is not None
                    and row.get("weighted_attention_y_mean") is not None
                    and float(row["weighted_query_index_mean"]) > 210.0
                    and float(row["weighted_attention_y_mean"]) < 0.2
                    and float(row.get("activation_frequency") or 0.0) >= 0.05
                ),
            ),
        ]
        for label, predicate in definitions:
            group = choose(label, predicate)
            if group is not None:
                groups.append(group)
        return groups

    raise ValueError(f"Unknown group strategy: {group_strategy!r}")


def run_global_d2e(model, processor, image, device):
    inputs = processor.process_image(image)
    pixel_values = inputs["pixel_values"][0].to(device=device, dtype=next(model.sam_model.parameters()).dtype)
    with torch.no_grad():
        sam_features = model.sam_model(pixel_values)
        query_out, attentions, _, _ = model.qwen2_model(
            sam_features,
            output_attentions=True,
            output_hidden_states=False,
        )
    return {
        "sam_features": sam_features,
        "query_outputs": query_out[0].detach(),
        "attentions": [layer[0].detach() for layer in attentions],
        "spatial_size": sam_features.shape[-1],
    }


def ablate_feature_group(
    model,
    sam_features,
    feature_indices: Sequence[int],
    layer: int,
    sae: SparseAutoencoder,
    mode: str,
):
    with InterventionManager(model) as mgr:
        mgr.ablate_sae_features_in_query_states(
            layer=layer,
            sae=sae,
            feature_indices=list(feature_indices),
            mode=mode,
        )
        with torch.no_grad():
            query_out, attentions, _, _ = model.qwen2_model(
                sam_features,
                output_attentions=True,
                output_hidden_states=False,
            )
    return {
        "query_outputs": query_out[0].detach(),
        "attentions": [layer_attn[0].detach() for layer_attn in attentions],
    }


def summarize_effect(
    baseline_queries: torch.Tensor,
    baseline_centers: torch.Tensor,
    ablated_queries: torch.Tensor,
    ablated_centers: torch.Tensor,
    top_k: int,
) -> Dict[str, object]:
    cosine_drop = 1.0 - F.cosine_similarity(
        baseline_queries.float(),
        ablated_queries.float(),
        dim=-1,
    )
    center_shift = (baseline_centers.float() - ablated_centers.float()).norm(dim=-1)
    query_index = torch.arange(cosine_drop.shape[0], dtype=torch.float32)

    top_values, top_indices = torch.topk(cosine_drop, k=min(top_k, cosine_drop.shape[0]))
    top_queries = []
    for idx, score in zip(top_indices.tolist(), top_values.tolist()):
        top_queries.append(
            {
                "query_index": idx,
                "cosine_drop": float(score),
                "attention_shift": float(center_shift[idx].item()),
                "baseline_center_x": float(baseline_centers[idx, 0].item()),
                "baseline_center_y": float(baseline_centers[idx, 1].item()),
                "ablated_center_x": float(ablated_centers[idx, 0].item()),
                "ablated_center_y": float(ablated_centers[idx, 1].item()),
            }
        )

    weight = cosine_drop.clamp_min(0)
    weighted_query_mean = None
    if float(weight.sum().item()) > 0.0:
        weighted_query_mean = float((weight * query_index).sum().item() / weight.sum().item())

    return {
        "mean_query_cosine_drop": float(cosine_drop.mean().item()),
        "max_query_cosine_drop": float(cosine_drop.max().item()),
        "mean_attention_shift": float(center_shift.mean().item()),
        "max_attention_shift": float(center_shift.max().item()),
        "weighted_query_effect_mean": weighted_query_mean,
        "top_affected_queries": top_queries,
    }


def write_summary_markdown(output_path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# SAE Feature Ablation Summary",
        "",
        f"- Layer: {payload['layer']}",
        f"- Mode: {payload['mode']}",
        "",
        "## Aggregate Effects",
        "",
        "| feature | tag | freq | q mean | y mean | mean query drop | mean attn shift | weighted query effect |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_results"]:
        lines.append(
            f"| {row['feature_index']} | {row.get('tag', 'NA')} | "
            f"{row.get('activation_frequency', 'NA')} | {row.get('weighted_query_index_mean', 'NA')} | "
            f"{row.get('weighted_attention_y_mean', 'NA')} | {row['mean_query_cosine_drop']:.4f} | "
            f"{row['mean_attention_shift']:.4f} | {row['weighted_query_effect_mean'] if row['weighted_query_effect_mean'] is not None else 'NA'} |"
        )
    if payload.get("aggregate_group_results"):
        lines.extend(
            [
                "",
                "## Group Effects",
                "",
                "| group | features | mean query drop | mean attn shift | weighted query effect |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in payload["aggregate_group_results"]:
            lines.append(
                f"| {row['group_label']} | {','.join(str(feature) for feature in row['feature_indices'])} | "
                f"{row['mean_query_cosine_drop']:.4f} | {row['mean_attention_shift']:.4f} | "
                f"{row['weighted_query_effect_mean'] if row['weighted_query_effect_mean'] is not None else 'NA'} |"
            )
    output_path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae, sae_config = load_sae_checkpoint(args.sae_checkpoint, args.device)
    layer = int(sae_config.get("layer", 12))
    model = DeepseekOCRModel.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation="eager",
        output_attentions=True,
        output_hidden_states=False,
    )
    model.eval()
    processor = ImageProcessor(crop_mode=False)
    stimuli = build_stimuli()
    selected_features = select_features(args.sae_summary, args.feature_ids, args.features_per_view)
    selected_groups = select_feature_groups(
        args.sae_summary,
        group_strategy=args.group_strategy,
        group_size=args.group_size,
    )

    baseline_by_stimulus = {}
    for stimulus_name, image in stimuli.items():
        result = run_global_d2e(model, processor, image, args.device)
        baseline_by_stimulus[stimulus_name] = result

    per_feature_results = []
    aggregate_results = []
    for feature_row in selected_features:
        feature_idx = int(feature_row["feature_index"])
        feature_records = []
        for stimulus_name, image in stimuli.items():
            baseline = baseline_by_stimulus[stimulus_name]
            ablated = ablate_feature_group(model, baseline["sam_features"], [feature_idx], layer, sae, args.mode)
            baseline_centers = compute_attention_centers(baseline["attentions"][-1], baseline["spatial_size"]).cpu()
            ablated_centers = compute_attention_centers(ablated["attentions"][-1], baseline["spatial_size"]).cpu()
            effect = summarize_effect(
                baseline["query_outputs"].cpu(),
                baseline_centers,
                ablated["query_outputs"].cpu(),
                ablated_centers,
                top_k=args.top_k_queries,
            )
            record = {
                "feature_index": feature_idx,
                "stimulus": stimulus_name,
                **effect,
            }
            feature_records.append(record)
            per_feature_results.append(record)

        aggregate_results.append(
            {
                **feature_row,
                "mean_query_cosine_drop": float(sum(item["mean_query_cosine_drop"] for item in feature_records) / len(feature_records)),
                "mean_attention_shift": float(sum(item["mean_attention_shift"] for item in feature_records) / len(feature_records)),
                "weighted_query_effect_mean": float(
                    sum(item["weighted_query_effect_mean"] or 0.0 for item in feature_records) / len(feature_records)
                ),
            }
        )

    aggregate_results.sort(key=lambda row: row["mean_query_cosine_drop"], reverse=True)
    per_group_results = []
    aggregate_group_results = []
    for group_row in selected_groups:
        feature_indices = [int(item) for item in group_row["feature_indices"]]
        group_records = []
        for stimulus_name, image in stimuli.items():
            baseline = baseline_by_stimulus[stimulus_name]
            ablated = ablate_feature_group(model, baseline["sam_features"], feature_indices, layer, sae, args.mode)
            baseline_centers = compute_attention_centers(baseline["attentions"][-1], baseline["spatial_size"]).cpu()
            ablated_centers = compute_attention_centers(ablated["attentions"][-1], baseline["spatial_size"]).cpu()
            effect = summarize_effect(
                baseline["query_outputs"].cpu(),
                baseline_centers,
                ablated["query_outputs"].cpu(),
                ablated_centers,
                top_k=args.top_k_queries,
            )
            record = {
                "group_label": group_row["group_label"],
                "feature_indices": feature_indices,
                "stimulus": stimulus_name,
                **effect,
            }
            group_records.append(record)
            per_group_results.append(record)

        aggregate_group_results.append(
            {
                **group_row,
                "mean_query_cosine_drop": float(sum(item["mean_query_cosine_drop"] for item in group_records) / len(group_records)),
                "mean_attention_shift": float(sum(item["mean_attention_shift"] for item in group_records) / len(group_records)),
                "weighted_query_effect_mean": float(
                    sum(item["weighted_query_effect_mean"] or 0.0 for item in group_records) / len(group_records)
                ),
            }
        )

    aggregate_group_results.sort(key=lambda row: row["mean_query_cosine_drop"], reverse=True)
    payload = {
        "layer": layer,
        "mode": args.mode,
        "sae_checkpoint": args.sae_checkpoint,
        "aggregate_results": aggregate_results,
        "aggregate_group_results": aggregate_group_results,
        "per_feature_results": per_feature_results,
        "per_group_results": per_group_results,
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="ascii")
    write_summary_markdown(summary_md, payload)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
