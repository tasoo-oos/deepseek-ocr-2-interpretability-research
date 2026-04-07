#!/usr/bin/env python3
"""
Extract and visualize intermediate representations from DeepSeek-OCR-2.

Usage:
    # Synthetic D2E-only data (no model weights needed):
    uv run python scripts/extract_attention.py \
        --synthetic \
        --output_dir output/attention_viz \
        --save_raw_data

    # With a real image:
    uv run python scripts/extract_attention.py \
        --image_path input/example.jpg \
        --output_dir output/attention_viz \
        --layers 0,6,12,18,23 \
        --full_report \
        --feature_viz_types sam,d2e,projector,trajectory \
        --save_raw_data
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DeepSeek-OCR-2 attention and intermediate features"
    )
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR-2",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument("--output_dir", type=str, default="output/attention_viz")
    parser.add_argument("--target_size", type=int, default=1024, choices=[768, 1024])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: 0,6,12,18,23)",
    )
    parser.add_argument(
        "--viz_types", type=str, default="mask,evolution,query_to_image,causal,entropy"
    )
    parser.add_argument(
        "--feature_viz_types",
        type=str,
        default="sam,d2e,projector,trajectory",
        help="Comma-separated non-attention visualizations to save",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--colormap", type=str, default="viridis")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no model weights needed)",
    )
    parser.add_argument(
        "--full_report", action="store_true", help="Generate complete report"
    )
    parser.add_argument("--include_animation", action="store_true")
    parser.add_argument(
        "--numeric_metric",
        type=str,
        choices=[
            "image_self_entropy",
            "query_to_image_entropy",
            "query_to_query_entropy",
            "query_to_image_ratio",
        ],
        help="Compute a scalar attention metric and save it",
    )
    parser.add_argument(
        "--numeric_layer",
        type=int,
        default=None,
        help="Optional layer index for --numeric_metric (default: average all layers)",
    )
    parser.add_argument(
        "--numeric_head",
        type=int,
        default=None,
        help="Optional head index for --numeric_metric (default: average all heads)",
    )
    parser.add_argument(
        "--numeric_output",
        type=str,
        default=None,
        help="Path to JSON file for the numeric metric (default: <output_dir>/numeric_value.json)",
    )
    parser.add_argument(
        "--save_raw_data",
        action="store_true",
        help="Save raw attention tensors and metadata",
    )
    parser.add_argument(
        "--raw_output",
        type=str,
        default=None,
        help="Path to raw attention dump (.pt) (default: <output_dir>/attention_raw.pt)",
    )
    return parser.parse_args()


def _select_attention_slice(attn, n_image, metric, head_idx=None):
    if head_idx is not None:
        attn = attn[:, head_idx : head_idx + 1]

    if metric == "image_self_entropy":
        return attn[:, :, :n_image, :n_image]
    if metric == "query_to_image_entropy":
        return attn[:, :, n_image:, :n_image]
    if metric == "query_to_query_entropy":
        return attn[:, :, n_image:, n_image:]
    if metric == "query_to_image_ratio":
        return attn[:, :, n_image:, :n_image]

    raise ValueError(f"Unsupported numeric metric: {metric}")


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def compute_numeric_metric(
    attention_weights, spatial_size, metric, layer_idx=None, head_idx=None
):
    from src.visualization.utils import compute_attention_entropy

    n_image = spatial_size * spatial_size
    selected_layers = (
        [attention_weights[layer_idx]] if layer_idx is not None else attention_weights
    )

    values = []
    for attn in selected_layers:
        if metric.endswith("_entropy"):
            attn_slice = _select_attention_slice(
                attn, n_image, metric, head_idx=head_idx
            )
            values.append(compute_attention_entropy(attn_slice).mean().item())
        elif metric == "query_to_image_ratio":
            q2i = _select_attention_slice(attn, n_image, metric, head_idx=head_idx)
            if head_idx is not None:
                total = attn[:, head_idx : head_idx + 1, n_image:, :].sum(dim=-1)
            else:
                total = attn[:, :, n_image:, :].sum(dim=-1)
            ratio = (q2i.sum(dim=-1) / (total + 1e-9)).mean().item()
            values.append(ratio)
        else:
            raise ValueError(f"Unsupported numeric metric: {metric}")

    return float(sum(values) / len(values))


def save_numeric_metric(args, attention_weights, spatial_size, output_dir):
    if not args.numeric_metric:
        return

    value = compute_numeric_metric(
        attention_weights,
        spatial_size,
        args.numeric_metric,
        layer_idx=args.numeric_layer,
        head_idx=args.numeric_head,
    )
    numeric_output = (
        Path(args.numeric_output)
        if args.numeric_output
        else output_dir / "numeric_value.json"
    )
    numeric_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": args.numeric_metric,
        "value": value,
        "layer": args.numeric_layer,
        "head": args.numeric_head,
    }
    with numeric_output.open("w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Saved {args.numeric_metric}={value:.6f} to {numeric_output}")


def save_raw_data(args, artifacts, layers, output_dir):
    if not args.save_raw_data:
        return

    raw_output = (
        Path(args.raw_output) if args.raw_output else output_dir / "attention_raw.pt"
    )
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attention_weights": artifacts["attention_weights"],
        "token_type_ids": artifacts["token_type_ids"],
        "spatial_size": artifacts["spatial_size"],
        "layers_requested": layers,
        "sam_features": artifacts.get("sam_features"),
        "d2e_query_outputs": artifacts.get("d2e_query_outputs"),
        "d2e_hidden_states": artifacts.get("d2e_hidden_states"),
        "projector_outputs": artifacts.get("projector_outputs"),
        "vision_intermediates": artifacts.get("vision_intermediates"),
    }
    torch.save(payload, raw_output)
    print(f"Saved raw model data to {raw_output}")


def run_synthetic(args):
    from addict import Dict as ADict
    from src.models.projector import MlpProjector
    from src.models.qwen2_d2e import Qwen2Decoder2Encoder

    print("Running in synthetic mode...")
    d2e = Qwen2Decoder2Encoder(
        decoder_layer=24,
        hidden_dimension=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        output_attentions=True,
        output_hidden_states=True,
    )
    projector = MlpProjector(
        ADict(projector_type="linear", input_dim=896, n_embed=1280)
    )
    if args.device == "cuda":
        d2e = d2e.cuda()
        projector = projector.cuda()
    d2e.eval()
    projector.eval()

    spatial_size = 16 if args.target_size == 1024 else 12
    sam_features = torch.randn(1, 896, spatial_size, spatial_size)
    if args.device == "cuda":
        sam_features = sam_features.cuda()

    with torch.no_grad():
        query_outputs, attentions, hidden_states, token_type_ids = d2e(
            sam_features,
            output_attentions=True,
            output_hidden_states=True,
        )
        projector_outputs = projector(query_outputs)

    image = Image.new(
        "RGB", (args.target_size, args.target_size), color=(128, 128, 128)
    )
    return {
        "attention_weights": _to_cpu(list(attentions)),
        "token_type_ids": _to_cpu(token_type_ids),
        "spatial_size": spatial_size,
        "image": image,
        "sam_features": _to_cpu(sam_features),
        "d2e_query_outputs": _to_cpu(query_outputs),
        "d2e_hidden_states": _to_cpu(
            list(hidden_states) if hidden_states is not None else None
        ),
        "projector_outputs": _to_cpu(projector_outputs),
        "vision_intermediates": None,
    }


def run_with_model(args):
    from src.models.deepseek_ocr import DeepseekOCRModel
    from src.preprocessing.image_transforms import ImageProcessor

    print("Loading model components...")
    model = DeepseekOCRModel.from_pretrained(
        args.model_path,
        use_language_model=False,
        output_attentions=True,
        output_hidden_states=True,
        device=args.device,
    )
    model.eval()
    sam = model.sam_model
    d2e = model.qwen2_model

    processor = ImageProcessor(base_size=args.target_size)
    image = Image.open(args.image_path).convert("RGB")
    inputs = processor.process_image(image)
    model_inputs = {
        "pixel_values": inputs["pixel_values"].to(args.device, dtype=torch.bfloat16),
        "images_crop": inputs["images_crop"].to(args.device, dtype=torch.bfloat16),
        "images_spatial_crop": inputs["images_spatial_crop"].to(args.device),
    }
    pixel_values = model_inputs["pixel_values"][0]

    with torch.no_grad():
        _, vision_intermediates = model.get_multimodal_embeddings(
            model_inputs["pixel_values"],
            model_inputs["images_crop"],
            model_inputs["images_spatial_crop"],
            return_intermediate=True,
        )
        sam_features = sam(pixel_values)
        query_outputs, attentions, hidden_states, token_type_ids = d2e(
            sam_features,
            output_attentions=True,
            output_hidden_states=True,
        )
        projector_outputs = model.projector(query_outputs)

    spatial_size = sam_features.shape[-1]
    return {
        "attention_weights": _to_cpu(list(attentions)),
        "token_type_ids": _to_cpu(token_type_ids),
        "spatial_size": spatial_size,
        "image": image,
        "sam_features": _to_cpu(sam_features),
        "d2e_query_outputs": _to_cpu(query_outputs),
        "d2e_hidden_states": _to_cpu(
            list(hidden_states) if hidden_states is not None else None
        ),
        "projector_outputs": _to_cpu(projector_outputs),
        "vision_intermediates": _to_cpu(vision_intermediates),
    }


def save_feature_visualizations(args, artifacts, layers, output_dir):
    from src.visualization.feature_viz import FeatureVisualizer
    import matplotlib.pyplot as plt

    feature_viz_types = [
        x.strip() for x in args.feature_viz_types.split(",") if x.strip()
    ]
    if not feature_viz_types:
        return

    viz = FeatureVisualizer()

    def save(fig, name):
        fig.savefig(output_dir / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    sam_features = artifacts.get("sam_features")
    hidden_states = artifacts.get("d2e_hidden_states")
    projector_outputs = artifacts.get("projector_outputs")

    if "sam" in feature_viz_types and sam_features is not None:
        save(viz.plot_sam_features(sam_features), "sam_features.png")

    if "d2e" in feature_viz_types and hidden_states:
        try:
            for layer in layers:
                hs_idx = min(layer + 1, len(hidden_states) - 1)
                save(
                    viz.plot_d2e_hidden_states(hidden_states[hs_idx], layer=layer),
                    f"d2e_hidden_layer_{layer:02d}.png",
                )
        except ImportError:
            print("scikit-learn not installed; skipping D2E hidden-state PCA plots")

    if "projector" in feature_viz_types and projector_outputs is not None:
        save(viz.plot_projector_output(projector_outputs), "projector_output.png")

    if "trajectory" in feature_viz_types:
        trajectory_inputs = {}
        if sam_features is not None:
            trajectory_inputs["sam_layer_0"] = sam_features
        if hidden_states:
            for layer in layers:
                hs_idx = min(layer + 1, len(hidden_states) - 1)
                trajectory_inputs[f"d2e_layer_{layer}"] = hidden_states[hs_idx]
        if projector_outputs is not None:
            trajectory_inputs["projector"] = projector_outputs
        if trajectory_inputs:
            save(
                viz.plot_activation_trajectory(trajectory_inputs),
                "activation_trajectory.png",
            )


def main():
    args = parse_args()
    layers = (
        [int(x) for x in args.layers.split(",")] if args.layers else [0, 6, 12, 18, 23]
    )
    viz_types = [x.strip() for x in args.viz_types.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        artifacts = run_synthetic(args)
    else:
        if not args.image_path:
            raise ValueError("--image_path required (or use --synthetic)")
        artifacts = run_with_model(args)

    attention_weights = artifacts["attention_weights"]
    token_type_ids = artifacts["token_type_ids"]
    spatial_size = artifacts["spatial_size"]
    image = artifacts["image"]

    print(
        f"Extracted {len(attention_weights)} attention layers, "
        f"{attention_weights[0].shape[1]} heads, "
        f"spatial size {spatial_size}"
    )

    from src.visualization.attention_viz import AttentionVisualizer
    import matplotlib.pyplot as plt

    viz = AttentionVisualizer(
        attention_weights=attention_weights,
        token_type_ids=token_type_ids,
        spatial_size=spatial_size,
        image=image,
        dpi=args.dpi,
        colormap=args.colormap,
    )

    if args.full_report:
        viz.create_summary_report(
            output_dir,
            layers_to_visualize=layers,
            include_animation=args.include_animation,
        )
    else:

        def save(fig, name):
            fig.savefig(output_dir / name, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

        if "mask" in viz_types:
            save(viz.plot_attention_mask(layer=layers[0]), "attention_mask.png")
        if "evolution" in viz_types:
            save(viz.plot_layer_evolution(), "layer_evolution.png")
        if "query_to_image" in viz_types:
            for l in layers:
                save(
                    viz.plot_query_to_image(viz.n_query // 2, l),
                    f"query_to_image_layer_{l:02d}.png",
                )
        if "causal" in viz_types:
            for l in layers:
                save(viz.plot_causal_flow(l), f"causal_flow_layer_{l:02d}.png")
        if "entropy" in viz_types:
            save(viz.plot_entropy_analysis(), "entropy_analysis.png")

    save_feature_visualizations(args, artifacts, layers, output_dir)

    save_numeric_metric(args, attention_weights, spatial_size, output_dir)
    save_raw_data(args, artifacts, layers, output_dir)

    print(f"Done! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
