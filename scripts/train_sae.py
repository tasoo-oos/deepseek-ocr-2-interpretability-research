#!/usr/bin/env python3
"""Train a sparse autoencoder on DeepSeek-OCR-2 D2E query states."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

from src.analysis.sparse_autoencoder import (
    SparseAutoencoder,
    SparseAutoencoderAnalyzer,
    SparseAutoencoderSummary,
    SparseAutoencoderTrainer,
)
from src.models.deepseek_ocr import DeepseekOCRModel
from src.preprocessing.image_transforms import ImageProcessor


HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SAE on D2E query activations")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--output_dir", default="output/sae")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layer", type=int, default=12, help="D2E layer to extract query states from")
    parser.add_argument("--num_documents", type=int, default=64)
    parser.add_argument("--dictionary_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1_coeff", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_example_count", type=int, default=12)
    parser.add_argument("--min_activation_frequency", type=float, default=0.01)
    parser.add_argument("--top_feature_count", type=int, default=24)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)
    if "/" not in model_path:
        return model_path
    org, repo = model_path.split("/", 1)
    cache_dir = HF_CACHE_ROOT / f"models--{org}--{repo}" / "snapshots"
    if not cache_dir.exists():
        return model_path
    snapshots = sorted([p for p in cache_dir.iterdir() if p.is_dir()])
    return str(snapshots[-1]) if snapshots else model_path


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def create_canvas() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1024, 1024), color="white")
    return image, ImageDraw.Draw(image)


def draw_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill: str) -> None:
    draw.rounded_rectangle(box, radius=12, outline="black", width=2, fill=fill)
    draw.text((box[0] + 14, box[1] + 14), text, fill="black", font=font)


def make_single_column(rng: random.Random, font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((60, 40), "Single Column", fill="black", font=font_title)
    n_lines = rng.randint(8, 12)
    start_y = rng.randint(110, 180)
    step = rng.randint(68, 86)
    for idx in range(n_lines):
        top = start_y + idx * step
        height = rng.randint(48, 66)
        draw_box(draw, (70, top, 950, top + height), f"Line {idx + 1:02d}  item  amount  note", font_body, "#f7f7f7")
    return image


def make_two_column(rng: random.Random, font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((60, 40), "Two Column Page", fill="black", font=font_title)
    rows = rng.randint(5, 7)
    start_y = rng.randint(110, 160)
    step = rng.randint(115, 135)
    left_width = rng.randint(330, 400)
    gap = rng.randint(80, 120)
    for row in range(rows):
        top = start_y + row * step
        height = rng.randint(70, 90)
        draw_box(draw, (70, top, 70 + left_width, top + height), f"L{row + 1:02d} left column block", font_body, "#edf4ff")
        right_x0 = 70 + left_width + gap
        draw_box(draw, (right_x0, top, 950, top + height), f"R{row + 1:02d} right column block", font_body, "#fff4ea")
    return image


def make_header_body_footer(rng: random.Random, font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    header_h = rng.randint(90, 140)
    footer_h = rng.randint(70, 100)
    draw_box(draw, (60, 40, 964, 40 + header_h), "HEADER  report  quarter  summary", font_title, "#e8f5e4")
    current_y = 40 + header_h + rng.randint(40, 60)
    block_count = rng.randint(4, 6)
    for idx in range(block_count):
        height = rng.randint(68, 100)
        draw_box(draw, (70, current_y, 954, current_y + height), f"Body block {idx + 1:02d}  paragraph  metrics  notes", font_body, "#f7f7f7")
        current_y += height + rng.randint(28, 44)
    draw_box(draw, (60, 964 - footer_h, 964, 964), "FOOTER  appendix  signatures", font_body, "#f6eee7")
    return image


def make_table(rng: random.Random, font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((60, 40), "Table Layout", fill="black", font=font_title)
    cols = rng.randint(3, 5)
    rows = rng.randint(5, 7)
    x_positions = [70]
    for _ in range(cols - 1):
        x_positions.append(x_positions[-1] + rng.randint(150, 220))
    x_positions.append(950)
    y = 140
    for row in range(rows):
        height = rng.randint(80, 100)
        for col in range(cols):
            x0 = x_positions[col]
            x1 = x_positions[col + 1] - 10
            label = f"r{row + 1} c{col + 1}"
            fill = "#eef3ff" if row == 0 else "#fafafa"
            draw_box(draw, (x0, y, x1, y + height), label, font_body, fill)
        y += height + 12
    return image


def make_zigzag(rng: random.Random, font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((60, 40), "Zigzag Notes", fill="black", font=font_title)
    y = rng.randint(140, 180)
    left = True
    for idx in range(rng.randint(6, 8)):
        width = rng.randint(320, 430)
        height = rng.randint(70, 95)
        if left:
            x0 = rng.randint(70, 130)
        else:
            x0 = rng.randint(520, 600)
        draw_box(draw, (x0, y, x0 + width, y + height), f"Step {idx + 1:02d}  process note", font_body, "#faf8ee")
        y += rng.randint(95, 120)
        left = not left
    return image


def make_random_document(rng: random.Random, font_body, font_title) -> Tuple[str, Image.Image]:
    builders = {
        "single_column": make_single_column,
        "two_column": make_two_column,
        "header_body_footer": make_header_body_footer,
        "table": make_table,
        "zigzag": make_zigzag,
    }
    layout = rng.choice(list(builders.keys()))
    return layout, builders[layout](rng, font_body, font_title)


def compute_attention_centers(attn: torch.Tensor, spatial_size: int) -> torch.Tensor:
    n_image = spatial_size * spatial_size
    q2i = attn[:, n_image:, :n_image].float().mean(dim=0)
    q2i = q2i / q2i.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    coords_y, coords_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, spatial_size, device=attn.device, dtype=q2i.dtype),
        torch.linspace(0.0, 1.0, spatial_size, device=attn.device, dtype=q2i.dtype),
        indexing="ij",
    )
    return torch.stack([q2i @ coords_x.reshape(-1), q2i @ coords_y.reshape(-1)], dim=-1)


def collect_query_states(
    model: DeepseekOCRModel,
    processor: ImageProcessor,
    image: Image.Image,
    layer: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    holder: Dict[str, torch.Tensor] = {}
    target = model.qwen2_model.model.model.layers[layer]

    def hook(_module, _input, output):
        holder["hidden"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = target.register_forward_hook(hook)
    try:
        inputs = processor.process_image(image)
        pixel_values = inputs["pixel_values"][0].to(device=device, dtype=next(model.sam_model.parameters()).dtype)
        with torch.no_grad():
            sam_features = model.sam_model(pixel_values)
            _, attentions, _, _ = model.qwen2_model(
                sam_features,
                output_attentions=True,
                output_hidden_states=False,
            )
        hidden = holder["hidden"][0]
        n_image = hidden.shape[0] // 2
        query_states = hidden[n_image:, :].detach().cpu().float()
        centers = compute_attention_centers(attentions[-1][0], sam_features.shape[-1]).detach().cpu().float()
        return query_states, centers
    finally:
        handle.remove()


def annotate_feature(summary: Dict[str, object]) -> str:
    tags: List[str] = []
    x_mean = summary.get("weighted_attention_x_mean")
    y_mean = summary.get("weighted_attention_y_mean")
    x_std = summary.get("weighted_attention_x_std")
    y_std = summary.get("weighted_attention_y_std")
    q_std = summary.get("weighted_query_index_std")

    if isinstance(y_mean, float) and isinstance(y_std, float) and y_std < 0.12:
        if y_mean < 0.25:
            tags.append("top")
        elif y_mean > 0.75:
            tags.append("bottom")
    if isinstance(x_mean, float) and isinstance(x_std, float) and x_std < 0.12:
        if x_mean < 0.25:
            tags.append("left")
        elif x_mean > 0.75:
            tags.append("right")
    if isinstance(q_std, float) and q_std < 35:
        tags.append("narrow_query_band")
    return "+".join(tags) if tags else "mixed"


def save_examples(examples: List[Tuple[str, Image.Image]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (layout, image) in enumerate(examples):
        image.save(output_dir / f"{idx:03d}_{layout}.png")


def save_feature_plot(feature_rows: List[Dict[str, object]], output_path: Path) -> None:
    if not feature_rows:
        return
    xs = [row["weighted_attention_x_mean"] for row in feature_rows if row["weighted_attention_x_mean"] is not None]
    ys = [row["weighted_attention_y_mean"] for row in feature_rows if row["weighted_attention_y_mean"] is not None]
    sizes = [max(20.0, float(row["activation_frequency"]) * 800.0) for row in feature_rows if row["weighted_attention_x_mean"] is not None]
    colors = [row["weighted_query_index_mean"] for row in feature_rows if row["weighted_attention_x_mean"] is not None]
    labels = [row["feature_index"] for row in feature_rows if row["weighted_attention_x_mean"] is not None]
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6, 6), dpi=170)
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.75)
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, str(label), fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Weighted attention x")
    ax.set_ylabel("Weighted attention y")
    ax.set_title("SAE feature centers")
    fig.colorbar(scatter, ax=ax, label="Weighted query index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def select_feature_views(
    all_rows: List[Dict[str, object]],
    count: int,
) -> Dict[str, List[Dict[str, object]]]:
    active = sorted(
        all_rows,
        key=lambda row: (row["activation_frequency"], row["mean_activation"]),
        reverse=True,
    )[:count]
    query_localized = sorted(
        [row for row in all_rows if row["weighted_query_index_std"] is not None and row["activation_frequency"] >= 0.05],
        key=lambda row: (row["weighted_query_index_std"], -row["mean_activation"]),
    )[:count]
    spatially_localized = sorted(
        [
            row
            for row in all_rows
            if row["weighted_attention_x_std"] is not None and row["weighted_attention_y_std"] is not None
            and row["activation_frequency"] >= 0.05
        ],
        key=lambda row: (
            row["weighted_attention_x_std"] + row["weighted_attention_y_std"],
            -row["mean_activation"],
        ),
    )[:count]
    return {
        "top_active_features": active,
        "top_query_localized_features": query_localized,
        "top_spatially_localized_features": spatially_localized,
    }


def write_summary_markdown(
    output_path: Path,
    summary: SparseAutoencoderSummary,
    feature_views: Dict[str, List[Dict[str, object]]],
    config: Dict[str, object],
) -> None:
    lines = [
        "# SAE Summary",
        "",
        f"- Layer: {config['layer']}",
        f"- Documents: {config['num_documents']}",
        f"- Dictionary size: {config['dictionary_size']}",
        f"- MSE: {summary.metrics.mse:.6f}",
        f"- Explained variance: {summary.metrics.explained_variance:.4f}",
        f"- Mean active features per sample: {summary.mean_active_features_per_sample:.2f}",
        f"- Dead feature fraction: {summary.dead_feature_fraction:.4f}",
        "",
        "## Top Active Features",
        "",
        "| feature | tag | freq | mean act | q mean | q std | x mean | y mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in feature_views["top_active_features"]:
        lines.append(
            f"| {row['feature_index']} | {row['tag']} | {row['activation_frequency']:.3f} | "
            f"{row['mean_activation']:.3f} | {row['weighted_query_index_mean'] if row['weighted_query_index_mean'] is not None else 'NA'} | "
            f"{row['weighted_query_index_std'] if row['weighted_query_index_std'] is not None else 'NA'} | "
            f"{row['weighted_attention_x_mean'] if row['weighted_attention_x_mean'] is not None else 'NA'} | "
            f"{row['weighted_attention_y_mean'] if row['weighted_attention_y_mean'] is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Query-Localized Features",
            "",
            "| feature | tag | freq | mean act | q mean | q std | x mean | y mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in feature_views["top_query_localized_features"]:
        lines.append(
            f"| {row['feature_index']} | {row['tag']} | {row['activation_frequency']:.3f} | "
            f"{row['mean_activation']:.3f} | {row['weighted_query_index_mean'] if row['weighted_query_index_mean'] is not None else 'NA'} | "
            f"{row['weighted_query_index_std'] if row['weighted_query_index_std'] is not None else 'NA'} | "
            f"{row['weighted_attention_x_mean'] if row['weighted_attention_x_mean'] is not None else 'NA'} | "
            f"{row['weighted_attention_y_mean'] if row['weighted_attention_y_mean'] is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Spatially Localized Features",
            "",
            "| feature | tag | freq | mean act | q mean | q std | x mean | y mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in feature_views["top_spatially_localized_features"]:
        lines.append(
            f"| {row['feature_index']} | {row['tag']} | {row['activation_frequency']:.3f} | "
            f"{row['mean_activation']:.3f} | {row['weighted_query_index_mean'] if row['weighted_query_index_mean'] is not None else 'NA'} | "
            f"{row['weighted_query_index_std'] if row['weighted_query_index_std'] is not None else 'NA'} | "
            f"{row['weighted_attention_x_mean'] if row['weighted_attention_x_mean'] is not None else 'NA'} | "
            f"{row['weighted_attention_y_mean'] if row['weighted_attention_y_mean'] is not None else 'NA'} |"
        )
    output_path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model_path = resolve_model_path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    model = DeepseekOCRModel.from_pretrained(
        model_path,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation="eager",
        output_attentions=True,
        output_hidden_states=False,
    )
    model.eval()
    processor = ImageProcessor(crop_mode=False)

    font_title = load_font(34)
    font_body = load_font(24)

    activations = []
    metadata = {
        "query_index": [],
        "attention_x": [],
        "attention_y": [],
        "stimulus": [],
    }
    examples: List[Tuple[str, Image.Image]] = []

    for doc_idx in range(args.num_documents):
        layout, image = make_random_document(rng, font_body, font_title)
        if doc_idx < args.save_example_count:
            examples.append((layout, image))
        query_states, centers = collect_query_states(model, processor, image, args.layer, args.device)
        activations.append(query_states)
        metadata["query_index"].append(torch.arange(query_states.shape[0], dtype=torch.float32))
        metadata["attention_x"].append(centers[:, 0])
        metadata["attention_y"].append(centers[:, 1])
        metadata["stimulus"].extend([layout] * query_states.shape[0])
        print(f"Collected document {doc_idx + 1}/{args.num_documents}: {layout}")

    activation_tensor = torch.cat(activations, dim=0)
    flat_metadata = {
        "query_index": torch.cat(metadata["query_index"], dim=0),
        "attention_x": torch.cat(metadata["attention_x"], dim=0),
        "attention_y": torch.cat(metadata["attention_y"], dim=0),
        "stimulus": metadata["stimulus"],
    }

    sae = SparseAutoencoder(input_dim=activation_tensor.shape[1], n_features=args.dictionary_size)
    trainer = SparseAutoencoderTrainer(
        sae,
        lr=args.lr,
        l1_coeff=args.l1_coeff,
        batch_size=args.batch_size,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
    )
    metrics = trainer.fit(activation_tensor)
    print(
        "SAE metrics:",
        json.dumps(
            {
                "mse": metrics.mse,
                "l1": metrics.l1,
                "l0": metrics.l0,
                "explained_variance": metrics.explained_variance,
            },
            indent=2,
        ),
    )

    analyzer = SparseAutoencoderAnalyzer(sae)
    summary = analyzer.summarize(
        activation_tensor,
        metadata=flat_metadata,
        min_activation_frequency=args.min_activation_frequency,
    )

    all_feature_rows = []
    for feature in summary.feature_summaries:
        row = asdict(feature)
        row["tag"] = annotate_feature(row)
        all_feature_rows.append(row)

    feature_views = select_feature_views(all_feature_rows, args.top_feature_count)

    save_examples(examples, output_dir / "examples")
    save_feature_plot(feature_views["top_spatially_localized_features"], output_dir / "plots" / "feature_centers.png")

    config = {
        "model_path": model_path,
        "layer": args.layer,
        "num_documents": args.num_documents,
        "dictionary_size": args.dictionary_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l1_coeff": args.l1_coeff,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
    }
    checkpoint_path = output_dir / f"sae_layer_{args.layer}.pt"
    torch.save(
        {
            "config": config,
            "state_dict": sae.state_dict(),
        },
        checkpoint_path,
    )

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    payload = {
        "config": config,
        "metrics": asdict(summary.metrics),
        "dead_feature_fraction": summary.dead_feature_fraction,
        "mean_active_features_per_sample": summary.mean_active_features_per_sample,
        "feature_views": feature_views,
        "all_features": all_feature_rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="ascii")
    write_summary_markdown(summary_md, summary, feature_views, config)
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
