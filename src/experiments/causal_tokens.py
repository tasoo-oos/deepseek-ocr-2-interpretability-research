#!/usr/bin/env python3
"""
Run a focused causal-token study for DeepSeek-OCR-2.

The experiment targets three questions:
1. Do query indices sweep through the page in a stable spatial order?
2. Do final query states linearly encode the attended spatial position?
3. Does ablating a query block inside D2E perturb later queries more than earlier ones?

Outputs:
- summary.json
- summary.md
- generated stimuli under <output_dir>/stimuli
- simple trajectory plots under <output_dir>/plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class TrajectoryMetrics:
    stimulus: str
    pearson_x: float
    pearson_y: float
    early_mean_y: float
    late_mean_y: float
    early_mean_x: float
    late_mean_x: float
    step_distance: float


@dataclass
class DirectionalityMetrics:
    stimulus: str
    layer: int
    start_idx: int
    end_idx: int
    prefix_cosine_drop: float
    suffix_cosine_drop: float
    suffix_over_prefix: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research causal query tokens in DeepSeek-OCR-2")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR-2",
        help="HuggingFace model id or local snapshot directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/causal_token_research",
        help="Directory for research outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--ablation_layer",
        type=int,
        default=12,
        help="D2E layer used for the causal directionality intervention",
    )
    parser.add_argument(
        "--block_start",
        type=int,
        default=96,
        help="First query index in the ablated block",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Number of query positions to ablate inside the chosen layer",
    )
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
    if not snapshots:
        return model_path

    return str(snapshots[-1])


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def create_canvas() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1024, 1024), color="white")
    draw = ImageDraw.Draw(image)
    return image, draw


def draw_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str, font, fill: str = "#f6f6f6") -> None:
    draw.rounded_rectangle(box, radius=14, outline="black", width=3, fill=fill)
    x0, y0, x1, y1 = box
    draw.text((x0 + 18, y0 + 18), label, fill="black", font=font)


def make_single_column(font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((72, 44), "Single Column Reading Order", fill="black", font=font_title)
    for idx in range(10):
        top = 120 + idx * 82
        draw_box(draw, (72, top, 952, top + 58), f"Line {idx + 1:02d}  invoice item  amount", font_body)
    return image


def make_two_column(font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((72, 44), "Two Column Layout", fill="black", font=font_title)
    for row in range(6):
        top = 140 + row * 128
        draw_box(draw, (72, top, 454, top + 84), f"L{row + 1:02d} left column text", font_body, fill="#f1f5ff")
        draw_box(draw, (570, top, 952, top + 84), f"R{row + 1:02d} right column text", font_body, fill="#fff5ee")
    return image


def make_header_body_footer(font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw_box(draw, (72, 44, 952, 164), "HEADER  quarterly report", font_title, fill="#eef8ea")
    for idx in range(5):
        top = 220 + idx * 122
        draw_box(draw, (72, top, 952, top + 82), f"Body block {idx + 1:02d}  paragraph  metrics  notes", font_body)
    draw_box(draw, (72, 890, 952, 980), "FOOTER  signatures  references", font_body, fill="#f8f0ea")
    return image


def make_zigzag(font_body, font_title) -> Image.Image:
    image, draw = create_canvas()
    draw.text((72, 44), "Zigzag Notes", fill="black", font=font_title)
    positions = [
        (72, 150, 470, 240),
        (554, 260, 952, 350),
        (72, 370, 470, 460),
        (554, 480, 952, 570),
        (72, 590, 470, 680),
        (554, 700, 952, 790),
    ]
    for idx, box in enumerate(positions):
        draw_box(draw, box, f"Step {idx + 1:02d}  process note", font_body, fill="#f9f9f2")
    return image


def build_stimuli() -> Dict[str, Image.Image]:
    font_title = load_font(34)
    font_body = load_font(24)
    return {
        "single_column": make_single_column(font_body, font_title),
        "two_column": make_two_column(font_body, font_title),
        "header_body_footer": make_header_body_footer(font_body, font_title),
        "zigzag": make_zigzag(font_body, font_title),
    }


def save_stimuli(stimuli: Dict[str, Image.Image], output_dir: Path) -> None:
    stimuli_dir = output_dir / "stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    for name, image in stimuli.items():
        image.save(stimuli_dir / f"{name}.png")


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x_centered * y_centered).sum().item() / denom.item())


def compute_attention_centers(attn: torch.Tensor, spatial_size: int) -> torch.Tensor:
    n_image = spatial_size * spatial_size
    q2i = attn[:, n_image:, :n_image].float().mean(dim=0)
    q2i = q2i / q2i.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    coords_y, coords_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, spatial_size, device=attn.device, dtype=q2i.dtype),
        torch.linspace(0.0, 1.0, spatial_size, device=attn.device, dtype=q2i.dtype),
        indexing="ij",
    )
    flat_x = coords_x.reshape(-1)
    flat_y = coords_y.reshape(-1)

    center_x = q2i @ flat_x
    center_y = q2i @ flat_y
    return torch.stack([center_x, center_y], dim=-1)


def compute_trajectory_metrics(name: str, centers: torch.Tensor) -> TrajectoryMetrics:
    n_query = centers.shape[0]
    query_idx = torch.arange(n_query, device=centers.device, dtype=torch.float32)
    quarter = max(1, n_query // 4)

    diffs = centers[1:] - centers[:-1]
    step_distance = diffs.norm(dim=-1).mean().item()

    return TrajectoryMetrics(
        stimulus=name,
        pearson_x=pearson(query_idx, centers[:, 0]),
        pearson_y=pearson(query_idx, centers[:, 1]),
        early_mean_y=float(centers[:quarter, 1].mean().item()),
        late_mean_y=float(centers[-quarter:, 1].mean().item()),
        early_mean_x=float(centers[:quarter, 0].mean().item()),
        late_mean_x=float(centers[-quarter:, 0].mean().item()),
        step_distance=float(step_distance),
    )


def plot_query_trajectory(name: str, centers: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = torch.arange(centers.shape[0]).cpu()
    x = centers[:, 0].detach().cpu()
    y = centers[:, 1].detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].plot(idx, y, color="tab:blue")
    axes[0].set_title(f"{name}: query index vs y")
    axes[0].set_xlabel("query index")
    axes[0].set_ylabel("attended y")

    axes[1].plot(idx, x, color="tab:orange")
    axes[1].set_title(f"{name}: query index vs x")
    axes[1].set_xlabel("query index")
    axes[1].set_ylabel("attended x")

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_trajectory.png")
    plt.close(fig)


def run_global_d2e(model, processor, image: Image.Image, device: str):
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


def cosine_drop(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    cos = F.cosine_similarity(reference.float(), candidate.float(), dim=-1)
    return float((1.0 - cos.mean()).item())


def run_directionality_experiment(
    model,
    processor,
    image: Image.Image,
    device: str,
    layer: int,
    start_idx: int,
    end_idx: int,
) -> Dict[str, float]:
    from src.analysis.interventions import InterventionManager

    baseline = run_global_d2e(model, processor, image, device)["query_outputs"]

    inputs = processor.process_image(image)
    pixel_values = inputs["pixel_values"][0].to(device=device, dtype=next(model.sam_model.parameters()).dtype)

    with InterventionManager(model) as mgr:
        mgr.ablate_query_states_in_layer(layer=layer, start_idx=start_idx, end_idx=end_idx)
        with torch.no_grad():
            sam_features = model.sam_model(pixel_values)
            ablated = model.qwen2_model(
                sam_features,
                output_attentions=False,
                output_hidden_states=False,
            )[0].detach()

    prefix = slice(0, start_idx)
    suffix = slice(end_idx, baseline.shape[0])

    prefix_drop = cosine_drop(baseline[prefix], ablated[prefix]) if start_idx > 0 else 0.0
    suffix_drop = cosine_drop(baseline[suffix], ablated[suffix]) if end_idx < baseline.shape[0] else 0.0
    ratio = suffix_drop / max(prefix_drop, 1e-9)

    return {
        "prefix_cosine_drop": prefix_drop,
        "suffix_cosine_drop": suffix_drop,
        "suffix_over_prefix": ratio,
    }


def build_probe_dataset(results: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    for result in results.values():
        final_queries = result["query_outputs"].detach().cpu()
        centers = compute_attention_centers(result["attentions"][-1].cpu(), result["spatial_size"])
        features.append(final_queries)
        targets.append(centers)
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def write_summary_markdown(
    output_path: Path,
    trajectory_metrics: List[TrajectoryMetrics],
    directionality_metrics: List[DirectionalityMetrics],
    probe_metrics: Dict[str, object],
    query_bank_metrics: Dict[str, float],
) -> None:
    lines = [
        "# Causal Token Research Summary",
        "",
        "## Query Trajectory",
        "",
        "| Stimulus | corr(query,x) | corr(query,y) | early y | late y | step distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in trajectory_metrics:
        lines.append(
            f"| {metric.stimulus} | {metric.pearson_x:.3f} | {metric.pearson_y:.3f} | "
            f"{metric.early_mean_y:.3f} | {metric.late_mean_y:.3f} | {metric.step_distance:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Directionality Ablation",
            "",
            "| Stimulus | Layer | Block | prefix drop | suffix drop | suffix/prefix |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in directionality_metrics:
        lines.append(
            f"| {metric.stimulus} | {metric.layer} | {metric.start_idx}:{metric.end_idx} | "
            f"{metric.prefix_cosine_drop:.4f} | {metric.suffix_cosine_drop:.4f} | {metric.suffix_over_prefix:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Probe And Query Geometry",
            "",
            f"- Spatial probe MSE: {probe_metrics['mse']:.6f}",
            f"- Spatial probe R2 x: {probe_metrics['r2_x']:.4f}",
            f"- Spatial probe R2 y: {probe_metrics['r2_y']:.4f}",
            f"- query_1024 mean abs cosine: {query_bank_metrics['query_1024_mean_abs_cosine']:.4f}",
            f"- query_768 mean abs cosine: {query_bank_metrics['query_768_mean_abs_cosine']:.4f}",
            f"- max cross-resolution cosine: {query_bank_metrics['max_cross_resolution_cosine']:.4f}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()

    from src.analysis.query_analysis import QuerySpecializationAnalyzer
    from src.analysis.spatial_analysis import LinearSpatialProbe
    from src.models.deepseek_ocr import DeepseekOCRModel
    from src.preprocessing.image_transforms import ImageProcessor

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
    stimuli = build_stimuli()
    save_stimuli(stimuli, output_dir)

    raw_results: Dict[str, Dict[str, torch.Tensor]] = {}
    trajectory_metrics: List[TrajectoryMetrics] = []
    directionality_metrics: List[DirectionalityMetrics] = []

    for name, image in stimuli.items():
        print(f"Running stimulus: {name}")
        result = run_global_d2e(model, processor, image, args.device)
        raw_results[name] = result

        centers = compute_attention_centers(result["attentions"][-1], result["spatial_size"])
        metric = compute_trajectory_metrics(name, centers)
        trajectory_metrics.append(metric)
        plot_query_trajectory(name, centers, output_dir / "plots")

        direction = run_directionality_experiment(
            model=model,
            processor=processor,
            image=image,
            device=args.device,
            layer=args.ablation_layer,
            start_idx=args.block_start,
            end_idx=args.block_start + args.block_size,
        )
        directionality_metrics.append(
            DirectionalityMetrics(
                stimulus=name,
                layer=args.ablation_layer,
                start_idx=args.block_start,
                end_idx=args.block_start + args.block_size,
                prefix_cosine_drop=direction["prefix_cosine_drop"],
                suffix_cosine_drop=direction["suffix_cosine_drop"],
                suffix_over_prefix=direction["suffix_over_prefix"],
            )
        )

    probe_x, probe_y = build_probe_dataset(raw_results)
    probe = LinearSpatialProbe(l2_penalty=1e-4).fit(probe_x, probe_y)
    probe_eval = probe.evaluate(probe_x, probe_y)
    probe_metrics = {
        "mse": float(probe_eval.mse),
        "r2_x": float(probe_eval.r2[0].item()),
        "r2_y": float(probe_eval.r2[1].item()),
    }

    query_analyzer = QuerySpecializationAnalyzer(model.qwen2_model)
    q1024 = query_analyzer.summarize_query_bank(1024)
    q768 = query_analyzer.summarize_query_bank(768)
    cross = query_analyzer.cross_resolution_similarity()
    query_bank_metrics = {
        "query_1024_mean_abs_cosine": float(q1024.mean_abs_cosine),
        "query_768_mean_abs_cosine": float(q768.mean_abs_cosine),
        "max_cross_resolution_cosine": float(cross.max().item()),
    }

    payload = {
        "model_path": model_path,
        "device": args.device,
        "dtype": args.dtype,
        "ablation_layer": args.ablation_layer,
        "ablation_block": [args.block_start, args.block_start + args.block_size],
        "trajectory_metrics": [asdict(metric) for metric in trajectory_metrics],
        "directionality_metrics": [asdict(metric) for metric in directionality_metrics],
        "probe_metrics": probe_metrics,
        "query_bank_metrics": query_bank_metrics,
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="ascii")
    write_summary_markdown(
        summary_md,
        trajectory_metrics,
        directionality_metrics,
        probe_metrics,
        query_bank_metrics,
    )

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
