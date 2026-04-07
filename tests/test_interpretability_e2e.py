"""End-to-end and gap-filling tests for interpretability tooling."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_attention_weights(
    n_layers: int = 2, n_heads: int = 2, n_image: int = 4
) -> list[torch.Tensor]:
    seq = n_image * 2
    weights = []
    for layer in range(n_layers):
        attn = torch.full((1, n_heads, seq, seq), 1e-3)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        # Head 0: focused query->image attention
        attn[0, 0, n_image:, :n_image] = 0.0
        attn[0, 0, n_image:, 0] = 1.0
        # Head 1: diffuse query->image attention
        attn[0, 1, n_image:, :n_image] = 0.25
        # Make one layer/head have especially strong magnitude.
        if layer == 1:
            attn[0, 1, n_image:, :n_image] = 0.0
            attn[0, 1, n_image:, 2] = 1.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        weights.append(attn)
    return weights


def _write_temp_image(tmpdir: str | Path, name: str = "sample.png") -> Path:
    path = Path(tmpdir) / name
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(path)
    return path


def _run_python_script(
    args: list[str], *, timeout: int = 600
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env=env,
    )


def test_attention_analyzer_specialization_and_ranking():
    from src.analysis.attention_analysis import AttentionAnalyzer
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder

    analyzer = AttentionAnalyzer(
        build_qwen2_decoder_as_encoder(output_attentions=True).eval()
    )
    weights = _make_attention_weights()

    stats = analyzer.analyze_head_specialization(weights, n_image_tokens=4)
    assert set(stats) == {"entropy_q2i", "entropy_i2i", "entropy_q2q", "q2i_ratio"}
    assert stats["entropy_q2i"].shape == (2, 2)
    assert torch.all(stats["q2i_ratio"] >= 0)
    assert torch.all(stats["q2i_ratio"] <= 1.0 + 1e-6)

    important_entropy = analyzer.find_important_heads(
        weights, n_image_tokens=4, metric="entropy", region="query_to_image", top_k=2
    )
    important_mag = analyzer.find_important_heads(
        weights, n_image_tokens=4, metric="magnitude", region="query_to_image", top_k=2
    )
    assert len(important_entropy) == 2
    assert len(important_mag) == 2
    assert all(isinstance(x, tuple) and len(x) == 2 for x in important_entropy)


def test_visualization_utils_gap_coverage():
    from src.visualization.utils import (
        aggregate_heads,
        compute_attention_distance,
        create_attention_mask_visualization,
        extract_attention_regions,
        get_top_k_attended_positions,
        normalize_attention,
        overlay_attention_on_image,
        position_to_spatial_coords,
        reshape_attention_to_spatial,
        spatial_coords_to_position,
    )

    attn = torch.arange(1, 1 + 1 * 2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
    norm = normalize_attention(attn)
    assert torch.allclose(
        norm.sum(dim=-1), torch.ones_like(norm.sum(dim=-1)), atol=1e-5
    )

    mean_attn = aggregate_heads(norm, method="mean")
    max_attn = aggregate_heads(norm, method="max")
    specific = aggregate_heads(norm, method="specific", head_indices=[1])
    assert mean_attn.shape == (1, 4, 4)
    assert max_attn.shape == (1, 4, 4)
    assert specific.shape == (1, 1, 4, 4)

    values, indices = get_top_k_attended_positions(norm, k=2)
    assert values.shape == (1, 2, 4, 2)
    assert indices.shape == (1, 2, 4, 2)

    spatial = reshape_attention_to_spatial(torch.arange(16.0), spatial_size=4)
    assert spatial.shape == (4, 4)

    i2i, i2q, q2i, q2q = extract_attention_regions(torch.rand(1, 2, 8, 8), 4, 4)
    assert i2i.shape == (1, 2, 4, 4)
    assert i2q.shape == (1, 2, 4, 4)
    assert q2i.shape == (1, 2, 4, 4)
    assert q2q.shape == (1, 2, 4, 4)

    dist = compute_attention_distance(
        normalize_attention(torch.ones(1, 1, 4, 4)), spatial_size=2
    )
    assert dist.shape == (1, 1, 4)

    mask = create_attention_mask_visualization(4, 4)
    assert mask.shape == (8, 8)
    assert mask[0, 6] == 0.0
    assert mask[7, 0] == 1.0

    image = Image.new("RGB", (32, 32), color=(0, 0, 0))
    overlay = overlay_attention_on_image(image, np.array([[0.0, 1.0], [0.5, 0.25]]))
    assert overlay.size == image.size

    row, col = position_to_spatial_coords(5, 4)
    assert (row, col) == (1, 1)
    assert spatial_coords_to_position(row, col, 4) == 5


def test_visualizers_and_report_generation():
    from src.visualization.attention_viz import AttentionVisualizer
    from src.visualization.feature_viz import FeatureVisualizer

    weights = _make_attention_weights(n_layers=3, n_heads=2, n_image=4)
    viz = AttentionVisualizer(
        attention_weights=weights,
        token_type_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        spatial_size=2,
        image=Image.new("RGB", (64, 64), color=(128, 128, 128)),
    )

    figs = [
        viz.plot_layer_evolution(),
        viz.plot_query_to_image(query_idx=1, layer=0),
        viz.plot_image_self_attention(position=(0, 1), layer=0),
        viz.plot_causal_flow(layer=0),
        viz.plot_head_comparison(layer=0),
        viz.plot_entropy_analysis(),
    ]
    assert all(fig is not None for fig in figs)
    for fig in figs:
        plt.close(fig)

    with tempfile.TemporaryDirectory() as tmpdir:
        viz.create_summary_report(
            tmpdir, layers_to_visualize=[0, 2], include_animation=False
        )
        assert (Path(tmpdir) / "attention_mask.png").exists()
        assert (Path(tmpdir) / "layer_evolution.png").exists()
        assert (Path(tmpdir) / "entropy_analysis.png").exists()
        assert (Path(tmpdir) / "metadata.json").exists()

    fv = FeatureVisualizer()
    pytest.importorskip("sklearn")
    sam_fig = fv.plot_sam_features(
        torch.rand(1, 8, 4, 4), channels=[0, 1, 2, 3], n_cols=2
    )
    d2e_fig = fv.plot_d2e_hidden_states(torch.rand(1, 8, 16), positions=[0, 3])
    proj_fig = fv.plot_projector_output(torch.rand(1, 8, 32))
    traj_fig = fv.plot_activation_trajectory(
        {
            "sam_layer_0": torch.rand(1, 8, 2, 2),
            "d2e_layer_0": torch.rand(1, 4, 8),
            "projector": torch.rand(1, 4, 8),
        },
        position=1,
    )
    for fig in (sam_fig, d2e_fig, proj_fig, traj_fig):
        assert fig is not None
        plt.close(fig)


def test_extract_attention_cli_synthetic_full_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_python_script(
            [
                "scripts/extract_attention.py",
                "--synthetic",
                "--output_dir",
                tmpdir,
                "--full_report",
            ]
        )
        assert "Done! Output saved" in result.stdout
        assert (Path(tmpdir) / "attention_mask.png").exists()
        assert (Path(tmpdir) / "layer_evolution.png").exists()
        assert (Path(tmpdir) / "entropy_analysis.png").exists()
        assert (Path(tmpdir) / "metadata.json").exists()


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="GPU interpretability tests skipped (set RUN_GPU_TESTS=1)",
)
def test_extract_attention_cli_real_model_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = _write_temp_image(tmpdir)
        output_dir = Path(tmpdir) / "attention_report"
        result = _run_python_script(
            [
                "scripts/extract_attention.py",
                "--image_path",
                str(image_path),
                "--output_dir",
                str(output_dir),
                "--device",
                "cuda",
                "--layers",
                "0,23",
                "--full_report",
            ],
            timeout=1800,
        )
        assert "Extracted 24 attention layers" in result.stdout
        assert (output_dir / "attention_mask.png").exists()
        assert (output_dir / "metadata.json").exists()


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="GPU interpretability tests skipped (set RUN_GPU_TESTS=1)",
)
def test_extract_features_cli_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = _write_temp_image(tmpdir)
        output_path = Path(tmpdir) / "features.pt"
        result = _run_python_script(
            [
                "scripts/extract_features.py",
                "--image_path",
                str(image_path),
                "--output_path",
                str(output_path),
                "--device",
                "cuda",
                "--sam_layers",
                "0,11",
                "--d2e_layers",
                "0,23",
            ],
            timeout=1800,
        )
        assert "Saved to" in result.stdout
        assert output_path.exists()
        activations = torch.load(output_path, map_location="cpu")
        assert "sam_layer_0" in activations
        assert "sam_layer_11" in activations
        assert "d2e_layer_0" in activations
        assert "d2e_layer_23" in activations
        assert "projector" in activations


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="GPU interpretability tests skipped (set RUN_GPU_TESTS=1)",
)
def test_run_interventions_cli_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = _write_temp_image(tmpdir)
        result = _run_python_script(
            [
                "scripts/run_interventions.py",
                "--image_path",
                str(image_path),
                "--ablate_queries",
                "--device",
                "cuda",
            ],
            timeout=1800,
        )
        assert "Baseline embedding norm:" in result.stdout
        assert "Ablated embedding norm:" in result.stdout
        baseline = float(
            re.search(r"Baseline embedding norm: ([0-9.]+)", result.stdout).group(1)
        )
        ablated = float(
            re.search(r"Ablated embedding norm: ([0-9.]+)", result.stdout).group(1)
        )
        assert baseline > 0
        assert ablated > 0
        assert abs(ablated - baseline) > 1e-6
