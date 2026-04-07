"""
Attention pattern visualizer for the D2E Visual Causal Flow mechanism.

Migrated from visualization/attention_visualizer.py with updated imports.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .utils import (
    attention_to_numpy,
    aggregate_heads,
    compute_attention_entropy,
    create_attention_mask_visualization,
    overlay_attention_on_image,
    position_to_spatial_coords,
    reshape_attention_to_spatial,
)


class AttentionVisualizer:
    """
    Visualizer for attention patterns extracted from the D2E model.

    Args:
        attention_weights: List of per-layer attention tensors [B, H, S, S].
        token_type_ids:    [B, S] with 0=image, 1=query.
        spatial_size:      H (= W) of the SAM feature grid.
        image:             Optional PIL image for overlays.
        figsize_scale:     Scale factor for figure sizes.
        dpi:               DPI for saved figures.
        colormap:          Default matplotlib colormap.
    """

    def __init__(
        self,
        attention_weights: List,
        token_type_ids=None,
        spatial_size: int = 16,
        image: Optional[Image.Image] = None,
        figsize_scale: float = 1.0,
        dpi: int = 150,
        colormap: str = "viridis",
    ):
        self.attention_weights = attention_weights
        self.token_type_ids = token_type_ids
        self.spatial_size = spatial_size
        self.image = image
        self.figsize_scale = figsize_scale
        self.dpi = dpi
        self.colormap = colormap

        self.n_layers = len(attention_weights)
        self.n_heads = attention_weights[0].shape[1]
        self.n_image = spatial_size * spatial_size
        self.n_query = self.n_image  # D2E uses same count for queries

    # ------------------------------------------------------------------
    # Core plots
    # ------------------------------------------------------------------

    def plot_attention_mask(
        self,
        layer: int = 0,
        head: Optional[int] = None,
        show_expected: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot actual attention matrix alongside the expected mask pattern."""
        attn = self.attention_weights[layer]
        if head is not None:
            attn = attn[:, head:head+1]
        attn_avg = aggregate_heads(attn, method="mean")[0]
        attn_np = attention_to_numpy(attn_avg)

        if show_expected:
            fig, axes = plt.subplots(1, 2, figsize=(12 * self.figsize_scale, 5 * self.figsize_scale))
            im1 = axes[0].imshow(attn_np, cmap=self.colormap, aspect='auto')
            axes[0].set_title(f"Actual Attention (Layer {layer})")
            axes[0].set_xlabel("Key Position")
            axes[0].set_ylabel("Query Position")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            self._add_region_labels(axes[0])

            expected = create_attention_mask_visualization(self.n_image, self.n_query)
            im2 = axes[1].imshow(expected, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title("Expected Mask Pattern")
            axes[1].set_xlabel("Key Position")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            self._add_region_labels(axes[1])
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8 * self.figsize_scale, 6 * self.figsize_scale))
            else:
                fig = ax.figure
            im = ax.imshow(attn_np, cmap=self.colormap, aspect='auto')
            ax.set_title(f"Attention Pattern (Layer {layer})")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._add_region_labels(ax)

        plt.tight_layout()
        return fig

    def _add_region_labels(self, ax: plt.Axes) -> None:
        total = self.n_image + self.n_query
        ax.axhline(y=self.n_image - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=self.n_image - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(self.n_image / 2, -total * 0.02, 'Image', ha='center', fontsize=8, color='blue')
        ax.text(self.n_image + self.n_query / 2, -total * 0.02, 'Query', ha='center', fontsize=8, color='orange')
        ax.text(-total * 0.02, self.n_image / 2, 'Img', va='center', ha='right', fontsize=8, color='blue', rotation=90)
        ax.text(-total * 0.02, self.n_image + self.n_query / 2, 'Qry', va='center', ha='right', fontsize=8, color='orange', rotation=90)

    def plot_layer_evolution(
        self,
        head: Optional[int] = None,
        region: str = "all",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """Grid of attention patterns across all layers."""
        n_cols = 6
        n_rows = (self.n_layers + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (3 * n_cols * self.figsize_scale, 3 * n_rows * self.figsize_scale)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for layer_idx in range(self.n_layers):
            attn = self.attention_weights[layer_idx]
            if head is not None:
                attn = attn[:, head:head+1]
            attn_avg = aggregate_heads(attn, method="mean")[0]

            region_map = {
                "image_to_image": attn_avg[:self.n_image, :self.n_image],
                "query_to_image": attn_avg[self.n_image:, :self.n_image],
                "query_to_query": attn_avg[self.n_image:, self.n_image:],
                "all": attn_avg,
            }
            title_suffix = {"image_to_image": "I→I", "query_to_image": "Q→I",
                            "query_to_query": "Q→Q", "all": ""}.get(region, "")
            attn_np = attention_to_numpy(region_map.get(region, attn_avg))

            axes[layer_idx].imshow(attn_np, cmap=self.colormap, aspect='auto')
            axes[layer_idx].set_title(f"L{layer_idx} {title_suffix}", fontsize=10)
            axes[layer_idx].set_xticks([])
            axes[layer_idx].set_yticks([])

        for idx in range(self.n_layers, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"Attention Evolution Across {self.n_layers} Layers", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_query_to_image(
        self,
        query_idx: int,
        layer: int,
        head: Optional[int] = None,
        overlay_image: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Show which image regions a specific query token attends to."""
        attn = self.attention_weights[layer]
        if head is not None:
            attn = attn[:, head:head+1]
        attn_avg = aggregate_heads(attn, method="mean")[0]
        query_pos = self.n_image + query_idx
        attention_to_image = attn_avg[query_pos, :self.n_image]
        spatial_attn = reshape_attention_to_spatial(attention_to_image, self.spatial_size)
        attn_np = attention_to_numpy(spatial_attn)
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-9)

        if overlay_image and self.image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10 * self.figsize_scale, 5 * self.figsize_scale))
            im = axes[0].imshow(attn_np, cmap=self.colormap)
            axes[0].set_title(f"Query {query_idx} → Image (Layer {layer})")
            plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
            overlay = overlay_attention_on_image(self.image, attn_np, alpha=0.5, colormap=self.colormap)
            axes[1].imshow(overlay)
            axes[1].set_title("Attention Overlay")
            axes[1].axis('off')
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6 * self.figsize_scale, 5 * self.figsize_scale))
            else:
                fig = ax.figure
            im = ax.imshow(attn_np, cmap=self.colormap)
            ax.set_title(f"Query {query_idx} → Image (Layer {layer})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def plot_image_self_attention(
        self,
        position: Union[int, Tuple[int, int]],
        layer: int,
        head: Optional[int] = None,
        overlay_image: bool = True,
    ) -> plt.Figure:
        """Visualize which image positions a given image token attends to."""
        if isinstance(position, tuple):
            row, col = position
            flat_pos = row * self.spatial_size + col
        else:
            flat_pos = position
            row, col = position_to_spatial_coords(flat_pos, self.spatial_size)

        attn = self.attention_weights[layer]
        if head is not None:
            attn = attn[:, head:head+1]
        attn_avg = aggregate_heads(attn, method="mean")[0]
        attn_pattern = attn_avg[flat_pos, :self.n_image]
        spatial_attn = reshape_attention_to_spatial(attn_pattern, self.spatial_size)
        attn_np = attention_to_numpy(spatial_attn)
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-9)

        if overlay_image and self.image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10 * self.figsize_scale, 5 * self.figsize_scale))
            im = axes[0].imshow(attn_np, cmap=self.colormap)
            axes[0].plot(col, row, 'rx', markersize=15, markeredgewidth=3)
            axes[0].set_title(f"Image Position ({row},{col}) Self-Attention (Layer {layer})")
            plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
            overlay = overlay_attention_on_image(self.image, attn_np, alpha=0.5, colormap=self.colormap)
            axes[1].imshow(overlay)
            axes[1].set_title("Attention Overlay")
            axes[1].axis('off')
        else:
            fig, ax = plt.subplots(figsize=(6 * self.figsize_scale, 5 * self.figsize_scale))
            im = ax.imshow(attn_np, cmap=self.colormap)
            ax.plot(col, row, 'rx', markersize=15, markeredgewidth=3)
            ax.set_title(f"Image Position ({row},{col}) Self-Attention (Layer {layer})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def plot_causal_flow(
        self,
        layer: int,
        head: Optional[int] = None,
        show_mask_verification: bool = True,
    ) -> plt.Figure:
        """Visualize causal attention among query tokens."""
        attn = self.attention_weights[layer]
        if head is not None:
            attn = attn[:, head:head+1]
        attn_avg = aggregate_heads(attn, method="mean")[0]
        q2q = attn_avg[self.n_image:, self.n_image:]
        attn_np = attention_to_numpy(q2q)

        if show_mask_verification:
            fig, axes = plt.subplots(1, 2, figsize=(12 * self.figsize_scale, 5 * self.figsize_scale))
            im1 = axes[0].imshow(attn_np, cmap=self.colormap, aspect='auto')
            axes[0].set_title(f"Query→Query Causal Attention (Layer {layer})")
            axes[0].set_xlabel("Key Query Position")
            axes[0].set_ylabel("Query Position")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            expected = np.tril(np.ones_like(attn_np))
            im2 = axes[1].imshow(expected, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title("Expected Causal Mask")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            fig, ax = plt.subplots(figsize=(8 * self.figsize_scale, 6 * self.figsize_scale))
            im = ax.imshow(attn_np, cmap=self.colormap, aspect='auto')
            ax.set_title(f"Query→Query Causal Attention (Layer {layer})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def plot_head_comparison(
        self,
        layer: int,
        region: str = "query_to_image",
        n_cols: int = 7,
    ) -> plt.Figure:
        """Compare all attention heads for a layer."""
        attn = self.attention_weights[layer][0]  # [H, S, S]
        n_rows = (self.n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.5 * n_cols * self.figsize_scale, 2.5 * n_rows * self.figsize_scale)
        )
        axes = axes.flatten()

        for h in range(self.n_heads):
            head_attn = attn[h]
            region_map = {
                "image_to_image": head_attn[:self.n_image, :self.n_image],
                "query_to_image": head_attn[self.n_image:, :self.n_image],
                "query_to_query": head_attn[self.n_image:, self.n_image:],
                "all": head_attn,
            }
            axes[h].imshow(attention_to_numpy(region_map.get(region, head_attn)),
                           cmap=self.colormap, aspect='auto')
            axes[h].set_title(f"Head {h}", fontsize=9)
            axes[h].set_xticks([])
            axes[h].set_yticks([])

        for idx in range(self.n_heads, len(axes)):
            axes[idx].axis('off')

        region_label = {
            "all": "Full", "image_to_image": "Image→Image",
            "query_to_image": "Query→Image", "query_to_query": "Query→Query",
        }.get(region, region)
        plt.suptitle(f"Head Comparison — Layer {layer} — {region_label}", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_entropy_analysis(self) -> plt.Figure:
        """Plot attention entropy across layers and heads."""
        fig, axes = plt.subplots(2, 2, figsize=(12 * self.figsize_scale, 10 * self.figsize_scale))

        entropies = {"image_self": [], "query_to_image": [], "query_to_query": []}
        for attn in self.attention_weights:
            entropies["image_self"].append(
                compute_attention_entropy(attn[:, :, :self.n_image, :self.n_image]).mean().item()
            )
            entropies["query_to_image"].append(
                compute_attention_entropy(attn[:, :, self.n_image:, :self.n_image]).mean().item()
            )
            entropies["query_to_query"].append(
                compute_attention_entropy(attn[:, :, self.n_image:, self.n_image:]).mean().item()
            )

        layers = list(range(self.n_layers))
        axes[0, 0].plot(layers, entropies['image_self'], 'b-o', label='Image Self', markersize=4)
        axes[0, 0].plot(layers, entropies['query_to_image'], 'g-s', label='Query→Image', markersize=4)
        axes[0, 0].plot(layers, entropies['query_to_query'], 'r-^', label='Query→Query', markersize=4)
        axes[0, 0].set_xlabel("Layer")
        axes[0, 0].set_ylabel("Entropy")
        axes[0, 0].set_title("Attention Entropy by Region")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        final_attn = self.attention_weights[-1][0]
        head_entropies = [
            compute_attention_entropy(final_attn[h:h+1].unsqueeze(0)).mean().item()
            for h in range(self.n_heads)
        ]
        axes[0, 1].bar(range(self.n_heads), head_entropies, color='steelblue')
        axes[0, 1].set_xlabel("Head")
        axes[0, 1].set_ylabel("Entropy")
        axes[0, 1].set_title(f"Per-Head Entropy (Layer {self.n_layers - 1})")

        entropy_matrix = np.zeros((self.n_layers, self.n_heads))
        for layer_idx, attn in enumerate(self.attention_weights):
            for h in range(self.n_heads):
                q2i = attn[0, h, self.n_image:, :self.n_image]
                entropy_matrix[layer_idx, h] = compute_attention_entropy(
                    q2i.unsqueeze(0).unsqueeze(0)
                ).mean().item()

        im = axes[1, 0].imshow(entropy_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_xlabel("Head")
        axes[1, 0].set_ylabel("Layer")
        axes[1, 0].set_title("Query→Image Entropy (Layers × Heads)")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].axis('off')
        stats_text = (
            f"Model Configuration\n"
            f"===================\n"
            f"Layers: {self.n_layers}\n"
            f"Heads:  {self.n_heads}\n"
            f"Spatial: {self.spatial_size}×{self.spatial_size}\n"
            f"Image tokens: {self.n_image}\n"
            f"Query tokens: {self.n_query}\n\n"
            f"Avg Image Self-Attn Entropy:  {np.mean(entropies['image_self']):.3f}\n"
            f"Avg Query→Image Entropy:      {np.mean(entropies['query_to_image']):.3f}\n"
            f"Avg Query→Query Entropy:      {np.mean(entropies['query_to_query']):.3f}\n"
        )
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        plt.tight_layout()
        return fig

    def create_summary_report(
        self,
        output_dir: Union[str, Path],
        layers_to_visualize: Optional[List[int]] = None,
        include_animation: bool = False,
    ) -> None:
        """Generate a full visualization report and save to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if layers_to_visualize is None:
            n = self.n_layers
            layers_to_visualize = [0, n // 4, n // 2, 3 * n // 4, n - 1]

        print(f"Generating report in {output_dir}...")

        def _save(fig, path):
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        _save(self.plot_attention_mask(layer=0), output_dir / "attention_mask.png")
        _save(self.plot_layer_evolution(), output_dir / "layer_evolution.png")
        _save(self.plot_entropy_analysis(), output_dir / "entropy_analysis.png")

        query_dir = output_dir / "query_to_image"
        query_dir.mkdir(exist_ok=True)
        for layer in layers_to_visualize:
            for qi in [0, self.n_query // 2, self.n_query - 1]:
                _save(
                    self.plot_query_to_image(query_idx=qi, layer=layer),
                    query_dir / f"layer_{layer:02d}_query_{qi:03d}.png",
                )

        causal_dir = output_dir / "causal_flow"
        causal_dir.mkdir(exist_ok=True)
        for layer in layers_to_visualize:
            _save(self.plot_causal_flow(layer=layer), causal_dir / f"layer_{layer:02d}.png")

        head_dir = output_dir / "head_analysis"
        head_dir.mkdir(exist_ok=True)
        _save(
            self.plot_head_comparison(layer=self.n_layers - 1),
            head_dir / "head_comparison_query_to_image.png",
        )

        if include_animation:
            self._create_animation(output_dir / "animation.gif")

        with open(output_dir / "metadata.json", "w") as f:
            json.dump({
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "n_image_tokens": self.n_image,
                "n_query_tokens": self.n_query,
                "spatial_size": self.spatial_size,
                "layers_visualized": layers_to_visualize,
            }, f, indent=2)

        print(f"Report complete! Output saved to {output_dir}")

    def _create_animation(self, output_path: Path) -> None:
        try:
            import imageio
        except ImportError:
            print("imageio not installed — skipping animation")
            return

        frames = []
        for layer in range(self.n_layers):
            fig = self.plot_attention_mask(layer=layer, show_expected=False)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)

        imageio.mimsave(output_path, frames, duration=0.3)
