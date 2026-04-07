"""
Visualize intermediate activations from non-attention components:
SAM feature maps, D2E hidden states, and projector outputs.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


class FeatureVisualizer:
    """
    Visualize SAM features, D2E hidden states, and projector outputs.

    All methods return matplotlib Figure objects — call fig.savefig() or plt.show()
    as needed.
    """

    def plot_sam_features(
        self,
        sam_output: torch.Tensor,
        channels: Optional[List[int]] = None,
        n_cols: int = 8,
    ) -> plt.Figure:
        """
        Visualize SAM encoder feature maps.

        Args:
            sam_output: [B, 896, H, W] — SAM encoder output.
            channels:   Channel indices to show (default: first 32).
            n_cols:     Columns in the subplot grid.

        Returns:
            matplotlib Figure.
        """
        feat = sam_output[0].float().cpu()  # [896, H, W]
        if channels is None:
            channels = list(range(min(32, feat.shape[0])))

        n_rows = (len(channels) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        axes = axes.flatten()

        for i, ch in enumerate(channels):
            channel_map = feat[ch].numpy()
            channel_map = (channel_map - channel_map.min()) / (
                channel_map.max() - channel_map.min() + 1e-9
            )
            axes[i].imshow(channel_map, cmap="viridis", aspect="auto")
            axes[i].set_title(f"ch {ch}", fontsize=7)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        for idx in range(len(channels), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("SAM Feature Maps", fontsize=13)
        plt.tight_layout()
        return fig

    def plot_d2e_hidden_states(
        self,
        hidden_states: torch.Tensor,
        layer: int = 0,
        positions: Optional[List[int]] = None,
        n_components: int = 2,
    ) -> plt.Figure:
        """
        Visualize D2E hidden states using PCA projection and per-token norms.

        Args:
            hidden_states: [B, seq_len, 896]
            layer:         Layer index (for title only).
            positions:     Subset of positions to label in the PCA plot.
            n_components:  Number of PCA components to plot.

        Returns:
            matplotlib Figure.
        """
        from sklearn.decomposition import PCA

        hs = hidden_states[0].float().cpu().numpy()  # [seq_len, 896]
        seq_len = hs.shape[0]

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(hs)  # [seq_len, 2]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # PCA scatter
        scatter = axes[0].scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=np.arange(seq_len),
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        if positions:
            for pos in positions:
                axes[0].annotate(
                    str(pos), (reduced[pos, 0], reduced[pos, 1]), fontsize=7
                )
        plt.colorbar(scatter, ax=axes[0], label="Token position")
        axes[0].set_title(f"PCA (Layer {layer})")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        # Token norms
        norms = np.linalg.norm(hs, axis=-1)
        axes[1].plot(norms)
        axes[1].set_title(f"Token Norms (Layer {layer})")
        axes[1].set_xlabel("Token position")
        axes[1].set_ylabel("L2 norm")
        axes[1].grid(True, alpha=0.3)

        # Activation heatmap (first 64 features)
        im = axes[2].imshow(hs[:, :64].T, cmap="coolwarm", aspect="auto")
        axes[2].set_title(f"First 64 Features (Layer {layer})")
        axes[2].set_xlabel("Token position")
        axes[2].set_ylabel("Feature index")
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        return fig

    def plot_projector_output(
        self,
        proj_output: torch.Tensor,
    ) -> plt.Figure:
        """
        Visualize projector output distribution.

        Args:
            proj_output: [B, seq_len, 1280]

        Returns:
            matplotlib Figure.
        """
        po = proj_output[0].float().cpu().numpy()  # [seq_len, 1280]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Token norms
        norms = np.linalg.norm(po, axis=-1)
        axes[0].plot(norms)
        axes[0].set_title("Projector Token Norms")
        axes[0].set_xlabel("Token position")
        axes[0].set_ylabel("L2 norm")
        axes[0].grid(True, alpha=0.3)

        # Feature distribution
        axes[1].hist(po.flatten(), bins=100, color="steelblue", alpha=0.7)
        axes[1].set_title("Projector Output Distribution")
        axes[1].set_xlabel("Activation value")
        axes[1].set_ylabel("Count")

        # Heatmap (first 64 features)
        im = axes[2].imshow(po[:, :64].T, cmap="coolwarm", aspect="auto")
        axes[2].set_title("First 64 Output Features")
        axes[2].set_xlabel("Token position")
        axes[2].set_ylabel("Feature index")
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        return fig

    def plot_activation_trajectory(
        self,
        activations: Dict[str, torch.Tensor],
        position: int = 0,
    ) -> plt.Figure:
        """
        Show how a single token position's representation evolves through the model.

        Expects activations dict with keys like 'sam_layer_X', 'd2e_layer_X', 'projector'.

        Args:
            activations: Dict from FeatureExtractor.extract()
            position:    Token position to track.

        Returns:
            matplotlib Figure.
        """
        layer_names = sorted(
            [k for k in activations if k.startswith(("sam_layer", "d2e_layer"))],
            key=lambda k: (k.split("_")[0], int(k.split("_")[-1])),
        )
        if "projector" in activations:
            layer_names.append("projector")

        norms = []
        for name in layer_names:
            act = activations[name]
            if act.dim() == 4:
                # SAM output: [B, C, H, W] — flatten to [B, H*W, C]
                act = act.flatten(2).permute(0, 2, 1)
            if position < act.shape[1]:
                norms.append(act[0, position, :].float().norm().item())
            else:
                norms.append(float("nan"))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(range(len(norms)), norms, "o-", markersize=5)
        axes[0].set_xticks(range(len(layer_names)))
        axes[0].set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
        axes[0].set_title(f"Representation Norm — Position {position}")
        axes[0].set_ylabel("L2 norm")
        axes[0].grid(True, alpha=0.3)

        # Compare first and last layers
        if len(layer_names) >= 2:
            first_act = activations[layer_names[0]]
            last_act = activations[layer_names[-1]]
            if first_act.dim() == 4:
                first_act = first_act.flatten(2).permute(0, 2, 1)
            if last_act.dim() == 4:
                last_act = last_act.flatten(2).permute(0, 2, 1)
            if position < first_act.shape[1] and position < last_act.shape[1]:
                n_features = min(64, first_act.shape[-1], last_act.shape[-1])
                first_vec = first_act[0, position, :n_features].float().cpu().numpy()
                last_vec = last_act[0, position, :n_features].float().cpu().numpy()
                x = np.arange(n_features)
                axes[1].bar(
                    x - 0.2, first_vec, width=0.4, label=layer_names[0], alpha=0.7
                )
                axes[1].bar(
                    x + 0.2, last_vec, width=0.4, label=layer_names[-1], alpha=0.7
                )
                axes[1].set_title(
                    f"First {n_features} Features Comparison — Position {position}"
                )
                axes[1].set_xlabel("Feature index")
                axes[1].legend()

        plt.tight_layout()
        return fig
