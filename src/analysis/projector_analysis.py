"""Projector-specific analysis helpers for the visual bottleneck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ProjectorSVD:
    """Singular value decomposition summary for the projector weight."""

    singular_values: torch.Tensor
    left_singular_vectors: torch.Tensor
    right_singular_vectors: torch.Tensor
    explained_variance_ratio: torch.Tensor


class ProjectorAnalyzer:
    """Analyze the linear visual-to-language projector."""

    def __init__(self, projector: nn.Module):
        self.projector = projector
        self.linear = self._resolve_linear_layer(projector)

    @staticmethod
    def _resolve_linear_layer(projector: nn.Module) -> nn.Linear:
        if isinstance(projector, nn.Linear):
            return projector
        layers = getattr(projector, "layers", None)
        if isinstance(layers, nn.Linear):
            return layers
        raise TypeError(
            "ProjectorAnalyzer requires a linear projector or a module exposing "
            "a linear `layers` attribute."
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.detach()

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.linear.bias is None:
            return None
        return self.linear.bias.detach()

    def compute_svd(self) -> ProjectorSVD:
        """Return the projector SVD and variance explained by each component."""
        left, singular_values, right_h = torch.linalg.svd(self.weight, full_matrices=False)
        variance = singular_values.square()
        explained_variance_ratio = variance / variance.sum().clamp_min(1e-12)
        return ProjectorSVD(
            singular_values=singular_values,
            left_singular_vectors=left,
            right_singular_vectors=right_h.transpose(-2, -1),
            explained_variance_ratio=explained_variance_ratio,
        )

    def effective_rank(self, energy_threshold: float = 0.99) -> int:
        """Return the number of singular directions needed to reach a target energy."""
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1].")

        svd = self.compute_svd()
        cumulative = torch.cumsum(svd.explained_variance_ratio, dim=0)
        rank = torch.searchsorted(cumulative, torch.tensor(energy_threshold, device=cumulative.device))
        return int(rank.item()) + 1

    def project(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features into the language-model embedding space."""
        return self.projector(visual_features)

    def logit_lens(
        self,
        visual_features: torch.Tensor,
        unembedding_weight: torch.Tensor,
        unembedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode projected visual features with a language-model unembedding matrix."""
        projected = self.project(visual_features)
        logits = projected @ unembedding_weight.transpose(-2, -1)
        if unembedding_bias is not None:
            logits = logits + unembedding_bias
        return logits
