"""Linear probes for measuring linearly decodable spatial information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SpatialProbeMetrics:
    """Basic metrics for a fitted spatial probe."""

    mse: float
    r2: torch.Tensor


class LinearSpatialProbe:
    """Closed-form ridge probe for coordinate prediction."""

    def __init__(self, l2_penalty: float = 1e-4):
        self.l2_penalty = l2_penalty
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> "LinearSpatialProbe":
        """Fit a linear map from features to coordinates."""
        x = features.float()
        y = targets.float()
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        design = torch.cat([x, ones], dim=-1)

        gram = design.transpose(0, 1) @ design
        regularizer = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * self.l2_penalty
        regularizer[-1, -1] = 0.0
        params = torch.linalg.solve(gram + regularizer, design.transpose(0, 1) @ y)

        self.weight = params[:-1, :]
        self.bias = params[-1, :]
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict coordinates from features."""
        self._check_is_fitted()
        return features.float() @ self.weight + self.bias

    def mean_squared_error(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute mean squared error."""
        predictions = self.predict(features)
        return float(torch.mean((predictions - targets.float()) ** 2).item())

    def r2_score(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-target-dimension R^2."""
        y_true = targets.float()
        y_pred = self.predict(features)
        residual = ((y_true - y_pred) ** 2).sum(dim=0)
        total = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
        return 1.0 - residual / total

    def evaluate(self, features: torch.Tensor, targets: torch.Tensor) -> SpatialProbeMetrics:
        """Return MSE and per-dimension R^2 for a fitted probe."""
        return SpatialProbeMetrics(
            mse=self.mean_squared_error(features, targets),
            r2=self.r2_score(features, targets),
        )

    def _check_is_fitted(self) -> None:
        if self.weight is None or self.bias is None:
            raise RuntimeError("LinearSpatialProbe must be fitted before prediction.")
