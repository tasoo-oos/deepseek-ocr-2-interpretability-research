"""View-level ablation utilities for global/local visual inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class ViewAblationResult:
    """Scores for baseline, local-view ablation, and global-view ablation."""

    baseline_score: float
    local_ablated_score: float
    global_ablated_score: float
    local_delta: float
    global_delta: float


class ViewAblationAnalyzer:
    """Compare how much local and global views contribute to a score."""

    def __init__(self, model: Any):
        self.model = model

    def compare(self, model_inputs: Mapping[str, torch.Tensor], score_fn=None) -> ViewAblationResult:
        """Run baseline, local-zeroed, and global-zeroed view comparisons."""
        score_fn = score_fn or self.default_score

        baseline_output = self._forward(model_inputs)
        baseline_score = float(score_fn(baseline_output))

        local_output = self._forward(self.ablate_local_views(model_inputs))
        local_score = float(score_fn(local_output))

        global_output = self._forward(self.ablate_global_views(model_inputs))
        global_score = float(score_fn(global_output))

        return ViewAblationResult(
            baseline_score=baseline_score,
            local_ablated_score=local_score,
            global_ablated_score=global_score,
            local_delta=baseline_score - local_score,
            global_delta=baseline_score - global_score,
        )

    @staticmethod
    def clone_inputs(model_inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clone tensor inputs so ablations do not mutate the caller's data."""
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }

    @classmethod
    def ablate_local_views(cls, model_inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ablated = cls.clone_inputs(model_inputs)
        ablated["images_crop"] = torch.zeros_like(ablated["images_crop"])
        return ablated

    @classmethod
    def ablate_global_views(cls, model_inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ablated = cls.clone_inputs(model_inputs)
        ablated["pixel_values"] = torch.zeros_like(ablated["pixel_values"])
        return ablated

    def _forward(self, model_inputs: Mapping[str, torch.Tensor]):
        if hasattr(self.model, "get_multimodal_embeddings"):
            return self.model.get_multimodal_embeddings(
                model_inputs["pixel_values"],
                model_inputs["images_crop"],
                model_inputs["images_spatial_crop"],
            )
        return self.model(**model_inputs)

    @classmethod
    def default_score(cls, output: Any) -> float:
        """Default scalar score based on tensor energy in the output."""
        return float(cls._score_value(output).item())

    @classmethod
    def _score_value(cls, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.float().norm()
        if isinstance(value, dict):
            parts = [cls._score_value(item) for item in value.values()]
            return sum(parts, start=torch.tensor(0.0))
        if isinstance(value, (list, tuple)):
            parts = [cls._score_value(item) for item in value]
            return sum(parts, start=torch.tensor(0.0))
        raise TypeError(f"Unsupported output type for scoring: {type(value)!r}")
