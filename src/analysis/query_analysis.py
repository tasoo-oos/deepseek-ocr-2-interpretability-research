"""Learned-query analysis helpers for the D2E visual encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Union

import torch
import torch.nn.functional as F


QueryResolution = Union[int, str]


@dataclass(frozen=True)
class QueryBankSummary:
    """Geometry summary for a learned query bank."""

    embeddings: torch.Tensor
    norms: torch.Tensor
    cosine_similarity: torch.Tensor
    mean_abs_cosine: float


class QuerySpecializationAnalyzer:
    """Inspect learned query banks and score query-group ablations."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def get_query_bank(self, resolution: QueryResolution) -> torch.Tensor:
        """Return the query embedding matrix for a given resolution."""
        key = str(resolution).lower()
        if resolution in (768, 144) or key in {"768", "144", "local"}:
            return self.model.query_768.weight.detach()
        if resolution in (1024, 256) or key in {"1024", "256", "global"}:
            return self.model.query_1024.weight.detach()
        raise ValueError(f"Unsupported query resolution: {resolution!r}")

    def summarize_query_bank(self, resolution: QueryResolution) -> QueryBankSummary:
        """Return norms and pairwise cosine similarity for one query bank."""
        embeddings = self.get_query_bank(resolution)
        normalized = F.normalize(embeddings, dim=-1)
        cosine_similarity = normalized @ normalized.transpose(0, 1)
        eye = torch.eye(cosine_similarity.shape[0], dtype=torch.bool, device=cosine_similarity.device)
        off_diagonal = cosine_similarity.masked_select(~eye)
        mean_abs_cosine = float(off_diagonal.abs().mean().item()) if off_diagonal.numel() else 0.0
        return QueryBankSummary(
            embeddings=embeddings,
            norms=embeddings.norm(dim=-1),
            cosine_similarity=cosine_similarity,
            mean_abs_cosine=mean_abs_cosine,
        )

    def cross_resolution_similarity(self) -> torch.Tensor:
        """Return cosine similarity between the 768-query and 1024-query banks."""
        query_768 = F.normalize(self.get_query_bank(768), dim=-1)
        query_1024 = F.normalize(self.get_query_bank(1024), dim=-1)
        return query_768 @ query_1024.transpose(0, 1)

    def measure_query_group_contributions(
        self,
        model_inputs: Any,
        query_groups: Mapping[str, Sequence[int]],
        score_fn,
    ) -> Dict[str, Any]:
        """Ablate query groups and report the score drop for each group."""
        baseline_output = self._run_model(model_inputs)
        baseline_score = float(score_fn(baseline_output))
        deltas: Dict[str, float] = {}

        for name, indices in query_groups.items():
            handle = self.model.register_forward_hook(self._make_query_ablation_hook(indices))
            try:
                ablated_output = self._run_model(model_inputs)
            finally:
                handle.remove()
            deltas[name] = baseline_score - float(score_fn(ablated_output))

        return {"baseline": baseline_score, "deltas": deltas}

    def _run_model(self, model_inputs: Any):
        if isinstance(model_inputs, dict):
            return self.model(**model_inputs)
        if isinstance(model_inputs, (tuple, list)):
            return self.model(*model_inputs)
        return self.model(model_inputs)

    @staticmethod
    def _make_query_ablation_hook(indices: Sequence[int]):
        index_tensor = torch.as_tensor(list(indices), dtype=torch.long)

        def hook(module, inputs, output):
            query_output = output[0] if isinstance(output, tuple) else output
            ablated = query_output.clone()
            if index_tensor.numel():
                ablated.index_fill_(1, index_tensor.to(ablated.device), 0.0)
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            return ablated

        return hook
