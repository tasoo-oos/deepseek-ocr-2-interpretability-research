"""Tests for projector bottleneck analysis utilities."""

import unittest
from typing import cast

import torch
from addict import Dict
from torch import nn

from src.analysis.projector_analysis import ProjectorAnalyzer
from src.models.projector import MlpProjector


def build_linear_projector(weight: torch.Tensor, bias: torch.Tensor | None = None) -> MlpProjector:
    projector = MlpProjector(
        Dict(projector_type="linear", input_dim=weight.shape[1], n_embed=weight.shape[0])
    )
    linear = cast(nn.Linear, projector.layers)
    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is None:
            assert linear.bias is not None
            linear.bias.zero_()
        else:
            assert linear.bias is not None
            linear.bias.copy_(bias)
    return projector


class ProjectorAnalysisTests(unittest.TestCase):
    def test_projector_svd_recovers_singular_values(self):
        weight = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
        analyzer = ProjectorAnalyzer(build_linear_projector(weight))

        svd = analyzer.compute_svd()

        self.assertTrue(torch.allclose(svd.singular_values, torch.tensor([5.0, 3.0, 1.0])))
        self.assertEqual(svd.left_singular_vectors.shape, (3, 3))
        self.assertEqual(svd.right_singular_vectors.shape, (3, 3))
        self.assertTrue(torch.isclose(svd.explained_variance_ratio.sum(), torch.tensor(1.0)))

    def test_projector_effective_rank_uses_energy_threshold(self):
        weight = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
        analyzer = ProjectorAnalyzer(build_linear_projector(weight))

        self.assertEqual(analyzer.effective_rank(0.95), 2)
        self.assertEqual(analyzer.effective_rank(0.999), 3)

    def test_cross_modal_logit_lens_decodes_projected_features(self):
        weight = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        bias = torch.tensor([0.5, -0.5])
        analyzer = ProjectorAnalyzer(build_linear_projector(weight, bias=bias))

        visual_features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        unembedding = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
            ]
        )

        logits = analyzer.logit_lens(visual_features, unembedding)

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(logits[0].argmax().item(), 0)
        self.assertEqual(logits[1].argmax().item(), 1)


if __name__ == "__main__":
    unittest.main()
