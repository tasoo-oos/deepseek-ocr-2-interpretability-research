"""Tests for learned-query specialization utilities."""

import unittest

import torch
import torch.nn as nn

from src.analysis.query_analysis import QuerySpecializationAnalyzer


class FakeQueryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_768 = nn.Embedding(3, 2)
        self.query_1024 = nn.Embedding(4, 2)
        with torch.no_grad():
            self.query_768.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
            self.query_1024.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_bank = self.query_768.weight[: x.shape[1]].unsqueeze(0)
        return x + query_bank


class QueryAnalysisTests(unittest.TestCase):
    def test_query_bank_summary_reports_geometry(self):
        analyzer = QuerySpecializationAnalyzer(FakeQueryModel())

        summary = analyzer.summarize_query_bank(768)

        self.assertEqual(summary.embeddings.shape, (3, 2))
        self.assertEqual(summary.norms.shape, (3,))
        self.assertEqual(summary.cosine_similarity.shape, (3, 3))
        self.assertTrue(torch.allclose(torch.diag(summary.cosine_similarity), torch.ones(3)))
        self.assertGreaterEqual(summary.mean_abs_cosine, 0.0)

    def test_cross_resolution_similarity_matches_query_dimensions(self):
        analyzer = QuerySpecializationAnalyzer(FakeQueryModel())

        similarity = analyzer.cross_resolution_similarity()

        self.assertEqual(similarity.shape, (3, 4))
        self.assertEqual(similarity[0, 0].item(), 1.0)

    def test_query_group_contributions_identify_salient_queries(self):
        analyzer = QuerySpecializationAnalyzer(FakeQueryModel())
        inputs = torch.zeros(1, 3, 2)

        result = analyzer.measure_query_group_contributions(
            inputs,
            query_groups={"dominant": [0], "weak": [2]},
            score_fn=lambda output: output[..., 0].sum().item(),
        )

        self.assertGreater(result["baseline"], 0.0)
        self.assertGreater(result["deltas"]["dominant"], result["deltas"]["weak"])


if __name__ == "__main__":
    unittest.main()
