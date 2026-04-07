"""Tests for linear spatial probing utilities."""

import unittest

import torch

from src.analysis.spatial_analysis import LinearSpatialProbe


class SpatialAnalysisTests(unittest.TestCase):
    def test_linear_spatial_probe_recovers_linearly_decodable_coordinates(self):
        torch.manual_seed(0)
        features = torch.randn(128, 4)
        true_weight = torch.tensor(
            [
                [2.0, -0.5],
                [0.25, 1.5],
                [-1.0, 0.75],
                [0.5, 0.25],
            ]
        )
        true_bias = torch.tensor([0.2, -0.1])
        targets = features @ true_weight + true_bias

        probe = LinearSpatialProbe(l2_penalty=1e-6).fit(features, targets)
        predictions = probe.predict(features)
        metrics = probe.evaluate(features, targets)

        self.assertTrue(torch.allclose(predictions, targets, atol=1e-4))
        self.assertLess(metrics.mse, 1e-8)
        self.assertTrue(torch.all(metrics.r2 > 0.99999))

    def test_linear_spatial_probe_requires_fit_before_prediction(self):
        probe = LinearSpatialProbe()
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            probe.predict(torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
