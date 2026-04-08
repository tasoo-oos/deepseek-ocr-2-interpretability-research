"""Tests for sparse autoencoder utilities."""

import unittest

import torch

from src.analysis.sparse_autoencoder import (
    SparseAutoencoder,
    SparseAutoencoderAnalyzer,
    SparseAutoencoderTrainer,
    ablate_sparse_features,
)


class SparseAutoencoderTests(unittest.TestCase):
    def test_training_reduces_reconstruction_error(self):
        torch.manual_seed(0)
        basis = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.3],
            ]
        )
        codes = torch.zeros(256, 3)
        codes[:, 0] = torch.rand(256) > 0.6
        codes[:, 1] = torch.rand(256) > 0.75
        codes[:, 2] = torch.rand(256) > 0.8
        activations = codes @ basis

        sae = SparseAutoencoder(input_dim=4, n_features=6)
        trainer = SparseAutoencoderTrainer(
            sae,
            lr=1e-2,
            l1_coeff=1e-3,
            batch_size=64,
            steps=120,
            device="cpu",
            seed=0,
        )

        before = trainer.evaluate(activations)
        after = trainer.fit(activations)

        self.assertLess(after.mse, before.mse)
        self.assertGreater(after.explained_variance, before.explained_variance)

    def test_feature_summary_uses_metadata(self):
        torch.manual_seed(0)
        activations = torch.randn(32, 4)
        sae = SparseAutoencoder(input_dim=4, n_features=5)
        trainer = SparseAutoencoderTrainer(
            sae,
            lr=5e-3,
            l1_coeff=1e-3,
            batch_size=16,
            steps=80,
            device="cpu",
            seed=0,
        )
        trainer.fit(activations)

        analyzer = SparseAutoencoderAnalyzer(sae)
        summary = analyzer.summarize(
            activations,
            metadata={
                "query_index": torch.arange(32),
                "attention_x": torch.linspace(0.0, 1.0, 32),
                "attention_y": torch.linspace(1.0, 0.0, 32),
                "stimulus": [f"doc_{idx % 2}" for idx in range(32)],
            },
            min_activation_frequency=0.0,
        )

        self.assertEqual(len(summary.feature_summaries), 5)
        self.assertGreaterEqual(summary.mean_active_features_per_sample, 0.0)
        self.assertGreaterEqual(summary.dead_feature_fraction, 0.0)
        live_feature = next(item for item in summary.feature_summaries if item.activation_frequency > 0.0)
        self.assertIsNotNone(live_feature.weighted_query_index_mean)
        self.assertIsNotNone(live_feature.weighted_attention_x_mean)
        self.assertTrue(all("sample_index" in row for row in live_feature.top_examples))

    def test_ablate_sparse_features_removes_selected_decoder_contribution(self):
        sae = SparseAutoencoder(input_dim=2, n_features=2)
        with torch.no_grad():
            sae.encoder.weight.copy_(torch.eye(2))
            sae.encoder.bias.zero_()
            sae.decoder.weight.copy_(torch.eye(2))

        activations = torch.tensor([[3.0, 2.0], [1.5, 0.5]])

        ablated = ablate_sparse_features(activations, sae, [0], mode="subtract_decoder")

        expected = torch.tensor([[0.0, 2.0], [0.0, 0.5]])
        self.assertTrue(torch.allclose(ablated, expected, atol=1e-6))

    def test_topk_encoder_limits_number_of_active_features(self):
        sae = SparseAutoencoder(input_dim=3, n_features=5, activation_mode="topk", top_k=2)
        with torch.no_grad():
            sae.encoder.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ]
                )
            )
            sae.encoder.bias.zero_()

        x = torch.tensor([[1.0, 2.0, 3.0]])
        codes = sae.encode(x)

        self.assertEqual(int((codes > 0).sum().item()), 2)


if __name__ == "__main__":
    unittest.main()
