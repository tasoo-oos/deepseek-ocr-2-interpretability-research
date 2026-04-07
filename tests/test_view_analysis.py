"""Tests for global-vs-local view ablation analysis."""

import unittest

import torch

from src.analysis.view_analysis import ViewAblationAnalyzer


class FakeVisionModel:
    def get_multimodal_embeddings(self, pixel_values, images_crop, images_spatial_crop):
        local_signal = images_crop.float().sum().view(1, 1)
        global_signal = pixel_values.float().sum().view(1, 1)
        return [torch.cat([local_signal, global_signal], dim=-1)]


def build_view_inputs():
    return {
        "pixel_values": torch.ones(1, 1, 3, 2, 2),
        "images_crop": torch.full((1, 1, 2, 3, 2, 2), 2.0),
        "images_spatial_crop": torch.tensor([[[1, 2]]]),
    }


class ViewAnalysisTests(unittest.TestCase):
    def test_view_ablation_separates_local_and_global_contributions(self):
        view_inputs = build_view_inputs()
        analyzer = ViewAblationAnalyzer(FakeVisionModel())

        local_result = analyzer.compare(
            view_inputs,
            score_fn=lambda output: output[0][..., 0].sum().item(),
        )
        global_result = analyzer.compare(
            view_inputs,
            score_fn=lambda output: output[0][..., 1].sum().item(),
        )

        self.assertGreater(local_result.local_delta, 0.0)
        self.assertAlmostEqual(local_result.global_delta, 0.0)
        self.assertGreater(global_result.global_delta, 0.0)
        self.assertAlmostEqual(global_result.local_delta, 0.0)

    def test_view_ablation_does_not_mutate_original_inputs(self):
        view_inputs = build_view_inputs()
        original_pixel_values = view_inputs["pixel_values"].clone()
        original_images_crop = view_inputs["images_crop"].clone()

        ViewAblationAnalyzer.ablate_local_views(view_inputs)
        ViewAblationAnalyzer.ablate_global_views(view_inputs)

        self.assertTrue(torch.equal(view_inputs["pixel_values"], original_pixel_values))
        self.assertTrue(torch.equal(view_inputs["images_crop"], original_images_crop))


if __name__ == "__main__":
    unittest.main()
