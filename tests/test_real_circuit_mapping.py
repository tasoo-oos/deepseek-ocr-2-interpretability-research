"""Tests for real-document circuit-mapping helpers."""

import unittest

import torch

from src.analysis.real_circuit_mapping import (
    bbox_to_grid_mask,
    extract_region_targets,
    map_bbox_to_padded_square,
    normalize_label,
    resolve_region_bbox,
    score_query_alignment,
    select_target_queries,
)


class RealCircuitMappingTests(unittest.TestCase):
    def test_resolve_region_bbox_scales_from_page_coordinates(self):
        det = {"category_type": "table", "bbox": [100, 200, 500, 600]}
        bbox = resolve_region_bbox(
            det,
            image_size=(1000, 500),
            page_size=(2000, 1000),
        )
        self.assertEqual(bbox, (50.0, 100.0, 250.0, 300.0))

    def test_extract_region_targets_filters_and_sorts(self):
        entry = {
            "layout_dets": [
                {"category_type": "table", "bbox": [0, 0, 300, 100]},
                {"category_type": "formula", "bbox": [0, 0, 50, 50]},
                {"category_type": "table_cell", "bbox": [0, 0, 200, 200]},
            ]
        }
        targets = extract_region_targets(
            entry,
            image_size=(400, 400),
            region_type="table",
            max_regions=2,
            min_area_ratio=0.0,
        )
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0].label, "table_cell")
        self.assertEqual(targets[1].label, "table")

    def test_map_bbox_to_padded_square_adds_vertical_padding(self):
        mapped = map_bbox_to_padded_square(
            (0.0, 0.0, 1000.0, 500.0),
            image_size=(1000, 500),
            square_size=1024,
        )
        self.assertAlmostEqual(mapped[0], 0.0)
        self.assertAlmostEqual(mapped[2], 1024.0)
        self.assertAlmostEqual(mapped[1], 256.0)
        self.assertAlmostEqual(mapped[3], 768.0)

    def test_bbox_to_grid_mask_marks_overlapping_cells(self):
        mask = bbox_to_grid_mask(
            (0.0, 0.0, 512.0, 512.0),
            square_size=1024,
            spatial_size=4,
        )
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(torch.all(mask[:2, :2] > 0.0))
        self.assertTrue(torch.all(mask[2:, :] == 0.0))

    def test_select_target_queries_prefers_box_focused_query(self):
        attn = torch.zeros(2, 8, 8)
        # 4 image tokens, 4 query tokens.
        # Query 1 focuses fully on image token 2, which is inside the mask.
        attn[:, 4 + 1, 2] = 1.0
        # Query 3 focuses on token 0, outside the mask.
        attn[:, 4 + 3, 0] = 1.0
        box_mask = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        indices, scores = select_target_queries(attn, box_mask=box_mask, top_k=2, n_image_tokens=4)
        self.assertEqual(indices[0], 1)
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_score_query_alignment_averages_cosine_on_selected_queries(self):
        clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        candidate = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        score = score_query_alignment(clean, candidate, [0, 1])
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_normalize_label(self):
        self.assertEqual(normalize_label("Table Cell"), "table_cell")


if __name__ == "__main__":
    unittest.main()
