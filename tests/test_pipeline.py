"""Tests for the inference pipeline and post-processing helpers.

These tests validate the public API without downloading model weights.
GPU / real-model tests are gated behind ``RUN_GPU_TESTS=1``.
"""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.inference.pipeline import (
    DeepseekOCRPipeline,
    _load_upstream_model,
    clean_prediction,
    _clean_formula,
    _strip_ref_det_tags,
)


class PostProcessingTests(unittest.TestCase):
    """Unit tests for the post-processing helpers."""

    def test_clean_formula_strips_quad_labels(self):
        raw = r"Here is \[E = mc^2 \quad (1)\] done."
        result = _clean_formula(raw)
        self.assertNotIn(r"\quad", result)
        self.assertIn(r"\[E = mc^2\]", result)

    def test_clean_formula_leaves_plain_formulas_alone(self):
        raw = r"\[x + y = z\]"
        self.assertEqual(_clean_formula(raw), raw)

    def test_strip_ref_det_tags_removes_annotations(self):
        raw = "<|ref|>text<|/ref|><|det|>[[10,20,30,40]]<|/det|>\nHello world"
        result = _strip_ref_det_tags(raw)
        self.assertNotIn("<|ref|>", result)
        self.assertNotIn("<|det|>", result)
        self.assertIn("Hello world", result)

    def test_strip_ref_det_tags_collapses_blank_lines(self):
        raw = "A\n\n\n\nB\n\n\nC"
        result = _strip_ref_det_tags(raw)
        self.assertNotIn("\n\n\n", result)

    def test_clean_prediction_combines_both(self):
        raw = (
            "<|ref|>sub_title<|/ref|><|det|>[[0,0,100,100]]<|/det|>\n"
            r"## Title"
            "\n\n"
            r"\[f(x) \quad (eq1)\]"
        )
        result = clean_prediction(raw)
        self.assertNotIn("<|ref|>", result)
        self.assertNotIn(r"\quad", result)
        self.assertIn("## Title", result)
        self.assertIn(r"\[f(x)\]", result)


class PipelineConstructionTests(unittest.TestCase):
    """Tests for the pipeline constructor and helper loading logic."""

    def test_pipeline_constructor_stores_attributes(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        pipeline = DeepseekOCRPipeline(
            mock_model,
            mock_tokenizer,
            device="cpu",
            prompt="test prompt",
        )
        self.assertEqual(pipeline.device, "cpu")
        self.assertEqual(pipeline.prompt, "test prompt")
        self.assertIs(pipeline.model, mock_model)
        self.assertIs(pipeline.tokenizer, mock_tokenizer)

    def test_pipeline_call_with_file_path(self):
        """Pipeline should accept a file path and call model.infer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # model.infer writes result.mmd into the output dir
        def fake_infer(tokenizer, *, prompt, image_file, output_path, **kwargs):
            Path(output_path).mkdir(parents=True, exist_ok=True)
            (Path(output_path) / "result.mmd").write_text(
                "## Hello\nWorld", encoding="utf-8"
            )

        mock_model.infer = fake_infer

        pipeline = DeepseekOCRPipeline(mock_model, mock_tokenizer, device="cpu")
        result = pipeline("/tmp/fake.jpg")
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_pipeline_call_with_pil_image(self):
        """Pipeline should accept a PIL Image (writes temp file)."""
        from PIL import Image

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def fake_infer(tokenizer, *, prompt, image_file, output_path, **kwargs):
            # Verify that the temp file actually exists
            assert Path(image_file).exists(), f"temp file {image_file} missing"
            Path(output_path).mkdir(parents=True, exist_ok=True)
            (Path(output_path) / "result.mmd").write_text(
                "## From PIL", encoding="utf-8"
            )

        mock_model.infer = fake_infer

        pipeline = DeepseekOCRPipeline(mock_model, mock_tokenizer, device="cpu")
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        result = pipeline(img)
        self.assertIn("From PIL", result)

    def test_pipeline_raw_mode_skips_cleaning(self):
        """When raw=True, ref/det tags should be preserved."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def fake_infer(tokenizer, *, prompt, image_file, output_path, **kwargs):
            Path(output_path).mkdir(parents=True, exist_ok=True)
            (Path(output_path) / "result.mmd").write_text(
                "<|ref|>text<|/ref|><|det|>[[0,0,1,1]]<|/det|>\nContent",
                encoding="utf-8",
            )

        mock_model.infer = fake_infer

        pipeline = DeepseekOCRPipeline(mock_model, mock_tokenizer, device="cpu")
        raw_result = pipeline("/tmp/fake.jpg", raw=True)
        self.assertIn("<|ref|>", raw_result)

        clean_result = pipeline("/tmp/fake.jpg", raw=False)
        self.assertNotIn("<|ref|>", clean_result)
        self.assertIn("Content", clean_result)


class CheckScriptTests(unittest.TestCase):
    """Tests for the check_omnidocbench_outputs evaluator helpers."""

    def test_validate_run_on_valid_output(self):
        """Validate that the check script correctly parses a manifest."""
        from scripts.check_omnidocbench_outputs import validate_run

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            # Write a fake manifest
            records = [
                {
                    "index": 1,
                    "sample_id": "page_1:1",
                    "image_path": "/fake/page_1.jpg",
                    "output_path": str(out / "page_1.md"),
                    "status": "written",
                    "page_no": 1,
                    "page_attributes": {"language": "english"},
                },
                {
                    "index": 2,
                    "sample_id": "page_2:2",
                    "image_path": "/fake/page_2.jpg",
                    "output_path": str(out / "page_2.md"),
                    "status": "written",
                    "page_no": 2,
                    "page_attributes": {"language": "english"},
                },
            ]
            (out / "run_manifest.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
                encoding="utf-8",
            )
            (out / "page_1.md").write_text(
                "## Title\nSome text here.", encoding="utf-8"
            )
            (out / "page_2.md").write_text("## Other\nMore text.", encoding="utf-8")

            run_report, page_reports = validate_run(out)

            self.assertEqual(run_report.total_pages, 2)
            self.assertEqual(run_report.missing_files, 0)
            self.assertEqual(run_report.empty_files, 0)
            self.assertTrue(all(p.exists for p in page_reports))
            self.assertGreater(run_report.avg_chars, 0)

    def test_validate_run_detects_missing_output(self):
        from scripts.check_omnidocbench_outputs import validate_run

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            records = [
                {
                    "index": 1,
                    "sample_id": "page_1:1",
                    "image_path": "/fake/page_1.jpg",
                    "output_path": str(out / "page_1.md"),
                    "status": "written",
                    "page_no": 1,
                    "page_attributes": {},
                },
            ]
            (out / "run_manifest.jsonl").write_text(
                json.dumps(records[0]) + "\n", encoding="utf-8"
            )
            # Intentionally do NOT write page_1.md

            run_report, page_reports = validate_run(out)
            self.assertEqual(run_report.missing_files, 1)
            self.assertEqual(page_reports[0].error, "output file missing")

    def test_validate_run_with_ground_truth(self):
        from scripts.check_omnidocbench_outputs import validate_run

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pred"
            gt = Path(tmpdir) / "gt"
            out.mkdir()
            gt.mkdir()

            records = [
                {
                    "index": 1,
                    "sample_id": "page_1:1",
                    "image_path": "/fake/page_1.jpg",
                    "output_path": str(out / "page_1.md"),
                    "status": "written",
                    "page_no": 1,
                    "page_attributes": {},
                },
            ]
            (out / "run_manifest.jsonl").write_text(
                json.dumps(records[0]) + "\n", encoding="utf-8"
            )
            (out / "page_1.md").write_text("Hello world", encoding="utf-8")
            (gt / "page_1.md").write_text("Hello world", encoding="utf-8")

            run_report, page_reports = validate_run(out, gt_dir=gt)
            self.assertIsNotNone(run_report.avg_bleu4)
            self.assertIsNotNone(run_report.avg_edit_distance_ratio)
            self.assertEqual(run_report.exact_match_rate, 1.0)
            self.assertEqual(page_reports[0].exact_match, True)
            self.assertAlmostEqual(page_reports[0].edit_distance_ratio, 0.0)


@unittest.skipUnless(
    os.environ.get("RUN_GPU_TESTS") == "1",
    "GPU tests skipped (set RUN_GPU_TESTS=1 to enable)",
)
class GPUPipelineTests(unittest.TestCase):
    """Integration tests that load real model weights on a GPU.

    Only run when ``RUN_GPU_TESTS=1`` is set in the environment.
    """

    def test_load_upstream_model(self):
        """Verify _load_upstream_model succeeds with the real model."""
        import torch

        model = _load_upstream_model(
            "deepseek-ai/DeepSeek-OCR-2",
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.assertTrue(hasattr(model, "infer"))

    def test_pipeline_from_pretrained(self):
        """Verify the full pipeline can be constructed."""
        pipeline = DeepseekOCRPipeline.from_pretrained(
            "deepseek-ai/DeepSeek-OCR-2",
            device="cuda",
        )
        self.assertIsNotNone(pipeline.model)
        self.assertIsNotNone(pipeline.tokenizer)

    def test_pipeline_inference_on_sample(self):
        """Run inference on a tiny synthetic image."""
        from PIL import Image

        pipeline = DeepseekOCRPipeline.from_pretrained(
            "deepseek-ai/DeepSeek-OCR-2",
            device="cuda",
        )
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        result = pipeline(img, max_new_tokens=64)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
