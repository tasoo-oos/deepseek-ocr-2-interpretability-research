"""Tests for OmniDocBench dataset loading and bulk export."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.benchmarks.omnidocbench import OmniDocBenchDataset, OmniDocBenchRunner


def make_entry(image_path: str, *, page_no: int, language: str) -> dict:
    return {
        "layout_dets": [],
        "page_info": {
            "page_no": page_no,
            "height": 1500,
            "width": 1000,
            "image_path": image_path,
            "page_attribute": {
                "language": language,
                "layout": "single_column",
            },
        },
        "extra": {"relation": []},
    }


class OmniDocBenchTests(unittest.TestCase):
    def test_dataset_loader_reads_manifest_and_filters_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(images_dir / "page_1.jpg")
            Image.new("RGB", (10, 10), color=(0, 0, 0)).save(images_dir / "page_2.jpg")

            manifest_path = root / "OmniDocBench.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        make_entry("images/page_1.jpg", page_no=1, language="english"),
                        make_entry("images/page_2.jpg", page_no=2, language="simplified_chinese"),
                    ]
                ),
                encoding="utf-8",
            )

            dataset = OmniDocBenchDataset.from_dataset_root(root, filters={"language": "english"})

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].image_name, "page_1.jpg")
            self.assertEqual(dataset[0].output_name, "page_1.md")

    def test_bulk_runner_writes_markdown_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            Image.new("RGB", (12, 8), color=(128, 128, 128)).save(images_dir / "page_1.jpg")

            manifest_path = root / "OmniDocBench.json"
            manifest_path.write_text(
                json.dumps([make_entry("images/page_1.jpg", page_no=1, language="english")]),
                encoding="utf-8",
            )

            dataset = OmniDocBenchDataset.from_dataset_root(root)
            runner = OmniDocBenchRunner(dataset)
            output_dir = root / "predictions"

            records = runner.run(
                lambda image, prompt=None: f"size={image.size};prompt={prompt}",
                output_dir,
                prompt="<image>\nFree OCR.",
            )

            self.assertEqual(len(records), 1)
            self.assertTrue((output_dir / "page_1.md").exists())
            self.assertIn("size=(12, 8)", (output_dir / "page_1.md").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "run_manifest.jsonl").exists())

    def test_dry_run_only_plans_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            Image.new("RGB", (12, 8), color=(128, 128, 128)).save(images_dir / "page_1.jpg")

            manifest_path = root / "OmniDocBench.json"
            manifest_path.write_text(
                json.dumps([make_entry("images/page_1.jpg", page_no=1, language="english")]),
                encoding="utf-8",
            )

            dataset = OmniDocBenchDataset.from_dataset_root(root)
            runner = OmniDocBenchRunner(dataset)
            output_dir = root / "predictions"

            records = runner.run(
                lambda image, prompt=None: "should not run",
                output_dir,
                prompt="<image>\nFree OCR.",
                dry_run=True,
            )

            self.assertEqual(records[0]["status"], "planned")
            self.assertFalse((output_dir / "page_1.md").exists())
            self.assertTrue((output_dir / "run_manifest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
