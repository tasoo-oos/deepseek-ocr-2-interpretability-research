#!/usr/bin/env python3
"""Validate and summarise OmniDocBench runner outputs.

This script reads the ``run_manifest.jsonl`` written by
``scripts/run_omnidocbench.py`` (or by ``OmniDocBenchRunner.run``)
and performs basic sanity checks:

* Every "written" entry has a corresponding non-empty ``.md`` file.
* Every "skipped" entry has a pre-existing ``.md`` file.
* Reports per-page statistics (char count, line count).
* Computes aggregate summary statistics across all pages.
* Optionally compares predictions against ground-truth markdown
  using simple text-overlap metrics (BLEU-4, character-level edit
  distance ratio, exact-match rate).

Usage
-----
Validate runner output only::

    python scripts/check_omnidocbench_outputs.py \\
        --output_dir output/omnidocbench

With ground-truth comparison (requires a directory of reference ``.md`` files
whose basenames match the prediction file names)::

    python scripts/check_omnidocbench_outputs.py \\
        --output_dir output/omnidocbench \\
        --gt_dir ground_truth/omnidocbench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class PageReport:
    """Per-page validation result."""

    sample_id: str
    status: str
    output_path: str
    exists: bool = False
    chars: int = 0
    lines: int = 0
    error: Optional[str] = None
    # Optional metrics against ground-truth
    gt_chars: Optional[int] = None
    edit_distance_ratio: Optional[float] = None
    bleu4: Optional[float] = None
    exact_match: Optional[bool] = None


@dataclass
class RunReport:
    """Aggregate report across all pages."""

    total_pages: int = 0
    status_counts: dict = field(default_factory=dict)
    missing_files: int = 0
    empty_files: int = 0
    total_chars: int = 0
    total_lines: int = 0
    avg_chars: float = 0.0
    avg_lines: float = 0.0
    # Ground-truth metrics (populated if --gt_dir given)
    avg_edit_distance_ratio: Optional[float] = None
    avg_bleu4: Optional[float] = None
    exact_match_rate: Optional[float] = None


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance (dynamic-programming)."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = cur
    return prev[-1]


def edit_distance_ratio(pred: str, ref: str) -> float:
    """Normalised edit-distance: 0.0 = identical, 1.0 = completely different."""
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 0.0
    return _edit_distance(pred, ref) / max_len


def simple_bleu4(pred: str, ref: str) -> float:
    """Very simple word-level BLEU-4 approximation (no smoothing)."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if len(pred_tokens) < 4 or len(ref_tokens) < 4:
        return 0.0

    from collections import Counter
    import math

    score = 0.0
    for n in range(1, 5):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(min(c, ref_ngrams[ng]) for ng, c in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)
        precision = clipped / total
        if precision == 0:
            return 0.0
        score += math.log(precision)

    # brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(score / 4)


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------


def validate_run(
    output_dir: Path,
    gt_dir: Optional[Path] = None,
) -> tuple[RunReport, list[PageReport]]:
    """Read ``run_manifest.jsonl`` and validate every entry."""
    manifest_path = output_dir / "run_manifest.jsonl"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    pages: list[PageReport] = []
    status_counts: Counter = Counter()

    for line in manifest_path.read_text(encoding="utf-8").strip().splitlines():
        record = json.loads(line)
        sample_id = record["sample_id"]
        status = record["status"]
        out_path = Path(record["output_path"])
        status_counts[status] += 1

        report = PageReport(
            sample_id=sample_id, status=status, output_path=str(out_path)
        )

        if not out_path.exists():
            if status in ("written", "skipped"):
                report.exists = False
                report.error = "output file missing"
            else:
                report.exists = False
        else:
            report.exists = True
            text = out_path.read_text(encoding="utf-8")
            report.chars = len(text)
            report.lines = len(text.splitlines())
            if report.chars == 0 and status == "written":
                report.error = "output file is empty"

            # Ground-truth comparison
            if gt_dir is not None:
                gt_path = gt_dir / out_path.name
                if gt_path.exists():
                    gt_text = gt_path.read_text(encoding="utf-8")
                    report.gt_chars = len(gt_text)
                    report.edit_distance_ratio = edit_distance_ratio(text, gt_text)
                    report.bleu4 = simple_bleu4(text, gt_text)
                    report.exact_match = text.strip() == gt_text.strip()

        pages.append(report)

    # Aggregate
    written_pages = [
        p for p in pages if p.exists and p.status in ("written", "skipped")
    ]
    total_chars = sum(p.chars for p in written_pages)
    total_lines = sum(p.lines for p in written_pages)
    n_written = max(len(written_pages), 1)

    run = RunReport(
        total_pages=len(pages),
        status_counts=dict(status_counts),
        missing_files=sum(1 for p in pages if p.error and "missing" in p.error),
        empty_files=sum(1 for p in pages if p.error and "empty" in p.error),
        total_chars=total_chars,
        total_lines=total_lines,
        avg_chars=total_chars / n_written,
        avg_lines=total_lines / n_written,
    )

    # Ground-truth aggregate
    edr_vals = [
        p.edit_distance_ratio for p in pages if p.edit_distance_ratio is not None
    ]
    bleu_vals = [p.bleu4 for p in pages if p.bleu4 is not None]
    em_vals = [p.exact_match for p in pages if p.exact_match is not None]
    if edr_vals:
        run.avg_edit_distance_ratio = sum(edr_vals) / len(edr_vals)
    if bleu_vals:
        run.avg_bleu4 = sum(bleu_vals) / len(bleu_vals)
    if em_vals:
        run.exact_match_rate = sum(em_vals) / len(em_vals)

    return run, pages


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarise OmniDocBench runner outputs."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing run_manifest.jsonl and .md outputs",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=None,
        help="Optional directory of ground-truth .md files for metric comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-page details",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_report, page_reports = validate_run(args.output_dir, gt_dir=args.gt_dir)

    if args.output_json:
        from dataclasses import asdict

        payload = {
            "summary": asdict(run_report),
            "pages": [asdict(p) for p in page_reports] if args.verbose else [],
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    # Human-readable output
    print("=" * 60)
    print("OmniDocBench Output Report")
    print("=" * 60)
    print(f"Total pages:     {run_report.total_pages}")
    for status, count in sorted(run_report.status_counts.items()):
        print(f"  {status:12s}:  {count}")
    print(f"Missing files:   {run_report.missing_files}")
    print(f"Empty files:     {run_report.empty_files}")
    print(f"Avg chars/page:  {run_report.avg_chars:.0f}")
    print(f"Avg lines/page:  {run_report.avg_lines:.1f}")

    if run_report.avg_edit_distance_ratio is not None:
        print("-" * 60)
        print("Ground-truth comparison:")
        print(f"  Avg edit-dist ratio: {run_report.avg_edit_distance_ratio:.4f}")
        print(f"  Avg BLEU-4:          {run_report.avg_bleu4:.4f}")
        print(f"  Exact-match rate:    {run_report.exact_match_rate:.4f}")

    # Errors
    errors = [p for p in page_reports if p.error]
    if errors:
        print("-" * 60)
        print(f"Errors ({len(errors)}):")
        for p in errors[:20]:
            print(f"  [{p.sample_id}] {p.error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    if args.verbose:
        print("-" * 60)
        print("Per-page details:")
        for p in page_reports:
            line = f"  {p.sample_id:40s}  status={p.status:8s}  chars={p.chars:5d}  lines={p.lines:3d}"
            if p.bleu4 is not None:
                line += f"  bleu4={p.bleu4:.4f}"
            if p.edit_distance_ratio is not None:
                line += f"  edr={p.edit_distance_ratio:.4f}"
            print(line)

    print("=" * 60)
    ok = run_report.missing_files == 0 and run_report.empty_files == 0
    print("RESULT: PASS" if ok else "RESULT: ISSUES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
