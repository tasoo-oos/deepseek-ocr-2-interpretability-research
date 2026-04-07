# Mechanistic Interpretability Test Guide

## Overview

The mechanistic interpretability test suite is split into two categories:

- Lightweight dry-run tests that use synthetic tensors, fake models, or tiny linear modules
- Heavy integration work that runs separately against the real DeepSeek-OCR-2 checkpoint on GPU

The lightweight tests are designed for slow hardware. They do not run full OCR inference, do not require downloading pretrained weights, and do not execute the full vision-language stack.

## Dry-Run Coverage

These tests validate the analysis code paths and keep each interpretability target isolated from raw model execution.

### Projector bottleneck

File: `tests/test_projector_analysis.py`

- SVD decomposition of the `896 -> 1280` projector
- Effective-rank calculation from the singular value spectrum
- Cross-modal logit-lens style decoding through a synthetic unembedding matrix

Implementation: `src/analysis/projector_analysis.py`

### Learned query specialization

File: `tests/test_query_analysis.py`

- Query-bank norm and cosine-similarity summaries
- Cross-resolution similarity between `query_768` and `query_1024`
- Query-group ablation scoring without running the full D2E stack

Implementation: `src/analysis/query_analysis.py`

### Global vs local view ablation

File: `tests/test_view_analysis.py`

- Local-view zero ablation bookkeeping
- Global-view zero ablation bookkeeping
- Verification that ablations do not mutate the caller's input tensors

Implementation: `src/analysis/view_analysis.py`

### Spatial probing

File: `tests/test_spatial_analysis.py`

- Closed-form linear probe fitting
- Coordinate prediction with mean-squared-error and `R^2` metrics
- Pre-fit error handling

Implementation: `src/analysis/spatial_analysis.py`

### Circuit discovery

File: `tests/test_circuits.py`

- Single-position activation patching
- Ranking causally important positions with fake feature and intervention managers

Implementation: existing `src/analysis/circuits.py`

## Running the Dry-Run Suite

Use `unittest` directly:

```bash
.venv/bin/python -m unittest \
  tests.test_projector_analysis \
  tests.test_query_analysis \
  tests.test_view_analysis \
  tests.test_spatial_analysis \
  tests.test_circuits -v
```

This suite is intended to finish quickly on CPU-only or low-end hardware.

## GPU Integration Coverage

The repo now includes an end-to-end interpretability integration suite in `tests/test_interpretability_e2e.py`.

It covers:

- `AttentionAnalyzer.analyze_head_specialization()` and `find_important_heads()`
- visualization helpers and report generation in `src/visualization/`
- `scripts/extract_attention.py --synthetic`
- real-GPU CLI runs for:
  - `scripts/extract_attention.py`
  - `scripts/extract_features.py`
  - `scripts/run_interventions.py`

Run it with:

```bash
RUN_GPU_TESTS=1 .venv/bin/python -m pytest tests/test_interpretability_e2e.py -v
```

Or as part of the full suite:

```bash
RUN_GPU_TESTS=1 .venv/bin/python -m pytest tests/ -v
```

## Why This Separation Exists

- Full DeepSeek-OCR-2 inference is expensive on slow hardware
- Mechanistic tooling should be testable without a live checkpoint
- Small dry-run tests make it easier to debug analysis logic independently from model execution

## Remaining Integration Work

The major extraction paths now have real-model coverage. Remaining optional follow-ups include:

- animation-path coverage for `AttentionVisualizer._create_animation()`
- CLI failure-path tests for invalid arguments and missing files
- deeper real-example causal patching / view-ablation experiments beyond smoke coverage

These heavier checks should stay separate from the dry-run suite so contributors can still validate the interpretability codebase on modest machines.
