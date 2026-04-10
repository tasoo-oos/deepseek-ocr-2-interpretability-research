# DeepSeek-OCR-2 Code Structure

This document describes the current repository layout and how the main research
components fit together. It reflects the code that exists in this repo today,
not the older upstream project layout.

## Directory Layout

```text
deepseek-ocr-2/
├── src/
│   ├── analysis/          # Interpretability tooling
│   ├── benchmarks/        # OmniDocBench helpers
│   ├── inference/         # End-to-end OCR pipeline
│   ├── models/            # SAM, D2E, projector, full vision stack
│   ├── preprocessing/     # Global-view + crop preprocessing
│   ├── visualization/     # Attention and feature plotting
│   └── config.py          # Shared constants
├── scripts/               # CLI entry points for experiments and utilities
├── tests/                 # CPU-friendly tests + optional GPU checks
├── docs/                  # Architecture, API, and research notes
├── input/                 # Dataset notes and local inputs
├── output/                # Saved experiment artifacts and reports
├── README.md
├── pyproject.toml
└── requirements.txt
```

## End-To-End Flow

```text
input image
    ->
ImageProcessor
    - global padded view: 1024 x 1024
    - local dynamic crops: N x 768 x 768
    ->
SAM encoder
    - ViT-B backbone
    - neck + two strided convs
    - output: [B, 896, H, W]
    ->
Qwen2 Decoder-as-Encoder
    - flatten image tokens
    - append learned query bank
    - apply Visual Causal Flow mask
    - keep query half of the sequence
    ->
linear projector
    - 896 -> 1280
    ->
concat [local, global, view_seperator]
    ->
optional language-model merge at <image> token positions
```

## Core Modules

### `src/models/`

- `sam_encoder.py`
  - Reimplementation of the SAM ViT-B image encoder with a `256 -> 512 -> 896`
    downsampling neck.
- `qwen2_d2e.py`
  - Qwen2-based Decoder-as-Encoder with the custom Visual Causal Flow mask.
- `projector.py`
  - Default linear projector from 896-dim visual states to 1280-dim language
    embeddings.
- `deepseek_ocr.py`
  - Full research model wrapper exposing the vision path and optional language
    model integration.

### `src/preprocessing/`

- `dynamic_cropping.py`
  - Chooses the crop grid from aspect ratio and crop-count constraints.
- `image_transforms.py`
  - Builds the global padded view and local crop tensors expected by the model.

### `src/analysis/`

- `attention_analysis.py`
  - Extracts D2E attentions and computes per-head specialization metrics.
- `feature_extractor.py`
  - Registers forward hooks on SAM blocks, D2E layers, and projector output.
- `interventions.py`
  - Head ablations, token-state ablations, activation patching, and SAE feature
    ablations.
- `circuits.py`
  - Activation-patching search over `(layer, position)` candidates.
- `projector_analysis.py`
  - SVD and logit-lens style analysis for the projector bottleneck.
- `query_analysis.py`
  - Query-bank geometry and query-group ablation utilities.
- `spatial_analysis.py`
  - Closed-form ridge probe for spatial coordinate decoding.
- `view_analysis.py`
  - Local-vs-global view ablation helpers.
- `sparse_autoencoder.py`
  - SAE training, summarization, and sparse-feature ablation support.

### `src/visualization/`

- `attention_viz.py`
  - Attention mask, query-to-image, causal-flow, and report-generation plots.
- `feature_viz.py`
  - SAM feature maps, D2E hidden-state plots, and projector visualizations.
- `utils.py`
  - Shared helpers for entropy, spatial reshaping, mask visualization, and
    image overlays.

### `src/inference/`

- `pipeline.py`
  - Loads the upstream Hugging Face model with `trust_remote_code=True` and
    exposes a cleaned OCR interface.
- `batch_inference.py`
  - Repeated pipeline execution across many images.

### `src/benchmarks/`

- `omnidocbench.py`
  - Dataset-aware loader used by the bulk benchmark runner.

## Important Structural Details

### Visual Causal Flow lives inside Qwen2 masking

The D2E module subclasses `Qwen2Model` and overrides the causal-mask update to
build a full 4-D mask from `token_type_ids`.

### Query banks are resolution-specific

- `query_768`: 144 queries
- `query_1024`: 256 queries

The query count always matches the number of image tokens for that resolution.

### Local and global views share weights

The same SAM, D2E, and projector modules process both view types. They are only
combined after projection.

### The separator token is appended last

The multimodal embedding sequence is:

- local projected tokens
- global projected tokens
- one learned `view_seperator` parameter

## Scripts

The main CLIs are:

- `scripts/extract_attention.py`
- `scripts/extract_features.py`
- `scripts/run_interventions.py`
- `scripts/train_sae.py`
- `scripts/run_sae_feature_ablation.py`
- `scripts/research_causal_tokens.py`
- `scripts/run_omnidocbench.py`
- `scripts/check_omnidocbench_outputs.py`
- `scripts/simple_inference.py`

## Outputs

The repo already contains experiment artifacts under `output/`, including:

- attention reports
- causal-token research summaries
- SAE summaries and ablation summaries

Those saved results are referenced by `docs/RESEARCH_AUDIT.md` and
`docs/SPARSE_AUTO_ENCODER.md`.
