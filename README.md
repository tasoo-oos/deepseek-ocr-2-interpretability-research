# DeepSeek-OCR-2 Mechanistic Interpretability Toolkit

A research toolkit for mechanistic interpretability of [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), a vision-language model that converts document images into structured markdown using a novel **Visual Causal Flow** attention mechanism.

This repo provides clean, dependency-light reimplementations of every vision component (SAM encoder, D2E transformer, projector) alongside purpose-built tools for attention analysis, causal intervention, circuit discovery, spatial probing, and more — all runnable without vLLM.

## Architecture Overview

```
Input Image
    |
    +--- Global View (1024x1024) ---+
    |                               |
    +--- Local Crops (Nx768x768) --+|
                                   ||
                    +--------------+|
                    v               v
             SAM ViT-B Encoder  (768-dim -> 896-dim)
                    |               |
                    v               v
             D2E: Qwen2 Decoder-as-Encoder (896-dim, 24 layers, 14 heads)
             [image tokens: bidirectional | query tokens: causal]
                    |               |
                    v               v
             Linear Projector  (896-dim -> 1280-dim)
                    |               |
                    v               v
             Concat [local, global, view_separator]
                    |
                    v
             DeepSeek Language Model -> Markdown
```

The core innovation is the **Visual Causal Flow** mask inside D2E: image tokens attend bidirectionally to all image tokens but are blocked from attending to query tokens, while query tokens attend causally to preceding queries and have full cross-attention to all image tokens. This allows simultaneous visual feature integration and sequential feature extraction in a single transformer pass.

## Contents

- [Setup](#setup)
- [Analysis Modules](#analysis-modules)
- [CLI Scripts](#cli-scripts)
- [Running Tests](#running-tests)
- [Upstream Inference](#upstream-inference)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)

## Setup

Requires Python >= 3.10. Tested with Python 3.13 and CUDA.

```bash
# Install with uv (recommended)
uv sync
uv run --extra dev pytest tests/ -v   # verify installation

# Or with pip
pip install -e ".[dev]"
```

Model weights are downloaded automatically from HuggingFace on first use (~6.4 GB).

### Dependencies

Core: `torch>=2.6.0`, `transformers==4.46.3`, `einops`, `safetensors`, `huggingface_hub`, `matplotlib`, `seaborn`, `scikit-learn`

Dev: `pytest`, `jupyter`

## Analysis Modules

All analysis tools live in `src/analysis/` and work with randomly-initialized model components (no GPU or model weights required for dry-run testing).

### Attention Analysis (`src/analysis/attention_analysis.py`)

Extract and analyze the D2E Visual Causal Flow attention patterns. Computes per-head entropy across query-to-image, image-to-image, and query-to-query regions. Ranks heads by focus or magnitude.

```python
from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
from src.analysis.attention_analysis import AttentionAnalyzer

model = build_qwen2_decoder_as_encoder(output_attentions=True)
analyzer = AttentionAnalyzer(model)

patterns = analyzer.extract_attention_patterns(sam_features)  # [B, 896, H, W]
spec = analyzer.analyze_head_specialization(patterns["attention_weights"], n_image_tokens=256)
top_heads = analyzer.find_important_heads(patterns["attention_weights"], n_image_tokens=256)
```

### Feature Extraction (`src/analysis/feature_extractor.py`)

Hook-based activation capture from SAM blocks, D2E layers, and the projector. Context-manager compatible.

```python
from src.analysis.feature_extractor import FeatureExtractor

with FeatureExtractor(model) as extractor:
    extractor.register_hooks(sam_layers=[0, 11], d2e_layers=[0, 23], projector=True)
    activations = extractor.extract(pixel_values, images_crop, images_spatial_crop)
    # activations["sam_layer_0"], activations["d2e_layer_23"], activations["projector"]
```

### Causal Interventions (`src/analysis/interventions.py`)

Ablate attention heads, zero query/image tokens, or patch activations at specific (layer, position) pairs. Context-manager ensures hook cleanup.

```python
from src.analysis.interventions import InterventionManager

with InterventionManager(model) as mgr:
    mgr.ablate_attention_head(layer=12, head=7, component="d2e")
    mgr.ablate_query_tokens(start_idx=0)
    output = model.get_multimodal_embeddings(pixel_values, images_crop, images_spatial_crop)
# hooks automatically removed
```

### Circuit Discovery (`src/analysis/circuits.py`)

Activation patching to identify causally important (layer, position) pairs. Patches clean activations into corrupted runs and ranks by metric impact.

```python
from src.analysis.circuits import CircuitDiscovery

discovery = CircuitDiscovery(model, feature_extractor, intervention_manager)
result = discovery.find_circuit_for_task(
    clean_input, corrupted_input, metric_fn, layers=[0, 6, 12, 23]
)
# result["critical_positions"] -> [(layer, position), ...] sorted by impact
```

### Projector Analysis (`src/analysis/projector_analysis.py`)

SVD decomposition of the linear projector weight. Computes effective rank, explained variance, and provides logit-lens decoding through a language model's unembedding matrix.

```python
from src.analysis.projector_analysis import ProjectorAnalyzer

analyzer = ProjectorAnalyzer(model.projector)
svd = analyzer.compute_svd()         # singular values, explained variance ratio
rank = analyzer.effective_rank(0.95)  # dimensions for 95% energy
logits = analyzer.logit_lens(visual_features, unembedding_weight)
```

### Query Specialization (`src/analysis/query_analysis.py`)

Inspect the learned query banks (norms, pairwise cosine similarity, cross-resolution similarity). Ablate query groups and measure score impact.

```python
from src.analysis.query_analysis import QuerySpecializationAnalyzer

analyzer = QuerySpecializationAnalyzer(d2e_model)
summary = analyzer.summarize_query_bank(768)    # norms, cosine similarity, mean_abs_cosine
cross = analyzer.cross_resolution_similarity()  # 768-bank vs 1024-bank cosine matrix
```

### Spatial Probing (`src/analysis/spatial_analysis.py`)

Closed-form ridge regression probe to test whether spatial coordinates are linearly decodable from intermediate activations. Reports MSE and per-dimension R-squared.

```python
from src.analysis.spatial_analysis import LinearSpatialProbe

probe = LinearSpatialProbe(l2_penalty=1e-4).fit(features, coordinates)
metrics = probe.evaluate(test_features, test_coordinates)
# metrics.mse, metrics.r2
```

### View Ablation (`src/analysis/view_analysis.py`)

Compare local-crop vs global-view contributions by zeroing each view's inputs and measuring score deltas.

```python
from src.analysis.view_analysis import ViewAblationAnalyzer

analyzer = ViewAblationAnalyzer(model)
result = analyzer.compare(model_inputs, score_fn=my_metric)
# result.local_delta, result.global_delta
```

## Visualization

`src/visualization/` provides plotting utilities for attention maps, feature trajectories, and comprehensive reports:

- **`AttentionVisualizer`** — attention mask plots, layer evolution grids, query-to-image overlays, head comparison, entropy heatmaps, animated GIFs across layers
- **`FeatureVisualizer`** — SAM feature channel grids, D2E hidden state PCA, activation norm heatmaps, projector output analysis

## CLI Scripts

| Script | Purpose |
|--------|---------|
| `scripts/extract_attention.py` | Extract and visualize D2E attention patterns. Supports `--synthetic` mode (no weights needed) and `--full_report`. |
| `scripts/extract_features.py` | Capture intermediate activations via hooks, save to `.pt` file. |
| `scripts/run_interventions.py` | Run ablation experiments (head, query, SAM head) and report embedding norm changes. |
| `scripts/run_real_circuit_mapping.py` | Map causal D2E sites on OmniDocBench pages by corrupting annotated regions and patching query states. |
| `scripts/run_omnidocbench.py` | Bulk OmniDocBench inference with filtering, pagination, dry-run, and benchmark-ready markdown export. |
| `scripts/check_omnidocbench_outputs.py` | Validate OmniDocBench runner outputs and optionally compare against reference markdown. |
| `scripts/simple_inference.py` | Single-image OCR inference via the research pipeline. |

Example:

```bash
# Attention extraction with synthetic data (no model download)
uv run python scripts/extract_attention.py --synthetic --output_dir output/attention

# Feature extraction from a real image
uv run python scripts/extract_features.py --image_path input/example.jpg --output_path output/features.pt

# Head ablation experiment
uv run python scripts/run_interventions.py --image_path input/example.jpg --ablate_head 12,7

# Real-document circuit mapping on table regions
uv run python scripts/run_real_circuit_mapping.py --dataset_root /path/to/OmniDocBench --region_type table --limit 10

# Bulk OmniDocBench run + output validation
uv run python scripts/run_omnidocbench.py --dataset_root /path/to/OmniDocBench --output_dir output/omnidocbench --limit 20
uv run python scripts/check_omnidocbench_outputs.py --output_dir output/omnidocbench --verbose
```

## Running Tests

The default suite runs CPU-friendly coverage for the core research code, benchmark runner, and pipeline helpers without requiring pretrained weights. Additional GPU-gated tests validate the real Hugging Face OCR path and the end-to-end interpretability CLIs.

```bash
uv run --extra dev pytest tests/ -v

# Include real-model GPU checks (requires CUDA + downloaded weights)
RUN_GPU_TESTS=1 uv run --extra dev pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_models.py` | SAM encoder, D2E forward (1024/768), attention output, projector, model init |
| `test_attention.py` | Attention shapes, token type IDs, visualizer, entropy, analyzer |
| `test_interpretability_e2e.py` | Attention ranking, visualization/report generation, CLI extraction scripts, GPU end-to-end interpretability runs |
| `test_interventions.py` | Feature extractor hooks, intervention context manager, head ablation |
| `test_circuits.py` | Activation patching, circuit ranking |
| `test_projector_analysis.py` | SVD recovery, effective rank, logit lens |
| `test_query_analysis.py` | Query bank geometry, cross-resolution similarity, group ablation |
| `test_real_circuit_mapping.py` | Region-box parsing, square-view mapping, target-query selection, alignment scoring |
| `test_spatial_analysis.py` | Linear probe fitting, pre-fit error guard |
| `test_view_analysis.py` | Local/global ablation, input immutability |
| `test_omnidocbench.py` | Dataset loading with filters, bulk runner output, dry-run |

## Upstream Inference

`DeepseekOCRPipeline.from_pretrained(...)` now follows the upstream `AutoModel(..., trust_remote_code=True)` loading path, with a fallback from FlashAttention to eager attention when `flash_attn` is unavailable. The pipeline returns cleaned markdown by default, stripping `<|ref|>/<|det|>` annotations from the saved `.mmd` output.

This repo does not vendor the original DeepSeek-OCR-2 source tree. The upstream
inference path is accessed directly through the Hugging Face model:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, _attn_implementation="eager",
    trust_remote_code=True, use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
res = model.infer(
    tokenizer, prompt=prompt, image_file="input/document.jpg",
    output_path="output/result", base_size=1024, image_size=768,
    crop_mode=True, save_results=True,
)
```

**Prompts:**
- With layout detection: `<image>\n<|grounding|>Convert the document to markdown.`
- Plain OCR: `<image>\nFree OCR.`

## Project Structure

```
deepseek-ocr-2/
+-- src/
|   +-- models/              # SAM encoder, D2E, projector, full model
|   +-- preprocessing/       # Image transforms, dynamic cropping
|   +-- analysis/            # 8 interpretability modules
|   +-- visualization/       # Attention and feature visualization
|   +-- inference/           # End-to-end pipeline (no vLLM)
|   +-- benchmarks/          # OmniDocBench dataset loader/runner
|   +-- config.py            # Central constants
+-- scripts/                 # CLI entry points
+-- tests/                   # Unit and integration coverage
+-- docs/                    # Architecture, API, and research notes
+-- input/                   # Dataset notes and sample inputs
+-- output/                  # Saved experiment artifacts and reports
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/RESEARCH_AUDIT.md](docs/RESEARCH_AUDIT.md) | Implementation-grounded findings, undocumented behaviors, and experiment-backed observations |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | Full architecture documentation with component details |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline diagrams, module overview, key dimensions |
| [docs/API.md](docs/API.md) | API reference for all public classes |
| [docs/ATTENTION_ANALYSIS.md](docs/ATTENTION_ANALYSIS.md) | Attention extraction and visualization guide |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | Current visualization workflow and report generation paths |
| [docs/MECH_INTERP_TESTS.md](docs/MECH_INTERP_TESTS.md) | Dry-run test philosophy and lightweight test guide |
| [docs/OMNIDOCBENCH.md](docs/OMNIDOCBENCH.md) | OmniDocBench data format and bulk runner usage |
| [docs/SPARSE_AUTO_ENCODER.md](docs/SPARSE_AUTO_ENCODER.md) | SAE workflow and layer-12 decomposition results |

## Acknowledgement

Built on top of [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) by DeepSeek AI. We also acknowledge [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR/), [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and the [OmniDocBench](https://github.com/opendatalab/OmniDocBench) benchmark.

## Citation

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
@article{wei2026deepseek,
  title={DeepSeek-OCR 2: Visual Causal Flow},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2601.20552},
  year={2026}
}
```

## License

This project is a derivative work based on [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) (Copyright 2023 DeepSeek). All modifications Copyright 2026 Tasoo Park. Licensed under Apache 2.0 — see [LICENSE.txt](LICENSE.txt).
