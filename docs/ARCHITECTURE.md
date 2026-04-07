# Architecture Reference

## Model Pipeline

DeepSeek-OCR-2 processes images through three sequential visual encoding stages:

```
Image (PIL)
    │
    ▼
ImageProcessor (src/preprocessing/)
    ├── Global view: pad to 1024×1024
    └── Local crops: dynamic tiling (768×768 tiles)
    │
    ▼
SAM Encoder (src/models/sam_encoder.py)
    ├── ViT-B backbone (12 layers, 768-dim)
    ├── Neck: 4× conv layers → 256-dim features
    ├── net_2: stride-2 conv → 512-dim
    └── net_3: stride-2 conv → 896-dim
    │   Output: [B, 896, H, W]  (H=W=16 for 1024px, 12 for 768px)
    │
    ▼
Qwen2 Decoder-as-Encoder (src/models/qwen2_d2e.py)
    ├── Input: flatten SAM features → [B, H*W, 896]
    ├── Prepend learnable queries: [B, H*W, 896]
    ├── Concatenate: [B, 2×H*W, 896]
    ├── Custom attention mask (Visual Causal Flow):
    │   ┌──────────────────────────────────┐
    │   │   Image → Image : bidirectional  │
    │   │   Query → Image : full attend    │
    │   │   Query → Query : causal         │
    │   │   Image → Query : blocked        │
    │   └──────────────────────────────────┘
    ├── 24× Qwen2 transformer layers (896-dim, 14 heads)
    └── Extract query outputs: [B, H*W, 896]
    │
    ▼
MLP Projector (src/models/projector.py)
    └── Linear: 896 → 1280
    │   Output: [B, H*W, 1280]
    │
    ▼
Language Model (e.g. DeepSeek-V3)
    └── Concatenate with text embeddings → generate
```

## Module Overview

### `src/models/`
| File | Contents | Source |
|------|----------|--------|
| `sam_encoder.py` | SAM ViT-B image encoder | `deepencoderv2/sam_vary_sdpa.py` |
| `qwen2_d2e.py` | Visual Causal Flow D2E | `deepencoderv2/qwen2_d2e.py` |
| `projector.py` | MLP projector | `deepencoderv2/build_linear.py` |
| `deepseek_ocr.py` | Full model (no vLLM) | `deepseek_ocr2.py` (cleaned) |

### `src/preprocessing/`
| File | Contents |
|------|----------|
| `dynamic_cropping.py` | `find_closest_aspect_ratio`, `count_tiles`, `dynamic_preprocess` |
| `image_transforms.py` | `ImageTransform`, `ImageProcessor` |

### `src/analysis/`
| File | Contents |
|------|----------|
| `attention_analysis.py` | `AttentionAnalyzer` — extract & rank attention patterns |
| `feature_extractor.py` | `FeatureExtractor` — hook-based activation capture |
| `interventions.py` | `InterventionManager` — ablations & activation patching |
| `circuits.py` | `CircuitDiscovery` — activation patching for circuit finding |

### `src/visualization/`
| File | Contents |
|------|----------|
| `utils.py` | Shared utilities (entropy, spatial reshape, overlay, ...) |
| `attention_viz.py` | `AttentionVisualizer` — full suite of attention plots |
| `feature_viz.py` | `FeatureVisualizer` — SAM features, hidden states, projector |

### `src/inference/`
| File | Contents |
|------|----------|
| `pipeline.py` | `DeepseekOCRPipeline` — end-to-end image→text |
| `batch_inference.py` | `run_batch()` — loop over image files |

## Key Dimensions

| Component | Input | Output |
|-----------|-------|--------|
| SAM encoder (1024px) | [B, 3, 1024, 1024] | [B, 896, 16, 16] |
| SAM encoder (768px)  | [B, 3, 768, 768]   | [B, 896, 12, 12] |
| D2E (1024px) | [B, 896, 16, 16] | [B, 256, 896] |
| D2E (768px)  | [B, 896, 12, 12] | [B, 144, 896] |
| Projector | [B, N, 896] | [B, N, 1280] |

## Visual Causal Flow

The D2E uses `token_type_ids` to build a custom 4-D attention mask:

```
token position:  0 … 255  │  256 … 511
token type:      image (0) │  query (1)
                           │
                 ┌─────────┬────────┐
image (row)      │  ✓ full  │   ✗   │
query (row)      │  ✓ full  │ ✓ causal│
                 └─────────┴────────┘
```

This design lets each query token aggregate information from all image tokens
while maintaining causal order among queries, enabling sequential feature extraction.
