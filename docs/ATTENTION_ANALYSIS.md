# Attention Pattern Analysis

## Overview

The Qwen2 Decoder-as-Encoder (D2E) uses a **Visual Causal Flow** attention mechanism
with 24 layers and 14 attention heads per layer.

## Quick Start

### Extract attention patterns (no model weights needed)

```python
import torch
from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
from src.visualization.attention_viz import AttentionVisualizer

# Build D2E with attention output enabled
d2e = build_qwen2_decoder_as_encoder(output_attentions=True)
d2e.eval()

# Synthetic SAM features (1024px mode: 16×16 spatial grid)
sam_features = torch.randn(1, 896, 16, 16)

with torch.no_grad():
    result = d2e(sam_features, output_attentions=True)
    query_outputs, attentions, hidden_states, token_type_ids = result

# Visualize
viz = AttentionVisualizer(
    attention_weights=[a.cpu() for a in attentions],
    token_type_ids=token_type_ids.cpu(),
    spatial_size=16,
)
fig = viz.plot_attention_mask(layer=0)
fig.savefig("attention_mask.png", dpi=150, bbox_inches='tight')
```

### With a real image

```python
from src.models.deepseek_ocr import DeepseekOCRModel
from src.preprocessing.image_transforms import ImageProcessor
from PIL import Image

model = DeepseekOCRModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR-2",
    use_language_model=False,
    output_attentions=True,
    device="cuda",
).eval()

sam = model.sam_model
d2e = model.qwen2_model

processor = ImageProcessor()
image = Image.open("input/document.jpg")
inputs = processor.process_image(image)
pixel_values = inputs["pixel_values"][0].to("cuda", dtype=torch.bfloat16)  # [1, 3, H, H]

with torch.no_grad():
    sam_features = sam(pixel_values)
    result = d2e(sam_features, output_attentions=True)
    _, attentions, _, token_type_ids = result
```

## Visualization Types

### `plot_attention_mask(layer, head=None, show_expected=True)`
Shows actual vs expected attention mask. The expected mask shows the Visual Causal Flow pattern.

### `plot_layer_evolution(region='all')`
Grid of attention maps across all 24 layers.

Regions: `"all"`, `"image_to_image"`, `"query_to_image"`, `"query_to_query"`

### `plot_query_to_image(query_idx, layer, overlay_image=True)`
Which image regions does a specific query token attend to?
With `overlay_image=True`, overlays the heatmap on the original image.

### `plot_image_self_attention(position, layer)`
Where does an image token attend to other image tokens?

### `plot_causal_flow(layer)`
Causal (lower-triangular) attention among query tokens.

### `plot_head_comparison(layer, region='query_to_image')`
Side-by-side comparison of all 14 heads for a given layer.

### `plot_entropy_analysis()`
Attention entropy heatmaps across all layers and heads.
Lower entropy = more focused attention = potentially more important heads.

## Using AttentionAnalyzer

```python
from src.analysis.attention_analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(d2e)

# Extract attention with one call
result = analyzer.extract_attention_patterns(sam_features, layers=[0, 6, 12, 18, 23])

# Find most focused heads (low entropy = focused)
important_heads = analyzer.find_important_heads(
    result["attention_weights"],
    n_image_tokens=256,
    metric="entropy",
    region="query_to_image",
    top_k=10,
)

# Quantify head specialization
stats = analyzer.analyze_head_specialization(
    result["attention_weights"], n_image_tokens=256
)
# stats contains: entropy_q2i, entropy_i2i, entropy_q2q, q2i_ratio
# All tensors of shape [n_layers, n_heads]
```

## Generate a Full Report

```python
viz.create_summary_report(
    output_dir="output/attention_report",
    layers_to_visualize=[0, 6, 12, 18, 23],
    include_animation=False,  # Set True for GIF (requires imageio)
)
```

Output directory structure:
```
attention_report/
├── attention_mask.png
├── layer_evolution.png
├── entropy_analysis.png
├── query_to_image/
│   └── layer_XX_query_YYY.png
├── causal_flow/
│   └── layer_XX.png
├── head_analysis/
│   ├── head_comparison_query_to_image.png
│   └── entropy_analysis.png
└── metadata.json
```

## Command-Line Usage

```bash
# Synthetic data (no model weights):
uv run python scripts/extract_attention.py --synthetic --output_dir output/attn

# Real image:
uv run python scripts/extract_attention.py \
    --image_path input/document.jpg \
    --output_dir output/attn \
    --device cuda \
    --layers 0,6,12,18,23 \
    --full_report
```

The CLI's real-image mode now loads pretrained SAM + D2E weights through `DeepseekOCRModel.from_pretrained(...)`, so the generated reports reflect the actual checkpoint instead of a randomly initialized dry-run model.
