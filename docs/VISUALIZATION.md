# DeepSeek-OCR-2: Attention Visualization Workflow

## Overview

The visualization module provides tools to extract and analyze attention patterns from the **Visual Causal Flow** mechanism in DeepSeek-OCR-2's Qwen2 Decoder-as-Encoder (D2E). It enables researchers to understand how the model processes visual information through its mixed causal/non-causal attention architecture.

## Architecture

```
Image Input
    │
    ├─── SAM Vision Encoder ───┐
    │                           │
    │                           ▼
    │                   Spatial Features [896, H, W]
    │                           │
    │                           ▼
    │              Qwen2 Decoder-as-Encoder (D2E)
    │              with Visual Causal Flow Attention
    │                           │
    │                           ▼
    │                 Attention Weights (24 layers × 14 heads)
    │                           │
    │                           ▼
    │                 AttentionVisualizer ────▶ Multiple Plot Types
```

## Components

### 1. Attention Extractor (`visualization/attention_extractor.py`)

**`AttentionExtractor`** – Main class for extracting attention weights:

- **SAM Encoder**: Vision Transformer backbone producing 896-dim spatial features
- **Qwen2 D2E with Attention Support**: Modified decoder with `output_attentions=True`
- **Visual Causal Flow Masking**: Implements mixed attention patterns:
  - Image tokens (type_id=0): **Non-causal** – attend to all image tokens
  - Query tokens (type_id=1): **Causal** – attend to all image tokens + causally to preceding queries

**Key Methods**:
- `extract_attention()`: Process image through SAM → D2E → return `AttentionOutput`
- `get_attention_mask_pattern()`: Generate expected mask for verification

**Data Container**:
```python
@dataclass
class AttentionOutput:
    image: Image.Image                    # Original image
    sam_features: torch.Tensor           # [B, 896, H, W]
    attention_weights: List[torch.Tensor] # List of [B, H, S, S] per layer
    token_type_ids: torch.Tensor         # [B, seq_len] (0=image, 1=query)
    query_outputs: torch.Tensor          # [B, n_queries, hidden_dim]
    n_image_tokens: int                  # H*W
    n_query_tokens: int                  # Same as n_image_tokens
    spatial_size: int                    # 16 (1024px) or 12 (768px)
```

### 2. Attention Visualizer (`visualization/attention_visualizer.py`)

**`AttentionVisualizer`** – Creates comprehensive visualizations from `AttentionOutput`:

- **Mask Pattern**: Compare actual attention vs. expected Visual Causal Flow mask
- **Layer Evolution**: Track attention changes across 24 transformer layers
- **Query→Image**: Show which image regions specific queries attend to
- **Image Self-Attention**: Visualize receptive fields for spatial positions
- **Causal Flow**: Analyze causal attention among query tokens
- **Head Comparison**: Compare attention patterns across 14 attention heads
- **Entropy Analysis**: Quantify attention concentration/diffusion

**Key Methods**:
- `plot_attention_mask()`: Visualize full attention matrix with region boundaries
- `plot_layer_evolution()`: Grid of attention patterns across all layers
- `plot_query_to_image()`: Overlay query attention on original image
- `plot_image_self_attention()`: Self-attention for specific spatial positions
- `plot_causal_flow()`: Query-to-query causal attention verification
- `plot_head_comparison()`: Compare all heads in a layer
- `plot_entropy_analysis()`: Compute and visualize attention entropy
- `create_summary_report()`: Generate comprehensive report with all visualizations

### 3. Utilities (`visualization/utils.py`)

Helper functions for attention processing:

- **Normalization**: `normalize_attention()` – Ensure attention sums to 1
- **Metrics**: `compute_attention_entropy()` – Measure attention concentration
- **Spatial Operations**: `reshape_attention_to_spatial()` – Convert flat→grid
- **Region Extraction**: `extract_attention_regions()` – Split into I→I, I→Q, Q→I, Q→Q
- **Mask Generation**: `create_attention_mask_visualization()` – Expected pattern
- **Image Overlay**: `overlay_attention_on_image()` – Blend attention maps

### 4. Command-Line Interface (`visualization/run_visualization.py`)

**Usage**:
```bash
# Real image with model
python visualization/run_visualization.py \
    --image_path input/example.jpg \
    --output_dir output/attention_viz \
    --layers 0,6,12,18,23 \
    --viz_types mask,evolution,query_to_image

# Synthetic data (no model loading)
python visualization/run_visualization.py \
    --synthetic --output_dir output/attention_viz

# Full comprehensive report
python visualization/run_visualization.py \
    --image_path input/example.jpg \
    --output_dir output/full_report \
    --full_report
```

**Arguments**:
- `--image_path`: Input image (required for real mode)
- `--sam_checkpoint`: Path to SAM encoder checkpoint
- `--d2e_checkpoint`: Path to D2E checkpoint
- `--target_size`: 768 or 1024 (default: 1024)
- `--layers`: Comma-separated layer indices
- `--viz_types`: Visualization types to generate
- `--full_report`: Generate comprehensive report with all visualizations
- `--synthetic`: Use synthetic data (no model loading)

## Workflow

### Step 1: Data Extraction
1. Load image and pad to target size (1024×1024 or 768×768)
2. Process through SAM encoder → 896-dim spatial features
3. Pass through Qwen2 D2E with `output_attentions=True`
4. Collect attention weights from all 24 layers × 14 heads

### Step 2: Analysis & Visualization
1. **Mask Verification**: Compare actual attention against expected Visual Causal Flow pattern
2. **Layer Analysis**: Examine how attention evolves through transformer depth
3. **Spatial Analysis**: Map attention patterns back to image coordinates
4. **Head Diversity**: Analyze specialization across attention heads
5. **Entropy Quantification**: Measure attention concentration trends

### Step 3: Output Generation
1. Generate individual plots for specific analyses
2. Create comprehensive PDF/HTML reports
3. Save metadata (layer counts, token counts, spatial dimensions)
4. Optional: Create GIF animations of layer evolution

## Visual Causal Flow Mask Pattern

The core innovation visualized is the **mixed attention pattern**:

```
          Image Tokens (n)     Query Tokens (n)
          ┌───────────────┬───────────────────┐
          │               │                   │
Image     │  Non-causal   │     Blocked       │
Tokens    │  (Full)       │   (Image→Query)   │
(n)       │               │                   │
          ├───────────────┼───────────────────┤
          │               │                   │
Query     │   Full        │     Causal        │
Tokens    │ Cross-attn    │   (Lower tri)     │
(n)       │ (Query→Image) │                   │
          └───────────────┴───────────────────┘
```

- **Image→Image**: Full bidirectional attention (non-causal)
- **Image→Query**: Blocked (image tokens cannot attend to queries)
- **Query→Image**: Full cross-attention (queries attend to all image tokens)
- **Query→Query**: Causal attention (lower triangular matrix)

## Example Visualizations

### 1. Attention Mask Pattern
![Mask Pattern](example_mask.png)
*Comparison of actual attention (left) vs. expected mask (right)*

### 2. Layer Evolution
![Layer Evolution](example_evolution.png)
*Attention patterns across 24 transformer layers*

### 3. Query→Image Attention
![Query to Image](example_query_to_image.png)
*Which image regions query #128 attends to (overlay on original)*

### 4. Head Comparison
![Head Comparison](example_head_comparison.png)
*Attention patterns across 14 heads in layer 23*

### 5. Entropy Analysis
![Entropy Analysis](example_entropy.png)
*Attention entropy across layers and heads*

## Usage Examples

### Research Analysis
```python
from visualization.attention_extractor import AttentionExtractor
from visualization.attention_visualizer import AttentionVisualizer

# Extract attention
extractor = AttentionExtractor(sam_checkpoint="sam_vit_b.pth")
attention_output = extractor.extract_attention(image, target_size=1024)

# Analyze
visualizer = AttentionVisualizer(attention_output)
fig = visualizer.plot_attention_mask(layer=12)
fig.savefig("attention_layer_12.png")

# Full report
visualizer.create_summary_report(
    output_dir="analysis_report",
    layers_to_visualize=[0, 6, 12, 18, 23],
    include_animation=True
)
```

### Model Debugging
```python
# Verify Visual Causal Flow implementation
expected_mask = extractor.get_attention_mask_pattern(n_tokens=256)
actual_mask = attention_output.attention_weights[0].mean(dim=1)[0]

# Check mask compliance
mask_diff = torch.abs(expected_mask - actual_mask)
print(f"Mask compliance: {(mask_diff < 0.1).float().mean():.2%}")
```

### Educational Visualization
```python
# Interactive exploration
for layer in [0, 8, 16, 23]:
    for query_idx in [0, 64, 128, 192, 255]:
        fig = visualizer.plot_query_to_image(
            query_idx=query_idx,
            layer=layer,
            overlay_image=True
        )
        plt.show()
```

## File Reference

- **`visualization/__init__.py`**: Module exports (`AttentionExtractor`, `AttentionVisualizer`)
- **`visualization/attention_extractor.py`**: Core extraction logic (463 lines)
- **`visualization/attention_visualizer.py`**: Visualization implementations (691 lines)
- **`visualization/run_visualization.py`**: CLI interface (315 lines)
- **`visualization/utils.py`**: Utility functions (254 lines)

## Dependencies

- **Core**: PyTorch, Transformers, PIL/Pillow
- **Visualization**: Matplotlib, NumPy
- **Optional**: imageio (for GIF animations)
- **Parent Module**: SAM encoder, Qwen2 D2E from `deepencoderv2/`

## Notes

1. **Synthetic Mode**: Useful for testing without model weights
2. **Memory Usage**: Attention extraction for 1024px images requires ~8GB GPU memory
3. **Batch Processing**: Currently supports single-image analysis
4. **Extension Points**: Easy to add custom visualizations via `AttentionVisualizer` subclass

This visualization toolkit provides deep insights into the Visual Causal Flow mechanism, enabling both research analysis and model debugging for DeepSeek-OCR-2's novel attention architecture.
