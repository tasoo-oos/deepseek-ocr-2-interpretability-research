# API Reference

## Models

### `DeepseekOCRModel`
`src.models.deepseek_ocr.DeepseekOCRModel`

Clean DeepSeek-OCR-2 model without vLLM dependencies.

```python
model = DeepseekOCRModel(
    use_language_model=False,    # Load LM component
    output_attentions=False,     # Default attention output for D2E
    output_hidden_states=False,  # Default hidden state output for D2E
    image_token_id=None,         # Token ID for <image>
    attn_implementation="sdpa",  # "sdpa" or "eager"
)

# Load from HuggingFace Hub or local directory
model = DeepseekOCRModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR-2",
    use_language_model=False,
    device="cuda",
    dtype=torch.bfloat16,
)

# Vision-only forward pass
embeddings = model.get_multimodal_embeddings(
    pixel_values,       # [n_img, 1, 3, H, W]
    images_crop,        # [n_img, 1, n_patches, 3, h, w]
    images_spatial_crop, # [n_img, 1, 2]
    return_intermediate=False,
)
```

### `Qwen2Decoder2Encoder`
`src.models.qwen2_d2e.Qwen2Decoder2Encoder`

```python
model = build_qwen2_decoder_as_encoder(
    decoder_layer=24,
    hidden_dimension=896,
    num_attention_heads=14,
    num_key_value_heads=2,
    intermediate_size=4864,
    attn_implementation="sdpa",
    output_attentions=False,
    output_hidden_states=False,
)

# Forward pass (returns query outputs if attentions/hidden_states disabled)
out = model(sam_features)  # [B, n_queries, 896]

# With attention output
query_out, attentions, hidden_states, token_type_ids = model(
    sam_features, output_attentions=True
)
```

### `build_sam_vit_b`
`src.models.sam_encoder.build_sam_vit_b`

```python
sam = build_sam_vit_b(checkpoint=None)  # checkpoint: optional path to .pt
out = sam(images)  # [B, 896, H//64, W//64]
```

---

## Preprocessing

### `ImageProcessor`
`src.preprocessing.image_transforms.ImageProcessor`

```python
processor = ImageProcessor(
    image_size=768,   # Local crop tile size
    base_size=1024,   # Global view size
    crop_mode=True,
)

inputs = processor.process_image(image)
# Returns:
#   pixel_values:        [1, 1, 3, base_size, base_size]
#   images_crop:         [1, 1, n_patches, 3, image_size, image_size]
#   images_spatial_crop: [1, 1, 2]
```

---

## Analysis

### `FeatureExtractor`
`src.analysis.feature_extractor.FeatureExtractor`

```python
extractor = FeatureExtractor(model)
extractor.register_hooks(sam_layers=[0,6,11], d2e_layers=[0,12,23], projector=True)
activations = extractor.extract(pixel_values, images_crop, images_spatial_crop)
extractor.clear_hooks()
# activations: dict[str, Tensor] with keys like "sam_layer_0", "d2e_layer_12", "projector"
```

### `InterventionManager`
`src.analysis.interventions.InterventionManager`

```python
with InterventionManager(model) as mgr:
    mgr.ablate_attention_head(layer=12, head=7, component="d2e")
    mgr.ablate_query_tokens(start_idx=0, end_idx=None)
    mgr.patch_activation(layer=6, position=128, new_value=tensor, component="d2e")
    output = model.get_multimodal_embeddings(...)
# Hooks automatically removed on exit
```

### `AttentionAnalyzer`
`src.analysis.attention_analysis.AttentionAnalyzer`

```python
analyzer = AttentionAnalyzer(qwen2_model)
result = analyzer.extract_attention_patterns(sam_features, layers=[0,12,23])
important_heads = analyzer.find_important_heads(
    result["attention_weights"], n_image_tokens=256, metric="entropy", top_k=5
)
stats = analyzer.analyze_head_specialization(
    result["attention_weights"], n_image_tokens=256
)
```

### `CircuitDiscovery`
`src.analysis.circuits.CircuitDiscovery`

```python
discovery = CircuitDiscovery(model, feature_extractor, intervention_manager)

# Single-position activation patch
impact = discovery.activation_patching(
    clean_input=clean_inputs,
    corrupted_input=corrupted_inputs,
    layer=12, position=128,
    metric_fn=lambda out: ...,
)

# Find circuit for a task
result = discovery.find_circuit_for_task(
    clean_input=..., corrupted_input=...,
    metric_fn=..., layers=[0,6,12,18,23], n_positions=16
)
# result["critical_positions"] = [(layer, pos), ...]
```

---

## Visualization

### `AttentionVisualizer`
`src.visualization.attention_viz.AttentionVisualizer`

```python
viz = AttentionVisualizer(
    attention_weights=attentions,   # List[[B,H,S,S]]
    token_type_ids=token_type_ids,  # [B, S]
    spatial_size=16,                # SAM feature spatial size
    image=pil_image,                # Optional, for overlays
)

viz.plot_attention_mask(layer, head=None, show_expected=True)
viz.plot_layer_evolution(region="query_to_image")
viz.plot_query_to_image(query_idx, layer)
viz.plot_image_self_attention(position, layer)
viz.plot_causal_flow(layer)
viz.plot_head_comparison(layer, region="query_to_image")
viz.plot_entropy_analysis()
viz.create_summary_report(output_dir, layers_to_visualize=[0,6,12,18,23])
```

### `FeatureVisualizer`
`src.visualization.feature_viz.FeatureVisualizer`

```python
fviz = FeatureVisualizer()
fviz.plot_sam_features(sam_output, channels=None)
fviz.plot_d2e_hidden_states(hidden_states, layer=0)
fviz.plot_projector_output(proj_output)
fviz.plot_activation_trajectory(activations, position=0)
```

---

## Inference

### `DeepseekOCRPipeline`
`src.inference.pipeline.DeepseekOCRPipeline`

```python
pipeline = DeepseekOCRPipeline.from_pretrained(
    "deepseek-ai/DeepSeek-OCR-2", device="cuda"
)
text = pipeline(image, prompt="<image>\nConvert to markdown.", max_new_tokens=2048)
raw_text = pipeline("input/document.jpg", raw=True)
```

Notes:

- `from_pretrained()` uses the upstream `AutoModel(..., trust_remote_code=True)` path and falls back from FlashAttention to eager attention when needed.
- `__call__()` accepts either a PIL image or an image path.
- Cleaned output is returned by default; set `raw=True` to keep `<|ref|>/<|det|>` annotations.
