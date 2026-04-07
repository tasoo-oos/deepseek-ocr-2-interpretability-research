# DeepSeek-OCR-2: Visual Causal Flow — Code Structure

## Directory Layout

```
deepseek-ocr-2/
├── DeepSeek-OCR2-master/
│   ├── DeepSeek-OCR2-vllm/                  # vLLM-based inference (primary)
│   │   ├── deepencoderv2/                    # Vision encoding modules
│   │   │   ├── sam_vary_sdpa.py              #   SAM ViT-B image encoder
│   │   │   ├── qwen2_d2e.py                  #   Qwen2 decoder-as-encoder
│   │   │   └── build_linear.py               #   MLP projector (feature projection)
│   │   ├── process/                          # Preprocessing pipeline
│   │   │   ├── image_process.py              #   Image tokenization & dynamic cropping
│   │   │   └── ngram_norepeat.py             #   N-gram repetition logits processor
│   │   ├── config.py                         # Runtime configuration
│   │   ├── deepseek_ocr2.py                  # Main vLLM model definition
│   │   ├── run_dpsk_ocr2_image.py            # Single-image inference (async streaming)
│   │   ├── run_dpsk_ocr2_pdf.py              # PDF batch processing
│   │   └── run_dpsk_ocr2_eval_batch.py       # Benchmark evaluation
│   └── DeepSeek-OCR2-hf/                     # HuggingFace Transformers inference
│       └── run_dpsk_ocr2.py                  #   Direct transformers usage
├── input/                                    # Input files
├── output/                                   # Generated results
├── assets/                                   # Documentation assets
├── requirements.txt                          # Python dependencies
├── LICENSE.txt
└── README.md
```

Additional documentation:

- `docs/ARCHITECTURE.md` — cleaned architecture reference for the research codebase
- `docs/API.md` — API reference for models, preprocessing, analysis, visualization, and inference
- `docs/ATTENTION_ANALYSIS.md` — D2E attention extraction and visualization guide
- `docs/MECH_INTERP_TESTS.md` — dry-run mechanistic interpretability test guide for slow hardware
- `docs/OMNIDOCBENCH.md` — OmniDocBench data format notes and bulk inference runner usage

## Model Architecture

The model follows a **Vision-to-Language** multimodal pipeline with a novel "Visual Causal Flow" attention mechanism.

```
Input Image
    │
    ├─── Global View (1024×1024, padded) ───┐
    │                                       │
    └─── Local Views (N × 768×768 crops) ──┐│
                                           ││
                          ┌────────────────┘│
                          ▼                 ▼
                   SAM ViT-B Encoder (768-dim → 896-dim via neck)
                          │                 │
                          ▼                 ▼
                   Qwen2 Decoder-as-Encoder (896-dim)
                   (mixed causal / non-causal attention)
                          │                 │
                          ▼                 ▼
                   MLP Projector (896-dim → 1280-dim)
                          │                 │
                          ▼                 ▼
                   Concat [local, global, view_separator]
                          │
                          ▼
                   Merge into text embeddings at <image> positions
                          │
                          ▼
                   DeepSeek Language Model → Markdown text
```

### Component Details

#### 1. SAM Vision Encoder (`sam_vary_sdpa.py`)

`ImageEncoderViT` — a Vision Transformer based on SAM (Segment Anything Model):

| Parameter       | Value                    |
|-----------------|--------------------------|
| Image size      | 1024×1024                |
| Patch size      | 16×16                    |
| Embedding dim   | 768                      |
| Depth           | 12 transformer blocks    |
| Attention heads | 12                       |
| Window size     | 14 (local attention)     |
| Global attn     | Layers 2, 5, 8, 11       |

The **neck** downsamples and projects: Conv(256→512, stride 2) → Conv(512→896, stride 2), producing 896-dim spatial features.

Key classes: `ImageEncoderViT`, `Block`, `Attention`, `PatchEmbed`, `LayerNorm2d`

#### 2. Qwen2 Decoder-as-Encoder (`qwen2_d2e.py`)

`Qwen2Decoder2Encoder` — uses a Qwen2 decoder with custom attention masking as a visual encoder.

| Parameter          | Value    |
|--------------------|----------|
| Decoder layers     | 24       |
| Hidden dim         | 896      |
| Attention heads    | 14       |
| KV heads           | 2        |
| Intermediate size  | 4864     |
| Max position embed | 131,072  |

**Visual Causal Flow mechanism:**
- `token_type_ids=0` (image tokens): **non-causal** attention — image tokens attend to all other image tokens
- `token_type_ids=1` (query tokens): **causal** attention — queries attend only to preceding tokens
- Learnable query embeddings: `query_768` (144 queries for 768px patches), `query_1024` (256 queries for 1024px global view)

Forward pass: flatten spatial features → concat with learned queries → mixed-attention Qwen2 → extract causal query outputs.

#### 3. MLP Projector (`build_linear.py`)

`MlpProjector` — projects vision features into the language model's embedding space.

Default: single linear layer (896 → 1280). Supports multiple modes: `identity`, `linear`, `mlp_gelu`, `downsample_mlp_gelu`, etc.

#### 4. Language Model (`deepseek_ocr2.py`)

`DeepseekOCR2ForCausalLM` — the full vision-language model. Selects language backbone based on config:
- `DeepseekV3ForCausalLM` (if `topk_method == "noaux_tc"`)
- `DeepseekV2ForCausalLM` (if `use_mla == True`)
- `DeepseekForCausalLM` (fallback)

A learnable `view_separator` (1280-dim) token separates local and global view embeddings.

## Image Processing Pipeline (`process/image_process.py`)

### Dynamic Cropping

`dynamic_preprocess(image, min_num=2, max_num=6, image_size=768)`:
1. Compute aspect ratio of input image
2. Find optimal tiling arrangement (e.g., 2×3) minimizing wasted area
3. Resize image to fill tile grid
4. Crop into individual 768×768 patches

### Tokenization

`tokenize_with_images(images, prompt)`:
1. Split prompt on `<image>` tags
2. For each image:
   - Create **global view**: pad to 1024×1024
   - Create **local views**: dynamic crop into 768×768 patches (if image > 768×768)
3. Normalize tensors (mean=0.5, std=0.5)
4. Compute visual token counts:
   - Per view: `(size // 16) // 4 = 16` → 16×16 = **256 tokens**
   - Total: `local_tokens + global_tokens + 1` (separator)
5. Output: `input_ids`, `pixel_values`, `images_crop`, `images_spatial_crop`, `images_seq_mask`

### Generation Constraints (`process/ngram_norepeat.py`)

`NoRepeatNGramLogitsProcessor` — prevents n-gram repetition during decoding:
- `ngram_size=20`, `window_size=90` (images) / `50` (PDFs)
- Whitelist: `{128821, 128822}` (`<td>`, `</td>`)

## Inference Modes

### 1. Single Image — `run_dpsk_ocr2_image.py`

Async streaming via vLLM `AsyncLLMEngine`:
- Loads image, corrects EXIF orientation
- Streams tokens as generated
- Post-processes: extracts grounding references, draws bounding boxes, saves markdown

Engine config: `dtype=bfloat16`, `max_model_len=8192`, `gpu_memory_utilization=0.75`

### 2. PDF Batch — `run_dpsk_ocr2_pdf.py`

Concurrent multi-page processing:
- Converts PDF pages to images via PyMuPDF (dpi=144)
- Preprocesses in parallel (64 workers)
- Batch generates with vLLM (max concurrency=100, GPU util=0.9)
- Outputs: raw `.mmd`, cleaned `.mmd`, layout visualization PDF

### 3. Benchmark Evaluation — `run_dpsk_ocr2_eval_batch.py`

Batch evaluation over image directories (e.g., OmniDocBench).

### 4. HuggingFace — `DeepSeek-OCR2-hf/run_dpsk_ocr2.py`

Direct `AutoModel.from_pretrained` usage with `model.infer()` API.

## Configuration (`config.py`)

| Parameter         | Default | Description                           |
|-------------------|---------|---------------------------------------|
| `BASE_SIZE`       | 1024    | Global view image size                |
| `IMAGE_SIZE`      | 768     | Local crop patch size                 |
| `CROP_MODE`       | True    | Enable dynamic cropping               |
| `MIN_CROPS`       | 2       | Minimum crop count                    |
| `MAX_CROPS`       | 6       | Maximum crop count                    |
| `MAX_CONCURRENCY` | 100     | Max concurrent vLLM requests          |
| `NUM_WORKERS`     | 64      | Image preprocessing workers           |
| `SKIP_REPEAT`     | True    | Skip duplicate outputs                |
| `MODEL_PATH`      | `deepseek-ai/DeepSeek-OCR-2` | HuggingFace model ID    |
| `PROMPT`          | `<image>\n<|grounding|>Convert the document to markdown.` | Default prompt |

## Special Tokens

| Token             | Purpose                              |
|-------------------|--------------------------------------|
| `<image>`         | Image placeholder (id=32000)         |
| `<\|grounding\|>` | Enables grounding/layout mode        |
| `<\|ref\|>` / `<\|/ref\|>` | Reference label wrapper     |
| `<\|det\|>` / `<\|/det\|>` | Bounding box coordinates    |

Grounding output example:
```
<|ref|>figure<|/ref|><|det|>[[100,200,300,400]]<|/det|>
```
Coordinates are normalized to 0–999 range and scaled to image dimensions.

## Dependencies

Core: `transformers==4.46.3`, `tokenizers==0.20.3`, `PyMuPDF`, `img2pdf`, `einops`, `easydict`, `addict`, `Pillow`, `numpy`

Runtime: PyTorch 2.6.0 (CUDA 11.8+), vLLM 0.8.5, Flash Attention 2.7.3
