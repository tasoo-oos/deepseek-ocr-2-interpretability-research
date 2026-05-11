# Experiment Plan: Visual Causal Flow

Goal: test whether DeepSeek-OCR 2's claimed visual causal flow is carried by DeepEncoder V2 causal query tokens, rather than by SAM features, crop order, projector behavior, or the DeepSeek decoder.

The experiments are ordered by required run/implementation step. Later causal claims should not be made until intervention runs are working and compared against controls.

## Run 0: Improve Capture Infrastructure

Purpose: make activation capture suitable for causal-flow work.

Current `ActivationStore` stores one value per module name, so repeated calls overwrite earlier outputs. This is not enough because modules can run for local crops, global views, and repeated decoder-generation steps.

Implement:

- Store a list of calls per module.
- Store call index, shape, dtype, and optionally summary stats.
- Keep raw tensor capture for targeted one-page runs only.
- Keep summary-only capture for broad surveys.

Primary files:

- `src/ocr2interp/hooks.py`
- `scripts/capture_activations.py`
- optionally `scripts/run_omnidocbench_activation_survey.py`

Verify:

```bash
uv run ruff check src scripts
PYTHONPATH=src uv run python -m compileall src scripts
```

Expected outcome:

- A single capture run can distinguish the two calls to `model.qwen2_model` on local/global views.
- No accidental loss of earlier activation calls.

Done!


## Run 1: Stable Baseline With Full Capture

Purpose: establish reproducible no-intervention outputs and collect concrete activation data from the unmodified model path.

Run on a small fixed image set first, then expand to OmniDocBench samples.

Record:

- image path
- prompt/config
- crop count and image-token count
- output markdown/transcript
- output token count
- runtime
- whether generation hits `max_new_tokens`
- reading-order metric if labels are available
- full activation tensors for targeted modules on small runs
- activation summaries for broad runs

Suggested no-capture command:

```bash
uv run python scripts/infer_one.py input.image_file=data/raw/example.png output.dir=outputs/runs/baseline_example
```

Suggested targeted full-capture command:

```bash
uv run python scripts/capture_activations.py input.image_file=data/raw/example.png output.dir=outputs/runs/baseline_capture_example capture.modules='["model.qwen2_model.model.model.layers.0","model.qwen2_model.model.model.layers.23","model.projector"]'
```

Expected outcome:

- Stable outputs under the current default config.
- Full activation data for a small number of high-value modules/images.
- A known baseline for every later ablation/patching run.

Done!


## Run 2: Refined Descriptive Activation Survey

Purpose: map where activity changes across document types and crop settings before doing interventions.

Hook these modules:

- `model.sam_model.neck`
- `model.qwen2_model.query_768`
- `model.qwen2_model.query_1024`
- `model.qwen2_model.model.model.layers.0`
- `model.qwen2_model.model.model.layers.6`
- `model.qwen2_model.model.model.layers.12`
- `model.qwen2_model.model.model.layers.18`
- `model.qwen2_model.model.model.layers.23`
- `model.projector`
- `model.layers.0`
- `model.layers.11`

Suggested small run:

```bash
uv run python scripts/run_omnidocbench_activation_survey.py --limit 20 --output-dir outputs/runs/activation_survey_vcf_small
```

Suggested full run:

```bash
uv run python scripts/run_omnidocbench_activation_survey.py --resume --output-dir outputs/runs/activation_survey_vcf_full
```

Expected outcome:

- Per-module call counts and shapes by page.
- Better understanding of local/global calls.
- Descriptive statistics only, not causal evidence.

Done!


## Run 3: Final Query Output Ablation

Purpose: directly test whether the final causal query sequence is necessary for OCR output and reading order.

Target point:

```text
after model.qwen2_model(...)
before model.projector(...)
```

Interventions:

- zero all query outputs
- replace query outputs with per-token or global mean activation
- add Gaussian noise with matched activation scale
- keep only first quarter, middle half, or last quarter of query tokens

Controls:

- no intervention
- comparable noise after `model.projector`
- comparable decoder-layer perturbation

Expected outcome if causal query outputs matter:

- Reading order degrades strongly.
- Text recognition may degrade, but reading-order errors should be especially sensitive.

## Run 4: Final Query Order Permutation

Purpose: test whether query-token order is the decoder-facing reading-order channel.

Target point:

```text
after model.qwen2_model(...)
before model.projector(...)
```

Interventions:

- reverse query-token order
- random permutation of all query tokens
- random permutation within local crops only
- random permutation within global view only
- block permutation preserving local token neighborhoods

Expected outcome if query-token order carries visual causal flow:

- Reversing or permuting query outputs should disrupt layout/reading order.
- Local-only and global-only permutations should reveal which view contributes most.

High-value first causal test:

- This is likely the simplest direct test of the paper claim.

## Run 5: Visual Half Versus Query Half Intervention Inside Qwen2 Encoder

Purpose: separate visual-token representations from causal-query representations inside DeepEncoder V2.

At Qwen2 visual encoder layer outputs, hidden states should have shape like:

```text
[batch, 2 * n_query, 896]
```

Token halves:

```text
x[:, :n_query, :]      # visual tokens
x[:, n_query:, :]      # causal query tokens
```

Interventions:

- zero visual half at layer L
- zero query half at layer L
- add matched noise to visual half at layer L
- add matched noise to query half at layer L

Layer set:

```text
0, 3, 6, 9, 12, 15, 18, 21, 23
```

Expected outcome if visual causal flow emerges in queries:

- Early visual-token interventions may be important.
- Later query-token interventions should be more reading-order-specific.

## Run 6: Layer Sweep For Query Ablation

Purpose: locate where reading-order information emerges across the Qwen2 visual encoder.

Use the same intervention at each selected layer:

- zero query half
- mean-replace query half
- clean/corrupt patch query half once patching exists

Layer set:

```text
0, 3, 6, 9, 12, 15, 18, 21, 23
```

Measure:

- text edit distance
- reading-order edit distance
- output length
- failure/repetition rate
- qualitative layout errors

Expected outcome:

- A layerwise effect curve showing whether causal-flow information is early, gradual, or late.

## Run 7: Synthetic Clean/Corrupt Reading-Order Pairs

Purpose: build controlled examples where content is similar but reading order changes.

Suggested pairs:

- two-column page versus swapped columns
- same text blocks in reordered spatial positions
- table rows swapped
- numbered blocks arranged raster order versus semantic order
- spiral or path-following text layout
- same paragraph blocks with different crop-triggering sizes

Run baseline outputs first, then use these pairs for patching.

Expected outcome:

- Clean/corrupt pairs where the model output differs primarily in reading order, not content recognition.

## Run 8: Clean/Corrupt Activation Patching

Purpose: test whether query-token states causally transfer reading order.

Patch from clean image into corrupt image.

Primary target:

```text
model.qwen2_model.model.model.layers.{L}
query-token positions only
```

Controls:

- patch visual-token positions only
- patch SAM outputs
- patch projector outputs
- patch decoder layers
- patch random token positions with same count

Expected outcome if paper claim is right:

- Patching query-token states from clean into corrupt should move corrupt output toward clean reading order.
- Query patching should be more reading-order-specific than SAM or decoder controls.

## Run 9: Attention Pattern Inspection

Purpose: inspect whether query tokens attend to visual regions in a semantic reading sequence.

Question:

```text
query token i -> visual token j
```

Look for:

- title-to-body-to-table/caption progression
- layerwise emergence of ordered attention
- differences between local and global views
- relation between attention maxima and document layout regions

Caveat:

- `model.qwen2_model` currently hardcodes `attn_implementation="sdpa"`, which makes attention probabilities harder to extract.
- If using eager attention or monkeypatching, first verify output drift against the original SDPA path.

Expected outcome:

- Attention maps may provide supportive evidence, but causal claims still require ablation/patching.

## Run 10: Crop-Order Controls

Purpose: ensure apparent causal flow is not mostly an artifact of local/global crop sequencing.

Tests:

- global-only pages
- pages with local plus global views
- local-query-output ablation only
- global-query-output ablation only
- local-query permutation only
- global-query permutation only

Important anomaly to watch:

- The remote code appears to insert placeholder image tokens in one order but concatenate local/global embeddings in another order. This may affect how the decoder consumes visual tokens.

Expected outcome:

- Clear separation of local-view and global-view contribution to reading order.

## Run 11: Decoder Contribution Controls

Purpose: distinguish encoder-side reordering from decoder autoregressive reasoning.

Compare interventions at:

- Qwen2 visual encoder query outputs
- projector outputs
- DeepSeek decoder layer 0
- DeepSeek decoder layer 11

Expected outcome if encoder causal queries are the main source:

- Query-output interventions should be strongly reading-order-specific.
- Decoder interventions may affect fluency and generation broadly, but should be less specific to visual reading order.

## Evidence Standard

Evidence supporting the paper claim should include most of the following:

- Query-token interventions hurt reading order more than visual-token or decoder controls.
- Layer sweep shows reading-order information emerging inside `model.qwen2_model`.
- Clean/corrupt query-token patching transfers reading order.
- Query-output permutation disrupts layout order even when text remains visible.
- Attention maps, if extracted, show query tokens tracking semantically ordered regions.

The claim weakens if:

- Decoder-layer patching explains most reading-order behavior.
- Projector output is insensitive to query order.
- Query-token permutation barely changes output.
- SAM/global/local crop order explains most changes.
- Clean/corrupt query patches do not transfer reading order.

## Recommended Immediate Next Steps

1. Implement call-preserving activation hooks.
2. Implement final query-output ablation/permutation before `model.projector`.
3. Build a small synthetic clean/corrupt reading-order image set.
4. Run the ablation/permutation tests on the synthetic set before scaling to OmniDocBench.
