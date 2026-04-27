# DeepSeek-OCR-2 Research Audit

This note is an implementation-grounded audit of the repo's internal mechanisms.
It combines:

- direct source inspection under `src/`
- the repo's passing test suite
- saved experiment summaries under `output/`

The goal is to separate what is confirmed by code from what is only implied by
the paper or older docs.

## Confirmed Internal Mechanisms

### 1. Visual Causal Flow is implemented as a custom 4-D mask inside Qwen2

The D2E path is not a separate transformer implementation. It subclasses
`Qwen2Model` and overrides `_update_causal_mask(...)` to inject a hand-built
mask based on `token_type_ids`.

Confirmed in:

- `src/models/qwen2_d2e.py`

The resulting pattern is:

- image -> image: fully bidirectional
- image -> query: blocked
- query -> image: full cross-attention
- query -> query: causal lower triangle

### 2. "Reordering" is implicit in the query stream, not an explicit token permutation

The paper describes dynamic visual token reordering, but this implementation
does not contain an explicit sort, gather, or permutation over image tokens.
The SAM features are flattened in raster order, concatenated with learned query
embeddings, and the final representation is taken from the query half of the
sequence.

Confirmed in:

- `src/models/qwen2_d2e.py`

Inference:

- the sequential structure is represented by learned query states that read the
  full image-token set under a causal query-to-query constraint
- in this repo, "reordering" is therefore implemented as a learned causal
  extraction process rather than a literal token shuffle

### 3. Query count matches image-token count at each resolution

The D2E uses one learned query per image token:

- `query_768`: 144 queries for `12 x 12` local features
- `query_1024`: 256 queries for `16 x 16` global features

Confirmed in:

- `src/models/qwen2_d2e.py`
- `tests/test_models.py`
- `tests/test_attention.py`

This is a strong architectural clue: the query stream is slot-aligned in count
with the source spatial grid even though it is allowed to develop its own causal
ordering.

### 4. Local and global views share the same vision stack

Local crops and the global padded image are processed by the same:

- SAM encoder
- D2E module
- projector

The merge happens only after projection to 1280 dimensions.

Confirmed in:

- `src/models/deepseek_ocr.py`

### 5. The multimodal embedding order is `[local, global, separator]`

The repo currently concatenates:

- all local projected tokens
- all global projected tokens
- one learned `view_seperator` token

If no local crops exist, the output becomes `[global, separator]`.

Confirmed in:

- `src/models/deepseek_ocr.py`

This matters for any downstream intervention or logit-lens analysis over the
multimodal sequence.

## Novel Findings From Existing Repo Experiments

These findings are backed by saved experiment outputs already present in the
repo, not by fresh reruns in this audit.

### 1. Full OmniDocBench sweep confirms query order tracks layout order

From `output/query_trace_mask_ablation_v2_full/summary.md`:

- the full sweep covers `1355` OmniDocBench pages
- local-path inside-layout Spearman is `0.6468`; global-path inside-layout
  Spearman is `0.4809`
- local-path inside coverage is `0.9418`; global-path inside coverage is
  `0.7960`
- local ordering remains stronger across single-column, double-column,
  three-column, and mixed layouts

Interpretation:

- the local crop path gives the strongest evidence for a learned reading-order
  trace
- the global path still carries order information, but its assignments are less
  clean on dense or multi-column pages

### 2. Mask and order ablations separate visual access from causal sequencing

From `output/query_trace_mask_ablation_v2_full/summary.md`:

- blocking query-to-visual attention collapses inside coverage to `0.0000`
- removing query-to-query causality shifts representations more on local tokens
  than global tokens: local index cosine shift `0.1589` vs global `0.0739`
- reversing final queries leaves unordered similarity near zero but destroys
  position-aligned order: global index cosine shift `0.8333`, local `0.7625`
- shuffling final queries drives inversion rates to random-order levels near
  `0.5`

Interpretation:

- visual access is necessary for layout grounding
- query-to-query causality is not the only source of layout assignment, but it
  helps stabilize the local-path trajectory
- final query position matters; the representation is not just an unordered bag
  of visual summaries

### 3. Query index tracks vertical reading order much more strongly than horizontal order

From `output/causal_token_research/summary.md`:

- `corr(query, y)` ranges from `0.162` to `0.343`
- `corr(query, x)` ranges from `-0.214` to `-0.069`
- late queries consistently attend lower regions than early queries

Interpretation:

- the causal query stream behaves more like a top-to-bottom traversal than a
  left-to-right sweep
- column handling appears secondary to vertical progression

### 4. Mid-layer query ablations mostly affect later queries, not earlier ones

From `output/causal_token_research/summary.md`:

- ablating layer 12, queries `96:128`, gives near-zero prefix damage
- suffix cosine drop is non-trivial across all four synthetic layouts
- suffix/prefix ratios are extremely large because prefix damage is near zero

Interpretation:

- the causal dependency is directional in the expected way
- later query states depend on earlier query states much more than the reverse

### 5. Final query states linearly encode spatial position very well

From `output/causal_token_research/summary.md`:

- spatial probe MSE: `0.001018`
- `R^2_x = 0.9567`
- `R^2_y = 0.9981`

Interpretation:

- by the end of D2E, spatial location is almost linearly recoverable
- vertical position is especially explicit in the final query representation

### 6. Query banks are fairly decorrelated, but not independent across resolutions

From `output/causal_token_research/summary.md`:

- `query_1024` mean absolute cosine: `0.0626`
- `query_768` mean absolute cosine: `0.0509`
- max cross-resolution cosine: `0.8594`

Interpretation:

- the learned queries are broadly diverse within each bank
- some 768 and 1024 query slots still align strongly, suggesting partial
  reuse of ordering structure across resolutions

### 7. Top-K SAE is a better intervention substrate than dense ReLU/L1 SAE

From:

- `output/sae_layer12_l1e2/summary.md`
- `output/sae_layer12_topk64/summary.md`
- `output/sae_feature_ablation_l12/summary.md`
- `output/sae_feature_ablation_topk64_l12/summary.md`

Observed tradeoff:

- ReLU/L1 SAE reconstructs better: explained variance `0.9782`
- Top-K SAE is much sparser: `64.00` active features per sample vs `299.72`
- Top-K feature ablations produce materially larger query and attention shifts

Interpretation:

- the denser SAE mixes too many subfeatures per sample for clean causal use
- Top-K loses some fidelity but yields a much better mechanistic handle

### 8. Layer-12 SAE features organize into interpretable reading bands

From `output/sae_layer12_topk64/summary.md` and
`output/sae_feature_ablation_topk64_groups_l12/summary.md`:

- early/top band: features `417, 324, 377, 309`
- mid/lower band: features `408, 722, 735, 579`
- late/top band: features `399, 705, 51, 325`

The strongest grouped intervention is the late/top band:

- mean query drop: `0.0033`
- mean attention shift: `0.0030`
- weighted query effect: `241.10`

Interpretation:

- the query stream is not just globally ordered
- it factorizes into narrower subcircuits tied to both query time and page
  region

## Documentation Drift Found During Audit

### Confirmed stale docs before this audit

- `README.md` listed the multimodal concat order incorrectly
- `README.md` referred to a vendored `DeepSeek-OCR2-master/` tree that is not
  present in this repo
- `docs/STRUCTURE.md` described the old upstream directory layout instead of the
  current repo
- `docs/VISUALIZATION.md` referred to old `visualization/...` files and a CLI
  path that do not exist here

## Open Questions

These are not resolved by the current code audit:

- how closely the cleaned reimplementation matches every detail of DeepSeek's
  original training-time model path
- whether the strongest causal reading-order effects persist on real document
  distributions, not only the synthetic stimuli used in the saved experiments
- how much of the late-query specialization is due to query embeddings versus
  emergent transformer dynamics across layers
