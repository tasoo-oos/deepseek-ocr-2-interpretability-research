# Sparse Autoencoder Research

## Overview

This document describes the sparse autoencoder (SAE) tooling added for
mechanistic interpretability of DeepSeek-OCR-2's visual causal-flow stack.

The goal is to go beyond token-level analysis of D2E query states and identify
sub-features inside those query vectors. Instead of asking only which query
token matters, the SAE asks which sparse latent components inside a query state
matter.

Current code lives in:

- `src/analysis/sparse_autoencoder.py`
- `scripts/train_sae.py`
- `scripts/run_sae_feature_ablation.py`

## Why Use An SAE Here

The D2E query tokens already show strong sequential structure:

- early queries attend earlier page regions
- later queries influence later queries through the causal query-to-query path
- final query states linearly encode spatial position

That still leaves an open question: what is mixed together inside each 896-dim
query state?

The SAE is meant to separate those mixed signals into sparse latent features
that can be:

- summarized by query index and attended page position
- ranked by specialization
- ablated causally inside a D2E layer

## Current SAE Architecture

The implementation now supports two encoder sparsification modes over the same
basic linear dictionary:

```text
input activation x                [896]
    ->
linear encoder                    [n_features]
    ->
pre-activation                    [n_features]
    ->
sparsification:
    - ReLU
    - ReLU + keep top-k positives
    ->
sparse code z                     [n_features]
    ->
linear decoder (no bias)          [896]
    ->
reconstruction x_hat
```

Implementation details:

- Encoder: `nn.Linear(input_dim, n_features)`
- Decoder: `nn.Linear(n_features, input_dim, bias=False)`
- Decoder columns are renormalized to unit norm after each optimizer step
- `activation_mode="relu"` uses plain `ReLU`
- `activation_mode="topk"` keeps only the largest `k` positive activations per
  sample
- Training loss: `MSE(reconstruction, activation) + l1_coeff * mean(code)`
- In practice, the Top-K path is now the better default for causal analysis

Relevant implementation:

- `SparseAutoencoder`
- `SparseAutoencoderTrainer`
- `SparseAutoencoderAnalyzer`
- `ablate_sparse_features()`

## Training Target

The default training target is D2E query hidden states from a chosen layer.

Current experiments focused on:

- layer 12 query states
- global-view only stimuli for controlled analysis
- synthetic but structured document layouts:
  - single column
  - two column
  - header/body/footer
  - table
  - zigzag layouts

The training script also records metadata for every query activation:

- query index
- attention center x
- attention center y
- stimulus/layout label

This metadata is used to describe each learned SAE feature.

## Analysis Outputs

The analyzer computes:

- activation frequency per feature
- mean and max feature activation
- weighted mean/std of query index
- weighted mean/std of attention center x/y
- top activating examples

The training script reports three views of the learned dictionary:

1. top active features
2. query-localized features
3. spatially localized features

This distinction matters because the most active features are often broad
background features rather than the most interpretable ones.

## Feature-Level Intervention

The repo now supports intervening on SAE features inside a D2E layer.

Current ablation modes:

- `subtract_decoder`
  - encode the query hidden state
  - isolate selected sparse features
  - subtract their decoder contribution from the original hidden state
- `reconstruct`
  - encode the query hidden state
  - zero selected sparse codes
  - replace the hidden state with the reconstruction from the remaining codes

This is wired into `InterventionManager.ablate_sae_features_in_query_states(...)`.

## Current Findings

### 1. ReLU/L1 SAE gave the first usable decomposition, but stayed too dense

Layer-12 SAE training on 48 synthetic documents with a 768-feature dictionary
produced strong reconstruction and identifiable localized features.

Run summary:

- output: `output/sae_layer12_l1e2/summary.json`
- explained variance: `0.9782`
- MSE: `0.000390`
- mean active features per sample: `299.72`
- dead feature fraction: `0.2917`

Interpretation:

- the dictionary clearly separates internal substructure in query states
- but it is still too dense for strong feature-level interventions

### 2. Top-K SAE trades some reconstruction for much cleaner sparsity

The stricter Top-K run at the same layer and dictionary size materially improved
sparsity quality.

Run summary:

- output: `output/sae_layer12_topk64/summary.json`
- activation mode: `topk`
- `k = 64`
- explained variance: `0.9581`
- MSE: `0.000750`
- mean active features per sample: `64.00`
- dead feature fraction: `0.7266`

Interpretation:

- Top-K loses some reconstruction quality relative to ReLU/L1
- but it sharply reduces feature overlap and yields much cleaner localized bands
- for intervention work, this is the better tradeoff

### 3. Query-band and spatial localization are both clearer under Top-K

Examples from the Top-K query-localized and spatially localized views:

- feature `417`
  - query mean/std: `13.91 / 8.27`
  - attention y mean: `0.162`
  - interpretation: early-query, upper-page feature

- feature `399`
  - query mean/std: `242.81 / 15.20`
  - attention y mean/std: `0.076 / 0.034`
  - interpretation: very-late-query, sharply top-localized feature

- feature `408`
  - query mean/std: `141.51 / 10.85`
  - attention y mean: `0.563`
  - interpretation: mid-query, lower-page feature

- feature `722`
  - query mean/std: `150.28 / 15.03`
  - attention y mean: `0.580`
  - interpretation: mid-query, lower-page feature

This is a stronger factorization than the original token-level analysis. The
token story said that query slots as a whole follow reading order; the Top-K
SAE shows that several narrower spatial/query-band variables live inside that
same query stream.

### 4. Single-feature Top-K ablations are materially stronger than ReLU/L1

Single-feature ablations were run with:

- ReLU/L1 output: `output/sae_feature_ablation_l12/summary.json`
- Top-K output: `output/sae_feature_ablation_topk64_l12/summary.json`
- layer: `12`
- mode: `subtract_decoder`

Compared with the original dense SAE, Top-K increased the average intervention
strength:

- mean query cosine drop:
  - ReLU/L1: `6.23e-06`
  - Top-K: `7.53e-05`
- mean attention shift:
  - ReLU/L1: `1.96e-04`
  - Top-K: `5.09e-04`

The Top-K single-feature effects remain modest in absolute terms, but they are
no longer near the numerical floor. They also remain aligned with the feature's
learned location:

- early/top features `417`, `324`, `377`, `309` affect early queries
- feature `706` affects a mid-early band around query `90`
- lower-page features such as `408` and `722` remain available for grouped
  interventions

### 5. Small Top-K feature groups produce a stronger causal regime

Grouped feature ablations were run with:

- output: `output/sae_feature_ablation_topk64_groups_l12/summary.json`
- grouping: `localized_bands`
- group size: `4`

Three localized groups were especially informative:

- early/top group: `417, 324, 377, 309`
  - mean query drop: `0.00053`
  - mean attention shift: `0.00152`
  - weighted affected query mean: `21.27`
  - top affected queries across layouts: mostly `3-18`

- mid/lower group: `408, 722, 735, 579`
  - mean query drop: `0.00055`
  - mean attention shift: `0.00223`
  - weighted affected query mean: `159.52`
  - top affected queries across layouts: mostly `141-179`

- late/top group: `399, 705, 51, 325`
  - mean query drop: `0.00335`
  - mean attention shift: `0.00300`
  - weighted affected query mean: `241.10`
  - top affected queries across layouts: mostly `235-254`

This is the strongest SAE result so far. It shows that a small set of localized
sparse features can be removed together and produce a concentrated downstream
effect on the matching query band.

## What We Get From SAE That Token Analysis Alone Misses

Without SAE:

- "late query tokens matter"
- "early query tokens read upper page regions"

With SAE:

- "this particular early, upper-page sparse feature perturbs early query slots"
- "this particular late, upper-page sparse feature perturbs late query slots"
- "a small late/top feature group drives the late-query band much more strongly
  than any single token ablation suggested"
- "different sparse features can live inside the same query token family"

So the SAE gives a more factorized view of the causal scratchpad used by the
query tokens.

## Current Limitations

The SAE work is now in a useful causal regime, but it is still not a final
dictionary.

Observed limitations:

- all reported runs still use synthetic controlled layouts rather than a real
  benchmark corpus
- Top-K improves separation, but the dictionary is highly selective and many
  features stay dead
- bottom/right and more semantic layout features remain weaker than the
  strongest top/narrow-band spatial features
- grouped ablations are clearly stronger than single-feature ablations, which
  means many variables are still split across several related features

So the current result is best understood as a good sparse decomposition of the
query scratchpad, not yet a clean monosemantic atlas of the whole model.

## Next Recommended Improvements

The most promising follow-ups are:

1. Treat Top-K as the baseline SAE rather than the ReLU/L1 model.
   - it gives the best current intervention tradeoff

2. Normalize or whiten activations before SAE training.
   - this should reduce dominance by broad background directions

3. Train on a larger and more diverse real-document dataset.
   - current results are mechanistically strong but still synthetic

4. Extend grouped ablations into feature steering and feature patching.
   - the grouped Top-K result is strong enough to justify that next step

5. Compare SAE behavior across layers.
   - especially layers `6`, `12`, and `18`

## Practical Summary

The current SAE work supports four conclusions:

1. Query tokens contain separable internal substructure.
2. Some of that substructure is localized by query position and page region.
3. Top-K sparsification exposes that structure more cleanly than the original
   dense ReLU/L1 SAE.
4. Small localized feature groups produce concentrated causal effects on the
   matching downstream query bands.

That is enough to justify continuing with SAE-based interpretability rather than
stopping at token-level analysis.
