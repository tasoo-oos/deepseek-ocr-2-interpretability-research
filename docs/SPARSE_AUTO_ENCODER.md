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

The current implementation is a small ReLU SAE:

```text
input activation x                [896]
    ->
linear encoder                    [n_features]
    ->
ReLU sparse code z                [n_features]
    ->
linear decoder (no bias)          [896]
    ->
reconstruction x_hat
```

Implementation details:

- Encoder: `nn.Linear(input_dim, n_features)`
- Nonlinearity: `ReLU`
- Decoder: `nn.Linear(n_features, input_dim, bias=False)`
- Decoder columns are renormalized to unit norm after each optimizer step
- Training loss: `MSE(reconstruction, activation) + l1_coeff * mean(code)`

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

### 1. The first SAE already separates more than raw query tokens

Layer-12 SAE training on 48 synthetic documents with a 768-feature dictionary
produced strong reconstruction and identifiable localized features.

Run summary:

- output: `output/sae_layer12_l1e2/summary.json`
- explained variance: `0.9782`
- MSE: `0.000390`
- mean active features per sample: `299.72`
- dead feature fraction: `0.2917`

Interpretation:

- the SAE reconstructs layer-12 query states very well
- but it is still too dense to claim clean monosemantic features

### 2. Query-band localization appears clearly

Examples from the query-localized feature list:

- feature `637`
  - frequency: `0.101`
  - query mean/std: `12.43 / 6.41`
  - attention y mean: `0.154`
  - interpretation: early-query, upper-page feature

- feature `327`
  - frequency: `0.063`
  - query mean/std: `235.37 / 8.07`
  - attention y mean: `0.076`
  - interpretation: very-late-query, upper-page feature

- feature `596`
  - frequency: `0.080`
  - query mean/std: `146.11 / 6.25`
  - attention y mean: `0.552`
  - interpretation: mid-query feature with lower-page bias

### 3. Spatial localization also appears

The spatially localized view found several upper-page features with low
attention-center dispersion, for example:

- feature `327`
  - `y mean 0.076`, `y std 0.041`
- feature `637`
  - `y mean 0.154`, `y std 0.037`
- feature `365`
  - `y mean 0.138`, `y std 0.039`

These are more specific than the original token-level analysis, which mostly
showed that query slots as a whole had a reading-order bias.

### 4. Feature ablations hit the expected downstream query bands

Single-feature ablations were run with:

- output: `output/sae_feature_ablation_l12/summary.json`
- layer: `12`
- mode: `subtract_decoder`

Although absolute single-feature effects are still small, the downstream query
positions affected by each feature align well with the feature's learned query
band:

- feature `637`:
  - learned query mean: about `12`
  - strongest affected queries: mostly `4-16`

- feature `596`:
  - learned query mean: about `146`
  - strongest affected queries: mostly `140-194`

- feature `227`:
  - learned query mean: about `218`
  - strongest affected queries: mostly `225-231`

- feature `327`:
  - learned query mean: about `235`
  - strongest affected queries: mostly `232-237`

- feature `386`:
  - learned query mean: about `244`
  - strongest affected queries: mostly `227-252`, concentrated late

This is important because it shows the SAE is not just discovering arbitrary
basis directions. The localized sparse features align with localized causal
effects in the D2E query stream.

## What We Get From SAE That Token Analysis Alone Misses

Without SAE:

- "late query tokens matter"
- "early query tokens read upper page regions"

With SAE:

- "this particular early, upper-page sparse feature perturbs early query slots"
- "this particular late, upper-page sparse feature perturbs late query slots"
- "different sparse features can live inside the same query token family"

So the SAE gives a more factorized view of the causal scratchpad used by the
query tokens.

## Current Limitations

The present SAE is useful, but not yet in a strong sparse regime.

Observed limitations:

- too many always-on or near-always-on features
- mean active feature count is still high
- single-feature ablations have small absolute effect sizes
- bottom/right localized features are much weaker than top/narrow-band features

This means the current SAE is better viewed as an initial decomposition layer
than as a final monosemantic feature dictionary.

## Next Recommended Improvements

The most promising follow-ups are:

1. Replace plain ReLU+L1 with a stricter sparse regime.
   - Top-K SAE is the strongest next candidate.

2. Normalize or whiten activations before SAE training.
   - This should reduce dominance by broad background directions.

3. Train on a larger and more diverse real-document dataset.
   - Current results are strong for controlled layouts but still synthetic.

4. Move from single-feature ablations to small feature-group ablations.
   - This may recover stronger causal effects while keeping interpretability.

5. Compare SAE behavior across layers.
   - especially layers 6, 12, and 18

## Practical Summary

The current SAE work supports three conclusions:

1. Query tokens contain separable internal substructure.
2. Some of that substructure is localized by query position and page region.
3. Those localized sparse features have matching localized causal effects when
   ablated inside D2E.

That is enough to justify continuing with SAE-based interpretability rather than
stopping at token-level analysis.
