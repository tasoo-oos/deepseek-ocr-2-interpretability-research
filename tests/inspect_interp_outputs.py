#!/usr/bin/env python3
"""Diagnostic script that prints actual intermediate values from all interp modules."""

import torch
import math
from typing import cast, Any, Optional

# ======================================================================
# 1. PROJECTOR ANALYSIS
# ======================================================================
print("=" * 70)
print("1. PROJECTOR ANALYSIS - SVD, effective rank, logit lens")
print("=" * 70)

from addict import Dict
from src.models.projector import MlpProjector
from src.analysis.projector_analysis import ProjectorAnalyzer

weight = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
proj = MlpProjector(Dict(projector_type="linear", input_dim=3, n_embed=3))
with torch.no_grad():
    proj.layers.weight.copy_(weight)
    proj.layers.bias.zero_()

analyzer = ProjectorAnalyzer(proj)
svd = analyzer.compute_svd()
print(f"  Singular values: {svd.singular_values}")
print(f"  Expected:        [5.0, 3.0, 1.0]")
print(f"  Explained variance ratio: {svd.explained_variance_ratio}")
print(f"  Expected ratio: [{25 / 35:.4f}, {9 / 35:.4f}, {1 / 35:.4f}]")
print(f"  Effective rank (0.95): {analyzer.effective_rank(0.95)} (expect 2)")
print(f"  Effective rank (0.999): {analyzer.effective_rank(0.999)} (expect 3)")

weight2 = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
bias2 = torch.tensor([0.5, -0.5])
proj2 = MlpProjector(Dict(projector_type="linear", input_dim=2, n_embed=2))
with torch.no_grad():
    proj2.layers.weight.copy_(weight2)
    proj2.layers.bias.copy_(bias2)

analyzer2 = ProjectorAnalyzer(proj2)
visual_features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
unembedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
logits = analyzer2.logit_lens(visual_features, unembedding)
print(f"  Logit lens output:\n{logits}")
print(f"  Expected argmax per row: [0, 1], got: {logits.argmax(dim=-1).tolist()}")

# ======================================================================
# 2. QUERY ANALYSIS
# ======================================================================
print()
print("=" * 70)
print("2. QUERY ANALYSIS - bank geometry, cross-res similarity, ablation")
print("=" * 70)

from src.analysis.query_analysis import QuerySpecializationAnalyzer
import torch.nn as nn


class FakeQueryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_768 = nn.Embedding(3, 2)
        self.query_1024 = nn.Embedding(4, 2)
        with torch.no_grad():
            self.query_768.weight.copy_(
                torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
            )
            self.query_1024.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
            )

    def forward(self, x):
        query_bank = self.query_768.weight[: x.shape[1]].unsqueeze(0)
        return x + query_bank


fmodel = FakeQueryModel()
qa = QuerySpecializationAnalyzer(fmodel)

summary = qa.summarize_query_bank(768)
print(f"  Query bank 768 embeddings:\n{summary.embeddings}")
print(f"  Norms: {summary.norms}")
print(f"  Expected norms: [3.0, 2.0, sqrt(2)={math.sqrt(2):.4f}]")
print(f"  Cosine similarity matrix:\n{summary.cosine_similarity}")
print(f"  Diagonal (should be all 1s): {torch.diag(summary.cosine_similarity)}")
print(f"  Mean abs off-diagonal cosine: {summary.mean_abs_cosine:.4f}")
# Expected: q0=[3,0], q1=[0,2], q2=[1,1]
# cos(q0,q1)=0, cos(q0,q2)=3/3*sqrt(2)=0.7071, cos(q1,q2)=2/2*sqrt(2)=0.7071
# mean abs off-diag = (0 + 0.7071 + 0 + 0.7071 + 0.7071 + 0.7071) / 6 = 0.4714
print(f"  Expected mean abs off-diag cosine: 0.4714")

cross_sim = qa.cross_resolution_similarity()
print(f"  Cross-resolution similarity (3x4):\n{cross_sim}")
print(f"  [0,0] should be 1.0: {cross_sim[0, 0].item():.4f}")

inputs = torch.zeros(1, 3, 2)
result = qa.measure_query_group_contributions(
    inputs,
    query_groups={"dominant": [0], "weak": [2]},
    score_fn=lambda output: output[..., 0].sum().item(),
)
print(f"  Baseline score: {result['baseline']:.4f}")
print(f"  Delta dominant: {result['deltas']['dominant']:.4f}")
print(f"  Delta weak: {result['deltas']['weak']:.4f}")
print(f"  dominant > weak? {result['deltas']['dominant'] > result['deltas']['weak']}")
# The model does: output = input + query_bank
# input is zeros, so output[..., 0] = [3.0, 0.0, 1.0], sum = 4.0
# Ablating dominant (idx 0): output[...,0] at pos 0 becomes 0 -> sum = 1.0, delta = 3.0
# Ablating weak (idx 2): output[...,0] at pos 2 becomes 0 -> sum = 3.0, delta = 1.0
print(f"  Expected baseline=4.0, delta_dominant=3.0, delta_weak=1.0")

# ======================================================================
# 3. SPATIAL ANALYSIS
# ======================================================================
print()
print("=" * 70)
print("3. SPATIAL ANALYSIS - linear probe fit & predict")
print("=" * 70)

from src.analysis.spatial_analysis import LinearSpatialProbe

torch.manual_seed(0)
features = torch.randn(128, 4)
true_weight = torch.tensor([[2.0, -0.5], [0.25, 1.5], [-1.0, 0.75], [0.5, 0.25]])
true_bias = torch.tensor([0.2, -0.1])
targets = features @ true_weight + true_bias

probe = LinearSpatialProbe(l2_penalty=1e-6).fit(features, targets)
predictions = probe.predict(features)
metrics = probe.evaluate(features, targets)

print(f"  Learned weight:\n{probe.weight}")
print(f"  True weight:\n{true_weight}")
print(
    f"  Weight match (atol=1e-3): {torch.allclose(probe.weight, true_weight, atol=1e-3)}"
)
print(f"  Learned bias: {probe.bias}")
print(f"  True bias:    {true_bias}")
print(f"  MSE: {metrics.mse:.2e} (should be ~0)")
print(f"  R^2: {metrics.r2.tolist()} (should be ~[1.0, 1.0])")
print(f"  Max prediction error: {(predictions - targets).abs().max().item():.2e}")

# Test error before fitting
try:
    LinearSpatialProbe().predict(torch.zeros(1, 2))
    print("  ERROR: should have raised RuntimeError!")
except RuntimeError as e:
    print(f"  Pre-fit error correctly raised: {e}")

# ======================================================================
# 4. VIEW ANALYSIS
# ======================================================================
print()
print("=" * 70)
print("4. VIEW ANALYSIS - local vs global ablation")
print("=" * 70)

from src.analysis.view_analysis import ViewAblationAnalyzer


class FakeVisionModel:
    def get_multimodal_embeddings(self, pixel_values, images_crop, images_spatial_crop):
        local_signal = images_crop.float().sum().view(1, 1)
        global_signal = pixel_values.float().sum().view(1, 1)
        return [torch.cat([local_signal, global_signal], dim=-1)]


view_inputs = {
    "pixel_values": torch.ones(1, 1, 3, 2, 2),
    "images_crop": torch.full((1, 1, 2, 3, 2, 2), 2.0),
    "images_spatial_crop": torch.tensor([[[1, 2]]]),
}

va = ViewAblationAnalyzer(FakeVisionModel())
local_result = va.compare(
    view_inputs, score_fn=lambda output: output[0][..., 0].sum().item()
)
global_result = va.compare(
    view_inputs, score_fn=lambda output: output[0][..., 1].sum().item()
)

# local signal = images_crop.sum() = 1*1*2*3*2*2 * 2.0 = 48.0
# global signal = pixel_values.sum() = 1*1*3*2*2 * 1.0 = 12.0
print(f"  Scoring local channel (images_crop sum -> dim 0):")
print(f"    baseline:       {local_result.baseline_score:.4f} (expect 48.0)")
print(f"    local_ablated:  {local_result.local_ablated_score:.4f} (expect 0.0)")
print(f"    global_ablated: {local_result.global_ablated_score:.4f} (expect 48.0)")
print(f"    local_delta:    {local_result.local_delta:.4f} (expect 48.0)")
print(f"    global_delta:   {local_result.global_delta:.4f} (expect 0.0)")

print(f"  Scoring global channel (pixel_values sum -> dim 1):")
print(f"    baseline:       {global_result.baseline_score:.4f} (expect 12.0)")
print(f"    local_ablated:  {global_result.local_ablated_score:.4f} (expect 12.0)")
print(f"    global_ablated: {global_result.global_ablated_score:.4f} (expect 0.0)")
print(f"    local_delta:    {global_result.local_delta:.4f} (expect 0.0)")
print(f"    global_delta:   {global_result.global_delta:.4f} (expect 12.0)")

# Mutation test
orig_pv = view_inputs["pixel_values"].clone()
orig_ic = view_inputs["images_crop"].clone()
ViewAblationAnalyzer.ablate_local_views(view_inputs)
ViewAblationAnalyzer.ablate_global_views(view_inputs)
pv_mutated = not torch.equal(view_inputs["pixel_values"], orig_pv)
ic_mutated = not torch.equal(view_inputs["images_crop"], orig_ic)
print(f"  Inputs mutated? pixel_values={pv_mutated}, images_crop={ic_mutated}")
print(f"  (Both should be False)")

# ======================================================================
# 5. CIRCUIT DISCOVERY
# ======================================================================
print()
print("=" * 70)
print("5. CIRCUIT DISCOVERY - activation patching")
print("=" * 70)

from src.analysis.circuits import CircuitDiscovery


class FakeCircuitModel:
    def __init__(self):
        self.patched_position = None
        self.patched_value = None

    def __call__(self, signal):
        output = signal.clone()
        if self.patched_position is not None:
            output[:, self.patched_position, :] = self.patched_value
        return output


class FakeFeatureExtractor:
    def __init__(self, activations):
        self.activations = activations
        self.requested_layers = []

    def register_hooks(self, sam_layers=None, d2e_layers=None, projector=False):
        self.requested_layers = d2e_layers or sam_layers or []

    def extract(self, **kwargs):
        return {
            f"d2e_layer_{l}": self.activations[f"d2e_layer_{l}"]
            for l in self.requested_layers
        }

    def clear_hooks(self):
        self.requested_layers = []


class FakeInterventionManager:
    def __init__(self, model):
        self.model = model

    def patch_activation(self, layer, position, new_value, component="d2e"):
        self.model.patched_position = position
        self.model.patched_value = new_value

    def clear_interventions(self):
        self.model.patched_position = None
        self.model.patched_value = None


# Test 1: single position patching
model = FakeCircuitModel()
clean_acts = {"d2e_layer_3": torch.tensor([[[5.0], [1.0], [0.0]]])}
discovery = CircuitDiscovery(
    model=cast(Any, model),
    feature_extractor=cast(Any, FakeFeatureExtractor(clean_acts)),
    intervention_manager=cast(Any, FakeInterventionManager(model)),
)

score = discovery.activation_patching(
    clean_input={"signal": torch.zeros(1, 3, 1)},
    corrupted_input={"signal": torch.zeros(1, 3, 1)},
    layer=3,
    position=0,
    metric_fn=lambda output: cast(torch.Tensor, output)[:, 0, 0].sum().item(),
)
print(f"  Patching score at position 0: {score} (expect 5.0)")
# corrupted_input is zeros. After patching pos 0 with clean[pos 0]=5.0,
# output[:,0,0] = 5.0. metric_fn returns 5.0.

# Test 2: circuit ranking
model2 = FakeCircuitModel()
clean_acts2 = {"d2e_layer_0": torch.tensor([[[1.0], [2.0], [8.0], [3.0]]])}
discovery2 = CircuitDiscovery(
    model=cast(Any, model2),
    feature_extractor=cast(Any, FakeFeatureExtractor(clean_acts2)),
    intervention_manager=cast(Any, FakeInterventionManager(model2)),
)

result2 = discovery2.find_circuit_for_task(
    clean_input={"signal": torch.zeros(1, 4, 1)},
    corrupted_input={"signal": torch.zeros(1, 4, 1)},
    metric_fn=lambda output: cast(torch.Tensor, output).sum().item(),
    layers=[0],
    n_positions=4,
)
print(
    f"  Top critical position: {result2['critical_positions'][0]} (expect (0, 2) — value 8.0)"
)
print(f"  All ranked positions: {result2['critical_positions']}")

# ======================================================================
# 6. ATTENTION ENTROPY
# ======================================================================
print()
print("=" * 70)
print("6. ATTENTION ENTROPY")
print("=" * 70)

from src.visualization.utils import compute_attention_entropy

attn_uniform = torch.ones(1, 1, 4, 4) / 4
entropy = compute_attention_entropy(attn_uniform)
print(f"  Entropy shape: {entropy.shape} (expect [1, 1, 4])")
print(f"  Entropy values: {entropy}")
print(f"  Mean: {entropy.mean().item():.6f}, Expected log(4)={math.log(4):.6f}")

attn_peaked = torch.zeros(1, 1, 4, 4)
attn_peaked[0, 0, :, 0] = 1.0
entropy_peaked = compute_attention_entropy(attn_peaked)
print(f"  Peaked attention entropy: {entropy_peaked.mean().item():.6f} (expect ~0)")

# ======================================================================
# 7. REAL MODEL COMPONENTS — attention shapes and token type IDs
# ======================================================================
print()
print("=" * 70)
print("7. REAL MODEL - D2E attention extraction")
print("=" * 70)

from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
from src.analysis.attention_analysis import AttentionAnalyzer

d2e = build_qwen2_decoder_as_encoder(output_attentions=True)
d2e.eval()

sam_features = torch.zeros(1, 896, 16, 16)
with torch.no_grad():
    result = d2e(sam_features, output_attentions=True)

query_out, attentions, hidden_states, token_type_ids = result
print(f"  Query output shape: {query_out.shape} (expect [1, 256, 896])")
print(f"  Num attention layers: {len(attentions)} (expect 24)")
print(f"  Attention shape layer 0: {attentions[0].shape} (expect [1, 14, 512, 512])")
print(f"  Token type IDs shape: {token_type_ids.shape} (expect [1, 512])")
print(
    f"  Image tokens (type 0) count: {(token_type_ids == 0).sum().item()} (expect 256)"
)
print(
    f"  Query tokens (type 1) count: {(token_type_ids == 1).sum().item()} (expect 256)"
)

# Check attention mask pattern: image-to-query should be blocked
attn0 = attentions[0][0]  # [14, 512, 512]
i2q_region = attn0[:, :256, 256:]  # image attending to query
q2i_region = attn0[:, 256:, :256]  # query attending to image
i2i_region = attn0[:, :256, :256]  # image attending to image
q2q_region = attn0[:, 256:, 256:]  # query attending to query

print(f"  Layer 0 attention stats:")
print(
    f"    image->query (should be ~0, masked): max={i2q_region.max().item():.6f}, mean={i2q_region.mean().item():.6f}"
)
print(
    f"    query->image (cross-attention):       max={q2i_region.max().item():.6f}, mean={q2i_region.mean().item():.6f}"
)
print(
    f"    image->image (bidirectional):          max={i2i_region.max().item():.6f}, mean={i2i_region.mean().item():.6f}"
)
print(
    f"    query->query (causal):                 max={q2q_region.max().item():.6f}, mean={q2q_region.mean().item():.6f}"
)

# Check if causal mask is present in q2q region
# For causal: upper triangle should be 0
q2q_sample = q2q_region[0]  # head 0, [256, 256]
upper_tri = torch.triu(q2q_sample, diagonal=1)
print(f"    q2q upper triangle (should be 0): max={upper_tri.max().item():.6f}")

# ======================================================================
# 8. AttentionAnalyzer head specialization
# ======================================================================
print()
print("=" * 70)
print("8. ATTENTION ANALYZER - head specialization")
print("=" * 70)

aa = AttentionAnalyzer(d2e)
with torch.no_grad():
    patterns = aa.extract_attention_patterns(sam_features)

print(f"  Keys: {list(patterns.keys())}")
print(f"  Spatial size: {patterns['spatial_size']} (expect 16)")

spec = aa.analyze_head_specialization(patterns["attention_weights"], n_image_tokens=256)
print(f"  Entropy q2i shape: {spec['entropy_q2i'].shape} (expect [24, 14])")
print(f"  Entropy i2i shape: {spec['entropy_i2i'].shape} (expect [24, 14])")
print(f"  q2i_ratio shape: {spec['q2i_ratio'].shape} (expect [24, 14])")
print(
    f"  Sample entropy_q2i (layer 0, heads 0-4): {spec['entropy_q2i'][0, :5].tolist()}"
)
print(
    f"  Sample entropy_i2i (layer 0, heads 0-4): {spec['entropy_i2i'][0, :5].tolist()}"
)
print(f"  Sample q2i_ratio (layer 0, heads 0-4): {spec['q2i_ratio'][0, :5].tolist()}")
print(
    f"  q2i_ratio range: [{spec['q2i_ratio'].min():.4f}, {spec['q2i_ratio'].max():.4f}]"
)

# Important heads
top_heads = aa.find_important_heads(
    patterns["attention_weights"],
    n_image_tokens=256,
    metric="entropy",
    region="query_to_image",
    top_k=5,
)
print(f"  Top 5 most focused q2i heads (by neg entropy): {top_heads}")

print()
print("=" * 70)
print("ALL INTERPRETABILITY OUTPUTS INSPECTED")
print("=" * 70)
