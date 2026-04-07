"""
Attention-specific analysis tools for the D2E Visual Causal Flow mechanism.

Wraps Qwen2Decoder2Encoder to extract and analyze attention patterns.
"""

import torch
from typing import Dict, List, Optional, Tuple

from src.models.qwen2_d2e import Qwen2Decoder2Encoder
from src.visualization.utils import (
    compute_attention_entropy,
    extract_attention_regions,
    aggregate_heads,
)


class AttentionAnalyzer:
    """
    Extract and analyze attention patterns from the D2E model.

    Args:
        qwen2_model: A Qwen2Decoder2Encoder instance.
                     Pass output_attentions=True (or set it here) to enable extraction.
    """

    def __init__(self, qwen2_model: Qwen2Decoder2Encoder):
        self.model = qwen2_model

    def extract_attention_patterns(
        self,
        sam_features: torch.Tensor,
        layers: Optional[List[int]] = None,
    ) -> Dict:
        """
        Extract attention weights from specified D2E layers.

        Args:
            sam_features: SAM encoder output [B, 896, H, W]
            layers: Layer indices to return (None = all layers)

        Returns:
            dict with keys:
                attention_weights: List[[B, num_heads, seq, seq]] — one per layer
                token_type_ids:    [B, seq_len]
                spatial_size:      int (H or W of SAM feature map)
        """
        result = self.model(sam_features, output_attentions=True)
        if isinstance(result, tuple):
            query_outputs, attentions, hidden_states, token_type_ids = result
        else:
            raise RuntimeError(
                "output_attentions=True must be set on the Qwen2Decoder2Encoder."
            )

        if attentions is None:
            raise RuntimeError("No attentions returned. Make sure output_attentions=True.")

        spatial_size = sam_features.shape[-1]

        if layers is not None:
            attentions = [attentions[i] for i in layers]

        return {
            "attention_weights": list(attentions),
            "token_type_ids": token_type_ids,
            "spatial_size": spatial_size,
            "query_outputs": query_outputs,
        }

    def analyze_head_specialization(
        self,
        attention_weights: List[torch.Tensor],
        n_image_tokens: int,
    ) -> Dict:
        """
        Compute per-head statistics for all layers.

        Args:
            attention_weights: List[[B, H, S, S]] per layer
            n_image_tokens: Number of image tokens in the sequence

        Returns:
            dict with layers × heads matrices for:
                entropy_q2i, entropy_i2i, entropy_q2q
                q2i_ratio (fraction of attention on image tokens from query)
        """
        n_layers = len(attention_weights)
        n_heads = attention_weights[0].shape[1]

        entropy_q2i = torch.zeros(n_layers, n_heads)
        entropy_i2i = torch.zeros(n_layers, n_heads)
        entropy_q2q = torch.zeros(n_layers, n_heads)
        q2i_ratio = torch.zeros(n_layers, n_heads)

        for layer_idx, attn in enumerate(attention_weights):
            i2i, _, q2i, q2q = extract_attention_regions(
                attn, n_image_tokens, attn.shape[2] - n_image_tokens
            )

            for h in range(n_heads):
                entropy_q2i[layer_idx, h] = compute_attention_entropy(
                    q2i[:, h:h+1, :, :]
                ).mean().item()
                entropy_i2i[layer_idx, h] = compute_attention_entropy(
                    i2i[:, h:h+1, :, :]
                ).mean().item()
                entropy_q2q[layer_idx, h] = compute_attention_entropy(
                    q2q[:, h:h+1, :, :]
                ).mean().item()

                # Fraction of query attention that goes to image tokens
                q2i_sum = q2i[:, h].sum(dim=-1).mean()
                total_sum = attn[:, h, n_image_tokens:, :].sum(dim=-1).mean()
                q2i_ratio[layer_idx, h] = (q2i_sum / (total_sum + 1e-9)).item()

        return {
            "entropy_q2i": entropy_q2i,
            "entropy_i2i": entropy_i2i,
            "entropy_q2q": entropy_q2q,
            "q2i_ratio": q2i_ratio,
        }

    def find_important_heads(
        self,
        attention_weights: List[torch.Tensor],
        n_image_tokens: int,
        metric: str = "entropy",
        region: str = "query_to_image",
        top_k: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        Rank attention heads by importance.

        Args:
            attention_weights: List[[B, H, S, S]] per layer
            n_image_tokens: Number of image tokens
            metric: "entropy" (lower = more focused) or "magnitude"
            region: "query_to_image", "image_to_image", "query_to_query"
            top_k: Number of top (layer, head) pairs to return

        Returns:
            List of (layer, head) tuples sorted by importance (descending).
        """
        scores: List[Tuple[float, int, int]] = []

        for layer_idx, attn in enumerate(attention_weights):
            n_heads = attn.shape[1]
            i2i, _, q2i, q2q = extract_attention_regions(
                attn, n_image_tokens, attn.shape[2] - n_image_tokens
            )

            region_map = {
                "image_to_image": i2i,
                "query_to_image": q2i,
                "query_to_query": q2q,
            }
            region_attn = region_map[region]

            for h in range(n_heads):
                head_attn = region_attn[:, h:h+1, :, :]
                if metric == "entropy":
                    # Lower entropy = more focused = more important
                    score = -compute_attention_entropy(head_attn).mean().item()
                elif metric == "magnitude":
                    score = head_attn.max().item()
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                scores.append((score, layer_idx, h))

        scores.sort(reverse=True)
        return [(layer, head) for _, layer, head in scores[:top_k]]
