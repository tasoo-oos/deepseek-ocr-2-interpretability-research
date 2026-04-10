"""
Utility functions for attention and feature visualization.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image


def normalize_attention(attention: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize attention weights to sum to 1 along the given dimension."""
    return attention / (attention.sum(dim=dim, keepdim=True) + 1e-9)


def attention_to_numpy(attention: torch.Tensor) -> np.ndarray:
    """Convert attention tensor to float32 numpy array."""
    if isinstance(attention, torch.Tensor):
        return attention.detach().cpu().float().numpy()
    return np.asarray(attention, dtype=np.float32)


def compute_attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of an attention distribution.

    Args:
        attention: [B, H, Q, K] — attention probabilities

    Returns:
        [B, H, Q] entropy per query position
    """
    eps = 1e-9
    attention = attention.clamp(min=eps)
    return -torch.sum(attention * torch.log(attention), dim=-1)


def compute_attention_distance(
    attention: torch.Tensor,
    spatial_size: int,
) -> torch.Tensor:
    """
    Compute average attended distance for spatially arranged tokens.

    Args:
        attention:    [B, H, n_tokens, n_tokens]
        spatial_size: Spatial grid side length

    Returns:
        [B, H, n_tokens] average Euclidean distance to attended positions
    """
    n_tokens = spatial_size * spatial_size
    device = attention.device

    positions = torch.arange(n_tokens, device=device)
    row = positions // spatial_size
    col = positions % spatial_size

    row_diff = row.unsqueeze(0).float() - row.unsqueeze(1).float()
    col_diff = col.unsqueeze(0).float() - col.unsqueeze(1).float()
    distances = torch.sqrt(row_diff ** 2 + col_diff ** 2)

    return torch.einsum('bhqk,qk->bhq', attention[:, :, :n_tokens, :n_tokens], distances)


def reshape_attention_to_spatial(
    attention: torch.Tensor,
    spatial_size: int,
) -> torch.Tensor:
    """
    Reshape flattened attention sequence to a spatial grid.

    Args:
        attention: [..., n_tokens]

    Returns:
        [..., spatial_size, spatial_size]
    """
    *prefix, n_tokens = attention.shape
    assert n_tokens == spatial_size ** 2, \
        f"Expected {spatial_size ** 2} tokens, got {n_tokens}"
    return attention.view(*prefix, spatial_size, spatial_size)


def extract_attention_regions(
    attention: torch.Tensor,
    n_image_tokens: int,
    n_query_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the attention matrix into its four Visual Causal Flow regions.

    Returns:
        (image_to_image, image_to_query, query_to_image, query_to_query)
        each of shape [B, H, rows, cols]
    """
    i = n_image_tokens
    image_to_image = attention[:, :, :i, :i]
    image_to_query = attention[:, :, :i, i:]
    query_to_image = attention[:, :, i:, :i]
    query_to_query = attention[:, :, i:, i:]
    return image_to_image, image_to_query, query_to_image, query_to_query


def aggregate_heads(
    attention: torch.Tensor,
    method: str = "mean",
    head_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Aggregate attention across heads.

    Args:
        attention:    [B, H, S, S]
        method:       "mean" | "max" | "specific"
        head_indices: Used only for method="specific"

    Returns:
        [B, S, S] (mean/max) or [B, len(head_indices), S, S] (specific)
    """
    if method == "mean":
        return attention.mean(dim=1)
    elif method == "max":
        return attention.max(dim=1)[0]
    elif method == "specific":
        if head_indices is None:
            raise ValueError("head_indices required for method='specific'")
        return attention[:, head_indices]
    else:
        raise ValueError(f"Unknown method: {method!r}")


def get_top_k_attended_positions(
    attention: torch.Tensor,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return top-k key positions by attention weight for each query.

    Args:
        attention: [B, H, Q, K]
        k:         Number of top keys

    Returns:
        (values [B, H, Q, k], indices [B, H, Q, k])
    """
    return torch.topk(attention, k=k, dim=-1)


def create_attention_mask_visualization(
    n_image_tokens: int,
    n_query_tokens: int,
) -> np.ndarray:
    """
    Create the expected Visual Causal Flow mask pattern.

    Returns:
        [total_seq, total_seq] float32 array:
        1.0 = can attend, 0.0 = blocked
    """
    total = n_image_tokens + n_query_tokens
    mask = np.zeros((total, total), dtype=np.float32)
    mask[:n_image_tokens, :n_image_tokens] = 1.0
    mask[n_image_tokens:, :n_image_tokens] = 1.0
    for i in range(n_query_tokens):
        mask[n_image_tokens + i, n_image_tokens:n_image_tokens + i + 1] = 1.0
    return mask


def overlay_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.6,
    colormap: str = "viridis",
) -> Image.Image:
    """
    Blend an attention heatmap over a PIL image.

    Args:
        image:         Input PIL image.
        attention_map: [H, W] attention weights (will be resized to image size).
        alpha:         Heatmap opacity (0.0 = image only, 1.0 = heatmap only).
        colormap:      Matplotlib colormap name.

    Returns:
        Blended PIL Image.
    """
    import matplotlib

    img_w, img_h = image.size
    attn_resized = np.array(
        Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
            (img_w, img_h), Image.BILINEAR
        )
    ) / 255.0

    cmap = matplotlib.colormaps.get_cmap(colormap)
    attn_colored = (cmap(attn_resized)[:, :, :3] * 255).astype(np.uint8)
    img_array = np.array(image.convert("RGB"))
    blended = (alpha * attn_colored + (1 - alpha) * img_array).astype(np.uint8)
    return Image.fromarray(blended)


def position_to_spatial_coords(position: int, spatial_size: int) -> Tuple[int, int]:
    """Convert flat position index to (row, col)."""
    return position // spatial_size, position % spatial_size


def spatial_coords_to_position(row: int, col: int, spatial_size: int) -> int:
    """Convert (row, col) to flat position index."""
    return row * spatial_size + col
