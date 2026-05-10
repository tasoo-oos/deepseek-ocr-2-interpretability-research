from __future__ import annotations

import torch


def zero_ablation(tensor, token_indices=None):
    result = tensor.clone()
    if token_indices is None:
        return torch.zeros_like(result)
    result[:, token_indices, :] = 0
    return result


def mean_ablation(tensor, token_indices=None):
    result = tensor.clone()
    mean = result.mean(dim=1, keepdim=True)
    if token_indices is None:
        return mean.expand_as(result)
    result[:, token_indices, :] = mean
    return result
