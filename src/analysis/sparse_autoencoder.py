"""Sparse autoencoder tooling for decomposing D2E query states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseAutoencoderMetrics:
    """Aggregate reconstruction and sparsity metrics."""

    mse: float
    l1: float
    l0: float
    explained_variance: float


@dataclass(frozen=True)
class SparseFeatureSummary:
    """Summary statistics for one SAE feature."""

    feature_index: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    weighted_query_index_mean: Optional[float]
    weighted_query_index_std: Optional[float]
    weighted_attention_x_mean: Optional[float]
    weighted_attention_x_std: Optional[float]
    weighted_attention_y_mean: Optional[float]
    weighted_attention_y_std: Optional[float]
    top_examples: List[Dict[str, object]]


@dataclass(frozen=True)
class SparseAutoencoderSummary:
    """Model-level summary after training or evaluation."""

    metrics: SparseAutoencoderMetrics
    dead_feature_fraction: float
    mean_active_features_per_sample: float
    feature_summaries: List[SparseFeatureSummary]


class SparseAutoencoder(nn.Module):
    """A small ReLU sparse autoencoder with unit-normalized decoder columns."""

    def __init__(self, input_dim: int, n_features: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5 ** 0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / max(1, self.n_features) ** 0.5)
        self.normalize_decoder()

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(x)
        reconstruction = self.decode(codes)
        return reconstruction, codes


class SparseAutoencoderTrainer:
    """Train and evaluate a sparse autoencoder on activation vectors."""

    def __init__(
        self,
        model: SparseAutoencoder,
        *,
        lr: float = 3e-4,
        l1_coeff: float = 1e-3,
        batch_size: int = 1024,
        steps: int = 500,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.model = model.to(device)
        self.lr = lr
        self.l1_coeff = l1_coeff
        self.batch_size = batch_size
        self.steps = steps
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def fit(self, activations: torch.Tensor) -> SparseAutoencoderMetrics:
        data = activations.to(self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.steps):
            indices = torch.randint(
                low=0,
                high=data.shape[0],
                size=(min(self.batch_size, data.shape[0]),),
                generator=self.generator,
            )
            batch = data.index_select(0, indices.to(data.device))
            reconstruction, codes = self.model(batch)
            mse = F.mse_loss(reconstruction, batch)
            l1 = codes.mean()
            loss = mse + self.l1_coeff * l1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.model.normalize_decoder()

        return self.evaluate(data)

    def evaluate(self, activations: torch.Tensor) -> SparseAutoencoderMetrics:
        data = activations.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            reconstruction, codes = self.model(data)
            mse = F.mse_loss(reconstruction, data)
            l1 = codes.mean()
            l0 = (codes > 0).float().sum(dim=-1).mean()
            variance = torch.var(data, dim=0, unbiased=False).mean().clamp_min(1e-12)
            explained_variance = 1.0 - mse / variance
        return SparseAutoencoderMetrics(
            mse=float(mse.item()),
            l1=float(l1.item()),
            l0=float(l0.item()),
            explained_variance=float(explained_variance.item()),
        )


class SparseAutoencoderAnalyzer:
    """Summarize sparse feature usage with optional query metadata."""

    def __init__(self, model: SparseAutoencoder):
        self.model = model

    def encode(self, activations: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        target_device = device or next(self.model.parameters()).device
        with torch.no_grad():
            codes = self.model.encode(activations.to(target_device, dtype=torch.float32))
        return codes.detach().cpu()

    def summarize(
        self,
        activations: torch.Tensor,
        metadata: Optional[Mapping[str, Sequence[object] | torch.Tensor]] = None,
        *,
        top_k_examples: int = 5,
        min_activation_frequency: float = 0.0,
    ) -> SparseAutoencoderSummary:
        codes = self.encode(activations)
        metrics = SparseAutoencoderTrainer(
            self.model,
            device=str(next(self.model.parameters()).device),
        ).evaluate(activations)

        frequency = (codes > 0).float().mean(dim=0)
        dead_feature_fraction = float((frequency == 0).float().mean().item())
        mean_active_features_per_sample = float((codes > 0).float().sum(dim=-1).mean().item())

        query_index = _maybe_tensor(metadata, "query_index")
        attn_x = _maybe_tensor(metadata, "attention_x")
        attn_y = _maybe_tensor(metadata, "attention_y")
        stimulus = _maybe_list(metadata, "stimulus")

        summaries: List[SparseFeatureSummary] = []
        for feature_idx in range(codes.shape[1]):
            values = codes[:, feature_idx]
            active_mask = values > 0
            activation_frequency = float(active_mask.float().mean().item())
            if activation_frequency < min_activation_frequency:
                continue

            active_values = values[active_mask]
            mean_activation = float(active_values.mean().item()) if active_values.numel() else 0.0
            max_activation = float(values.max().item()) if values.numel() else 0.0

            top_values, top_indices = torch.topk(values, k=min(top_k_examples, values.shape[0]))
            top_examples = []
            for score, sample_idx in zip(top_values.tolist(), top_indices.tolist()):
                if score <= 0:
                    continue
                record: Dict[str, object] = {
                    "sample_index": int(sample_idx),
                    "activation": float(score),
                }
                if stimulus is not None:
                    record["stimulus"] = stimulus[sample_idx]
                if query_index is not None:
                    record["query_index"] = float(query_index[sample_idx].item())
                if attn_x is not None:
                    record["attention_x"] = float(attn_x[sample_idx].item())
                if attn_y is not None:
                    record["attention_y"] = float(attn_y[sample_idx].item())
                top_examples.append(record)

            summaries.append(
                SparseFeatureSummary(
                    feature_index=feature_idx,
                    activation_frequency=activation_frequency,
                    mean_activation=mean_activation,
                    max_activation=max_activation,
                    weighted_query_index_mean=_weighted_mean(query_index, values),
                    weighted_query_index_std=_weighted_std(query_index, values),
                    weighted_attention_x_mean=_weighted_mean(attn_x, values),
                    weighted_attention_x_std=_weighted_std(attn_x, values),
                    weighted_attention_y_mean=_weighted_mean(attn_y, values),
                    weighted_attention_y_std=_weighted_std(attn_y, values),
                    top_examples=top_examples,
                )
            )

        summaries.sort(key=lambda item: (item.activation_frequency, item.mean_activation), reverse=True)
        return SparseAutoencoderSummary(
            metrics=metrics,
            dead_feature_fraction=dead_feature_fraction,
            mean_active_features_per_sample=mean_active_features_per_sample,
            feature_summaries=summaries,
        )


def ablate_sparse_features(
    activations: torch.Tensor,
    model: SparseAutoencoder,
    feature_indices: Sequence[int],
    *,
    mode: str = "subtract_decoder",
) -> torch.Tensor:
    """
    Remove selected sparse features from activation vectors.

    Args:
        activations: [..., input_dim] activation tensor.
        model: Trained sparse autoencoder.
        feature_indices: Sparse feature ids to remove.
        mode:
            - ``subtract_decoder``: subtract the selected decoder contributions
              from the original activations.
            - ``reconstruct``: zero the selected sparse codes and reconstruct.
    """
    if not feature_indices:
        return activations.clone()

    input_shape = activations.shape
    input_dtype = activations.dtype
    input_device = activations.device
    sae_device = next(model.parameters()).device

    flat = activations.reshape(-1, input_shape[-1]).to(sae_device, dtype=torch.float32)
    index_tensor = torch.as_tensor(list(feature_indices), dtype=torch.long, device=sae_device)

    with torch.no_grad():
        codes = model.encode(flat)
        if mode == "subtract_decoder":
            contribution = codes[:, index_tensor] @ model.decoder.weight[:, index_tensor].transpose(0, 1)
            ablated = flat - contribution
        elif mode == "reconstruct":
            codes = codes.clone()
            codes[:, index_tensor] = 0.0
            ablated = model.decode(codes)
        else:
            raise ValueError(f"Unknown ablation mode: {mode!r}")

    return ablated.reshape(input_shape).to(device=input_device, dtype=input_dtype)


def _maybe_tensor(
    metadata: Optional[Mapping[str, Sequence[object] | torch.Tensor]],
    key: str,
) -> Optional[torch.Tensor]:
    if metadata is None or key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _maybe_list(
    metadata: Optional[Mapping[str, Sequence[object] | torch.Tensor]],
    key: str,
) -> Optional[List[object]]:
    if metadata is None or key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _weighted_mean(values: Optional[torch.Tensor], weights: torch.Tensor) -> Optional[float]:
    if values is None:
        return None
    total_weight = weights.sum().item()
    if total_weight <= 0:
        return None
    return float((values * weights).sum().item() / total_weight)


def _weighted_std(values: Optional[torch.Tensor], weights: torch.Tensor) -> Optional[float]:
    mean = _weighted_mean(values, weights)
    if values is None or mean is None:
        return None
    total_weight = weights.sum().item()
    if total_weight <= 0:
        return None
    variance = (((values - mean) ** 2) * weights).sum().item() / total_weight
    return float(variance ** 0.5)
