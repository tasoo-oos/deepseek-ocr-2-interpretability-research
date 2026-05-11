from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value


def _shape(value):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, tuple | list):
        return [_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    return type(value).__name__


def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()) if values.numel() > 1 else 0.0,
        "abs_mean": float(values.abs().mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "numel": int(values.numel()),
    }


def _summary(value):
    if torch.is_tensor(value):
        return _tensor_summary(value)
    if isinstance(value, tuple | list):
        return [_summary(item) for item in value]
    if isinstance(value, dict):
        return {key: _summary(item) for key, item in value.items()}
    return {"type": type(value).__name__}


@dataclass
class ActivationStore:
    activations: dict[str, list[object]] = field(default_factory=dict)
    metadata: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    def clear(self) -> None:
        self.activations.clear()
        self.metadata.clear()

    def save(self, name: str, module, inputs, output) -> None:
        call_index = len(self.activations.get(name, []))
        self.activations.setdefault(name, []).append(_detach(output))
        self.metadata.setdefault(name, []).append(
            {
                "call_index": call_index,
                "module_class": module.__class__.__name__,
                "input_shapes": _shape(inputs),
                "output_shapes": _shape(output),
                "output_summary": _summary(output),
            }
        )

    def shapes(self) -> dict[str, object]:
        return {name: [_shape(call) for call in calls] for name, calls in self.activations.items()}

    def call_counts(self) -> dict[str, int]:
        return {name: len(calls) for name, calls in self.activations.items()}


class HookManager:
    def __init__(self) -> None:
        self._handles = []

    def add_activation_hook(self, model, module_name, store: ActivationStore) -> None:
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise KeyError(f"Unknown module: {module_name}")

        def hook(module, inputs, output):
            store.save(module_name, module, inputs, output)

        self._handles.append(modules[module_name].register_forward_hook(hook))

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
