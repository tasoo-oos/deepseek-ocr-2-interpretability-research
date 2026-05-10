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


@dataclass
class ActivationStore:
    activations: dict[str, object] = field(default_factory=dict)

    def clear(self) -> None:
        self.activations.clear()

    def save(self, name: str, tensor) -> None:
        self.activations[name] = _detach(tensor)

    def shapes(self) -> dict[str, object]:
        return {name: _shape(value) for name, value in self.activations.items()}


class HookManager:
    def __init__(self) -> None:
        self._handles = []

    def add_activation_hook(self, model, module_name, store: ActivationStore) -> None:
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise KeyError(f"Unknown module: {module_name}")

        def hook(_module, _inputs, output):
            store.save(module_name, output)

        self._handles.append(modules[module_name].register_forward_hook(hook))

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
