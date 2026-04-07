"""Tests for circuit discovery patching utilities."""

import unittest
from typing import Any, Dict, Optional, cast

import torch

from src.analysis.circuits import CircuitDiscovery


class FakeCircuitModel:
    def __init__(self):
        self.patched_position: Optional[int] = None
        self.patched_value: Optional[torch.Tensor] = None

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        output = signal.clone()
        if self.patched_position is not None:
            assert self.patched_value is not None
            output[:, self.patched_position, :] = self.patched_value
        return output


class FakeFeatureExtractor:
    def __init__(self, activations: Dict[str, torch.Tensor]):
        self.activations = activations
        self.requested_layers: list[int] = []

    def register_hooks(self, sam_layers=None, d2e_layers=None, projector=False):
        self.requested_layers = d2e_layers or sam_layers or []

    def extract(self, **kwargs):
        return {
            f"d2e_layer_{layer}": self.activations[f"d2e_layer_{layer}"]
            for layer in self.requested_layers
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


class CircuitDiscoveryTests(unittest.TestCase):
    def test_activation_patching_restores_clean_signal_at_target_position(self):
        model = FakeCircuitModel()
        clean_activations = {"d2e_layer_3": torch.tensor([[[5.0], [1.0], [0.0]]])}
        discovery = CircuitDiscovery(
            model=cast(Any, model),
            feature_extractor=cast(Any, FakeFeatureExtractor(clean_activations)),
            intervention_manager=cast(Any, FakeInterventionManager(model)),
        )

        score = discovery.activation_patching(
            clean_input={"signal": torch.zeros(1, 3, 1)},
            corrupted_input={"signal": torch.zeros(1, 3, 1)},
            layer=3,
            position=0,
            metric_fn=lambda output: cast(torch.Tensor, output)[:, 0, 0].sum().item(),
        )

        self.assertEqual(score, 5.0)

    def test_find_circuit_for_task_ranks_most_important_position_first(self):
        model = FakeCircuitModel()
        clean_activations = {
            "d2e_layer_0": torch.tensor([[[1.0], [2.0], [8.0], [3.0]]]),
        }
        discovery = CircuitDiscovery(
            model=cast(Any, model),
            feature_extractor=cast(Any, FakeFeatureExtractor(clean_activations)),
            intervention_manager=cast(Any, FakeInterventionManager(model)),
        )

        result = discovery.find_circuit_for_task(
            clean_input={"signal": torch.zeros(1, 4, 1)},
            corrupted_input={"signal": torch.zeros(1, 4, 1)},
            metric_fn=lambda output: cast(torch.Tensor, output).sum().item(),
            layers=[0],
            n_positions=4,
        )

        self.assertEqual(result["critical_positions"][0], (0, 2))


if __name__ == "__main__":
    unittest.main()
