"""
Circuit discovery via activation patching.

Uses clean→corrupted patching to measure how much each (layer, position)
contributes to a given behavioural metric.
"""

import torch
from typing import Callable, Dict, List, Optional, Tuple

from src.models.deepseek_ocr import DeepseekOCRModel
from .feature_extractor import FeatureExtractor
from .interventions import InterventionManager


class CircuitDiscovery:
    """
    Discover computational circuits using causal activation patching.

    Args:
        model:            The DeepseekOCRModel to analyse.
        feature_extractor: A FeatureExtractor bound to the same model.
        intervention_manager: An InterventionManager bound to the same model.
    """

    def __init__(
        self,
        model: DeepseekOCRModel,
        feature_extractor: FeatureExtractor,
        intervention_manager: InterventionManager,
    ):
        self.model = model
        self.features = feature_extractor
        self.interventions = intervention_manager

    def activation_patching(
        self,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Dict[str, torch.Tensor],
        layer: int,
        position: int,
        metric_fn: Callable[[object], float],
        component: str = "d2e",
    ) -> float:
        """
        Patch a single (layer, position) activation from the clean run into the corrupted run.

        The metric_fn receives the model output and returns a scalar score
        (higher = more "correct" behaviour).

        Args:
            clean_input:      Dict with pixel_values, images_crop, images_spatial_crop for clean image.
            corrupted_input:  Same dict for the corrupted image.
            layer:            D2E / SAM layer index.
            position:         Token position index.
            metric_fn:        fn(model_output) -> float
            component:        "d2e" or "sam".

        Returns:
            Impact score = metric_fn(patched_output)
        """
        # --- Step 1: clean forward pass to collect activations ---
        self.features.register_hooks(
            d2e_layers=[layer] if component == "d2e" else None,
            sam_layers=[layer] if component == "sam" else None,
            projector=False,
        )
        clean_acts = self.features.extract(**clean_input)
        self.features.clear_hooks()

        key = f"{component}_layer_{layer}"
        if key not in clean_acts:
            raise KeyError(f"Activation key '{key}' not found. Available: {list(clean_acts.keys())}")

        clean_activation = clean_acts[key][:, position, :]  # [B, hidden_dim]

        # --- Step 2: corrupted forward pass with patch applied ---
        self.interventions.patch_activation(
            layer=layer,
            position=position,
            new_value=clean_activation,
            component=component,
        )
        with torch.no_grad():
            output = self.model(**corrupted_input)
        self.interventions.clear_interventions()

        return metric_fn(output)

    def find_circuit_for_task(
        self,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Dict[str, torch.Tensor],
        metric_fn: Callable[[object], float],
        layers: Optional[List[int]] = None,
        n_positions: int = 10,
        component: str = "d2e",
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Identify the most causally important (layer, position) pairs for a task.

        Runs activation patching for each (layer, position) and ranks by impact.

        Args:
            clean_input:      Dict for the "correct" run.
            corrupted_input:  Dict for the degraded run.
            metric_fn:        Callable(output) -> float (higher = better).
            layers:           Layers to test (None = all 24 D2E layers).
            n_positions:      Number of positions per layer to test.
            component:        "d2e" or "sam".

        Returns:
            {
                'critical_positions': [(layer, position), ...] sorted by impact
            }
        """
        if layers is None:
            layers = list(range(24))

        # Get baseline corrupted score
        with torch.no_grad():
            baseline_output = self.model(**corrupted_input)
        baseline_score = metric_fn(baseline_output)

        impact_scores: List[Tuple[float, int, int]] = []

        # Sample positions evenly
        # Estimate sequence length from a quick forward pass
        self.features.register_hooks(d2e_layers=[0], projector=False)
        acts = self.features.extract(**clean_input)
        self.features.clear_hooks()
        seq_len = acts.get("d2e_layer_0", torch.zeros(1, 1, 1)).shape[1]
        positions = [int(i * seq_len / n_positions) for i in range(n_positions)]

        for layer in layers:
            for pos in positions:
                score = self.activation_patching(
                    clean_input=clean_input,
                    corrupted_input=corrupted_input,
                    layer=layer,
                    position=pos,
                    metric_fn=metric_fn,
                    component=component,
                )
                impact = score - baseline_score
                impact_scores.append((impact, layer, pos))
                print(f"  Layer {layer:2d}, pos {pos:4d}: impact = {impact:+.4f}")

        impact_scores.sort(reverse=True)
        critical_positions = [(layer, pos) for _, layer, pos in impact_scores[:10]]

        return {"critical_positions": critical_positions}
