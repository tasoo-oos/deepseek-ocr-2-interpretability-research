"""
Hook-based activation extraction for all DeepSeek-OCR-2 components.

Registers forward hooks on SAM blocks, D2E layers, and the projector
to capture intermediate activations without modifying the model.
"""

import torch
from typing import Callable, Dict, List, Optional

from src.models.deepseek_ocr import DeepseekOCRModel


class FeatureExtractor:
    """
    Extract intermediate activations from DeepSeek-OCR-2 using PyTorch hooks.

    Usage:
        extractor = FeatureExtractor(model)
        extractor.register_hooks(sam_layers=[0, 6, 11], d2e_layers=[0, 12, 23])
        activations = extractor.extract(pixel_values, images_crop, images_spatial_crop)
        extractor.clear_hooks()
    """

    def __init__(self, model: DeepseekOCRModel):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, name: str) -> Callable:
        """Create a forward hook that stores the output under ``name``."""
        def hook(module, input, output):
            # Handle tuple outputs (e.g. Qwen2 layer returns tuple)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook

    def register_hooks(
        self,
        sam_layers: Optional[List[int]] = None,
        d2e_layers: Optional[List[int]] = None,
        projector: bool = True,
    ) -> None:
        """
        Register forward hooks to capture activations.

        Args:
            sam_layers: SAM block indices to hook (0–11).
            d2e_layers: D2E transformer layer indices to hook (0–23).
            projector:  Whether to hook the MLP projector output.
        """
        # SAM encoder blocks
        if sam_layers:
            for layer_idx in sam_layers:
                block = self.model.sam_model.blocks[layer_idx]
                handle = block.register_forward_hook(
                    self._make_hook(f"sam_layer_{layer_idx}")
                )
                self.hooks.append(handle)

        # D2E (Qwen2) transformer layers
        if d2e_layers:
            for layer_idx in d2e_layers:
                layer = self.model.qwen2_model.model.model.layers[layer_idx]
                handle = layer.register_forward_hook(
                    self._make_hook(f"d2e_layer_{layer_idx}")
                )
                self.hooks.append(handle)

        # MLP projector
        if projector:
            handle = self.model.projector.register_forward_hook(
                self._make_hook("projector")
            )
            self.hooks.append(handle)

    def extract(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and collect hooked activations.

        Args:
            pixel_values: Global view [n_images, 1, 3, H, W]
            images_crop: Local crops [n_images, 1, n_patches, 3, h, w]
            images_spatial_crop: Crop grid [n_images, 1, 2]

        Returns:
            Dict mapping hook names to activation tensors.
        """
        self.activations = {}
        with torch.no_grad():
            self.model.get_multimodal_embeddings(
                pixel_values, images_crop, images_spatial_crop
            )
        return dict(self.activations)

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.activations = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_hooks()
