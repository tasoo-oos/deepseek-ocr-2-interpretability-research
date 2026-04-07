"""
Causal intervention tools — ablations, activation patching, activation steering.

Use InterventionManager as a context manager to ensure hooks are always cleaned up:

    with InterventionManager(model) as mgr:
        mgr.ablate_attention_head(layer=12, head=7)
        output = model(pixel_values, images_crop, images_spatial_crop)
"""

import torch
from typing import List, Optional

from src.models.deepseek_ocr import DeepseekOCRModel


class InterventionManager:
    """
    Apply causal interventions (ablations, patches) to DeepSeek-OCR-2.

    Args:
        model: The DeepseekOCRModel to intervene on.
    """

    def __init__(self, model: DeepseekOCRModel):
        self.model = model
        self.interventions: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Attention-level interventions
    # ------------------------------------------------------------------

    def ablate_attention_head(
        self,
        layer: int,
        head: int,
        component: str = "d2e",
    ) -> None:
        """
        Zero out all outputs contributed by a specific attention head.

        Args:
            layer: Layer index.
            head:  Head index.
            component: "d2e" (Qwen2 decoder layers) or "sam" (SAM blocks).
        """
        if component == "d2e":
            target = self.model.qwen2_model.model.model.layers[layer].self_attn
        elif component == "sam":
            target = self.model.sam_model.blocks[layer].attn
        else:
            raise ValueError(f"Unknown component: {component!r}. Use 'd2e' or 'sam'.")

        n_heads = target.num_heads

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden, *rest = output
                head_dim = hidden.shape[-1] // n_heads
                hidden_clone = hidden.clone()
                hidden_clone[..., head * head_dim:(head + 1) * head_dim] = 0.0
                return (hidden_clone, *rest)
            else:
                head_dim = output.shape[-1] // n_heads
                out_clone = output.clone()
                out_clone[..., head * head_dim:(head + 1) * head_dim] = 0.0
                return out_clone

        handle = target.register_forward_hook(hook)
        self.interventions.append(handle)

    # ------------------------------------------------------------------
    # Token-level interventions
    # ------------------------------------------------------------------

    def ablate_query_tokens(self, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        """
        Zero out query token representations in the D2E output.

        This intervenes after the entire D2E forward pass, zeroing the
        query portion of the last hidden state before it is passed to the projector.

        Args:
            start_idx: First query position to zero (relative to query sequence start).
            end_idx:   Last query position (exclusive). None = zero all remaining.
        """
        def hook(module, input, output):
            y = output  # output of Qwen2Decoder2Encoder.forward() is just the query tensor
            n_img = y.shape[1]
            if isinstance(output, tuple):
                y = output[0]
            y = y.clone()
            if end_idx is None:
                y[:, start_idx:, :] = 0.0
            else:
                y[:, start_idx:end_idx, :] = 0.0
            if isinstance(output, tuple):
                return (y,) + output[1:]
            return y

        handle = self.model.qwen2_model.register_forward_hook(hook)
        self.interventions.append(handle)

    def ablate_image_tokens(
        self, layer: int, start_idx: int = 0, end_idx: Optional[int] = None
    ) -> None:
        """
        Zero out image token representations in a D2E layer's output.

        Args:
            layer:     D2E layer index.
            start_idx: First image position to zero.
            end_idx:   Last image position (exclusive). None = all.
        """
        target = self.model.qwen2_model.model.model.layers[layer]

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                if end_idx is None:
                    hidden[:, start_idx:, :] = 0.0
                else:
                    hidden[:, start_idx:end_idx, :] = 0.0
                return (hidden,) + output[1:]
            else:
                out = output.clone()
                if end_idx is None:
                    out[:, start_idx:, :] = 0.0
                else:
                    out[:, start_idx:end_idx, :] = 0.0
                return out

        handle = target.register_forward_hook(hook)
        self.interventions.append(handle)

    # ------------------------------------------------------------------
    # Activation patching
    # ------------------------------------------------------------------

    def patch_activation(
        self,
        layer: int,
        position: int,
        new_value: torch.Tensor,
        component: str = "d2e",
    ) -> None:
        """
        Replace the activation at a specific (layer, position) with new_value.

        Args:
            layer:     Layer index.
            position:  Token position in the sequence.
            new_value: Replacement activation [hidden_dim] or [B, hidden_dim].
            component: "d2e" or "sam".
        """
        if component == "d2e":
            target = self.model.qwen2_model.model.model.layers[layer]
        elif component == "sam":
            target = self.model.sam_model.blocks[layer]
        else:
            raise ValueError(f"Unknown component: {component!r}")

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, position, :] = new_value.to(hidden.device, hidden.dtype)
                return (hidden,) + output[1:]
            else:
                out = output.clone()
                out[:, position, :] = new_value.to(out.device, out.dtype)
                return out

        handle = target.register_forward_hook(hook)
        self.interventions.append(handle)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_interventions(self) -> None:
        """Remove all registered intervention hooks."""
        for handle in self.interventions:
            handle.remove()
        self.interventions = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_interventions()
