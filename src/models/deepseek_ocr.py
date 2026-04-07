"""
DeepSeek-OCR-2 model — clean HuggingFace Transformers implementation.

No vLLM dependencies. Suitable for mechanistic interpretability research.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from addict import Dict as ADict

from .sam_encoder import build_sam_vit_b
from .qwen2_d2e import build_qwen2_decoder_as_encoder
from .projector import MlpProjector

_IMAGE_TOKEN = "<image>"
_N_EMBED = 1280  # Language model embedding dimension


class DeepseekOCRModel(nn.Module):
    """
    Clean DeepSeek-OCR-2 model for mechanistic interpretability research.

    Vision pipeline: SAM encoder → D2E (Qwen2 Decoder-as-Encoder) → MLP projector
    Optional language model for full inference.

    Args:
        use_language_model: If True, load and expose the language model component.
        output_attentions: Enable attention weight output from D2E by default.
        output_hidden_states: Enable hidden state output from D2E by default.
        image_token_id: Token ID for <image> (required for full inference).
    """

    def __init__(
        self,
        use_language_model: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        image_token_id: Optional[int] = None,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        self.use_language_model = use_language_model
        self.image_token_id = image_token_id

        # Vision components
        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_qwen2_decoder_as_encoder(
            attn_implementation=attn_implementation,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        n_embed = _N_EMBED
        self.projector = MlpProjector(
            ADict(projector_type="linear", input_dim=896, n_embed=n_embed)
        )

        # Learned separator token between local and global views
        embed_std = 1 / math.sqrt(n_embed)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        # Language model (optional)
        self.language_model: Optional[nn.Module] = None

        self.sam_model.to(dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        use_language_model: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "DeepseekOCRModel":
        """
        Load model weights from a HuggingFace Hub model or local directory.

        Args:
            model_path: HuggingFace model ID (e.g. 'deepseek-ai/DeepSeek-OCR-2')
                        or path to a local directory containing model weights.
            use_language_model: Whether to load the language model component.
            device: Device to load weights onto ('cpu', 'cuda', etc.).
            dtype: Weight dtype.

        Returns:
            Loaded DeepseekOCRModel.
        """
        from transformers import AutoTokenizer

        # Resolve image_token_id
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            image_token_id = tokenizer.vocab.get(_IMAGE_TOKEN)
        except Exception:
            image_token_id = None

        model = cls(
            use_language_model=use_language_model,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            image_token_id=image_token_id,
            **kwargs,
        )

        # Load weights
        weight_files = cls._find_weight_files(model_path)
        if weight_files:
            state_dict = cls._load_weights(weight_files)
            model._load_remapped_weights(
                state_dict, use_language_model=use_language_model
            )
        else:
            print(
                f"Warning: No weight files found at '{model_path}'. Model has random weights."
            )

        model = model.to(device=device, dtype=dtype)
        return model

    @staticmethod
    def _find_weight_files(model_path: str) -> List[str]:
        """Find weight files (safetensors or pytorch_model.bin) in the model directory."""
        import glob

        path = Path(model_path)

        # Try local directory first
        if path.is_dir():
            files = sorted(glob.glob(str(path / "*.safetensors")))
            if not files:
                files = sorted(glob.glob(str(path / "pytorch_model*.bin")))
            return files

        # Try HuggingFace Hub
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=model_path)
            files = sorted(glob.glob(f"{local_dir}/*.safetensors"))
            if not files:
                files = sorted(glob.glob(f"{local_dir}/pytorch_model*.bin"))
            return files
        except Exception as e:
            print(f"Could not download from Hub: {e}")
            return []

    @staticmethod
    def _load_weights(weight_files: List[str]) -> Dict[str, torch.Tensor]:
        """Load and merge all weight files into a single state dict."""
        state_dict = {}
        for wf in weight_files:
            if wf.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict.update(load_file(wf))
            else:
                state_dict.update(torch.load(wf, map_location="cpu"))
        return state_dict

    def _load_remapped_weights(
        self,
        raw_state_dict: Dict[str, torch.Tensor],
        use_language_model: bool = False,
    ) -> None:
        """
        Apply the weight name remapping from HuggingFace format to this model's format.

        HuggingFace naming convention:
          model.sam_model.*       → sam_model.*
          model.qwen2_model.*     → qwen2_model.*
          model.projector.*       → projector.*
          model.view_seperator    → view_seperator
          language.*              → language_model.* (if use_language_model)
        """
        vision_state = {}
        language_state = {}

        for name, tensor in raw_state_dict.items():
            if any(
                k in name
                for k in ("sam_model", "qwen2_model", "projector", "view_seperator")
            ):
                new_name = name.replace("model.", "", 1)
                vision_state[new_name] = tensor
            else:
                language_state[name] = tensor

        missing, unexpected = self.load_state_dict(vision_state, strict=False)
        if missing:
            print(
                f"  Missing keys (vision): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"  Unexpected keys (vision): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )

        if use_language_model and self.language_model is not None and language_state:
            missing_lm, _ = self.language_model.load_state_dict(
                language_state, strict=False
            )
            if missing_lm:
                print(f"  Missing keys (language): {missing_lm[:5]}")

    def load_language_model(self, model_path: str) -> None:
        """
        Load the language model from a HuggingFace path.

        Uses ``AutoModel`` with ``trust_remote_code=True`` so that custom
        architectures (like DeepseekOCR2) are supported.  Call after
        constructing the model with ``use_language_model=True``.
        """
        from transformers import AutoModel

        self.language_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.use_language_model = True

    # ------------------------------------------------------------------
    # Core vision pipeline
    # ------------------------------------------------------------------

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Run the SAM → D2E → Projector pipeline for one batch.

        Args:
            pixel_values: Global views [n_images, B, 3, H, W]
            images_crop: Local crop views [n_images, B, n_patches, 3, h, w]
            images_spatial_crop: Crop grid shape [n_images, B, 2]
            return_intermediate: If True, also return intermediate tensors.

        Returns:
            List of per-image feature tensors (local + global + separator).
            If return_intermediate: (features_list, intermediates_dict)
        """
        images_in_this_batch = []
        intermediates: Dict[str, list] = {
            "sam_features_local": [],
            "sam_features_global": [],
            "d2e_outputs_local": [],
            "d2e_outputs_global": [],
            "projected_local": [],
            "projected_global": [],
        }

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                patches = images_crop[jdx][0].to(torch.bfloat16)
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:
                    # Local crops
                    local_sam = self.sam_model(patches)
                    local_d2e = self.qwen2_model(local_sam)
                    if isinstance(local_d2e, tuple):
                        local_d2e = local_d2e[0]
                    local_features = self.projector(local_d2e)

                    # Global view
                    global_sam = self.sam_model(image_ori)
                    global_d2e = self.qwen2_model(global_sam)
                    if isinstance(global_d2e, tuple):
                        global_d2e = global_d2e[0]
                    global_features = self.projector(global_d2e)

                    if return_intermediate:
                        intermediates["sam_features_local"].append(local_sam.cpu())
                        intermediates["sam_features_global"].append(global_sam.cpu())
                        intermediates["d2e_outputs_local"].append(local_d2e.cpu())
                        intermediates["d2e_outputs_global"].append(global_d2e.cpu())
                        intermediates["projected_local"].append(local_features.cpu())
                        intermediates["projected_global"].append(global_features.cpu())

                    _, hw, n_dim = global_features.shape
                    global_features = global_features.view(-1, n_dim)
                    local_features = local_features.view(-1, local_features.shape[-1])

                    global_local_features = torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )
                else:
                    # No local crops — global only
                    global_sam = self.sam_model(image_ori)
                    global_d2e = self.qwen2_model(global_sam)
                    if isinstance(global_d2e, tuple):
                        global_d2e = global_d2e[0]
                    global_features = self.projector(global_d2e)

                    if return_intermediate:
                        intermediates["sam_features_global"].append(global_sam.cpu())
                        intermediates["d2e_outputs_global"].append(global_d2e.cpu())
                        intermediates["projected_global"].append(global_features.cpu())

                    _, hw, n_dim = global_features.shape
                    global_features = global_features.view(-1, n_dim)
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        if return_intermediate:
            return images_in_this_batch, intermediates
        return images_in_this_batch

    def get_multimodal_embeddings(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        return_intermediate: bool = False,
    ):
        """Process image inputs through the vision pipeline."""
        pixel_values = pixel_values.to(torch.bfloat16)
        images_spatial_crop = images_spatial_crop.to(dtype=torch.long)
        return self._pixel_values_to_embedding(
            pixel_values,
            images_crop,
            images_spatial_crop,
            return_intermediate=return_intermediate,
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Merge text token embeddings with vision embeddings.
        Requires a language model to be loaded.
        """
        if self.language_model is None:
            raise RuntimeError(
                "Language model not loaded. Call load_language_model() first."
            )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if multimodal_embeddings is not None:
            from einops import rearrange

            # Replace <image> token positions with vision embeddings
            image_token_mask = input_ids == self.image_token_id
            flat_vision = torch.cat(multimodal_embeddings, dim=0)
            inputs_embeds[image_token_mask] = flat_vision.to(inputs_embeds.dtype)

        return inputs_embeds

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        images_crop: Optional[torch.Tensor] = None,
        images_spatial_crop: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        **kwargs,
    ):
        """
        Forward pass.

        Vision-only mode (input_ids=None):
            Returns multimodal embeddings (List of per-image tensors).

        Full inference mode (input_ids provided, language model loaded):
            Returns language model output.

        Args:
            pixel_values: Global view images [n_images, 1, 3, H, W]
            images_crop: Local crop patches [n_images, 1, n_patches, 3, h, w]
            images_spatial_crop: Crop grid [n_images, 1, 2]
            input_ids: Text token IDs [B, L]
            inputs_embeds: Pre-computed embeddings (skips vision encoding)
            return_intermediate: Return intermediate activation tensors

        Returns:
            Depends on mode — see above.
        """
        if inputs_embeds is None and pixel_values is not None:
            vision_out = self.get_multimodal_embeddings(
                pixel_values,
                images_crop,
                images_spatial_crop,
                return_intermediate=return_intermediate,
            )
            if return_intermediate:
                vision_embeddings, intermediates = vision_out
            else:
                vision_embeddings = vision_out
                intermediates = None

            if input_ids is None:
                if return_intermediate:
                    return vision_embeddings, intermediates
                return vision_embeddings

            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        if self.language_model is not None and inputs_embeds is not None:
            return self.language_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        return inputs_embeds
