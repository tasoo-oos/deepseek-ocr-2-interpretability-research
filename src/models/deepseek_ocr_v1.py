"""
DeepSeek-OCR v1 vision model components.

This module is intentionally separate from ``src.models.deepseek_ocr`` because
OCR v1 uses a different vision bridge:

    SAM ViT-B compressor -> CLIP-L style global transformer -> linear projector

OCR v2 instead uses:

    SAM ViT-B compressor -> Qwen2 decoder-as-encoder -> linear projector
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict as ADict

from .projector import MlpProjector
from .sam_encoder import ImageEncoderViT

_N_EMBED = 1280


class ConfigNamespace(SimpleNamespace):
    """Small attribute config with the ``get`` method expected by upstream code."""

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _resize_clip_abs_pos(abs_pos: torch.Tensor, token_count: int) -> torch.Tensor:
    dim = abs_pos.size(-1)
    abs_pos = abs_pos.squeeze(0)
    cls_token, grid_pos = abs_pos[:1], abs_pos[1:]
    src_size = int(math.sqrt(grid_pos.shape[0]))
    tgt_size = int(math.sqrt(token_count - 1))

    if src_size == tgt_size:
        return abs_pos.view(1, token_count, dim)

    dtype = grid_pos.dtype
    grid_pos = grid_pos.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).float()
    grid_pos = F.interpolate(
        grid_pos,
        size=(tgt_size, tgt_size),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).to(dtype)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(tgt_size * tgt_size, dim)
    return torch.cat([cls_token, grid_pos], dim=0).view(1, token_count, dim)


@torch.jit.script
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class CLIPVisionEmbeddingsV1(nn.Module):
    """CLIP-L embeddings adapted to accept pre-compressed SAM patch embeddings."""

    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            num_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        pos = self.position_embedding(self.position_ids)
        return embeddings + _resize_clip_abs_pos(pos, embeddings.size(1))


class NoTPAttentionV1(nn.Module):
    def __init__(self, cfg: ConfigNamespace):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.qkv_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = torch.split(qkv, 1, dim=2)
        q = q.squeeze(2).permute(0, 2, 1, 3)
        k = k.squeeze(2).permute(0, 2, 1, 3)
        v = v.squeeze(2).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class NoTPFeedForwardV1(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(quick_gelu(self.fc1(x)))


class NoTPTransformerBlockV1(nn.Module):
    def __init__(self, cfg: ConfigNamespace, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = NoTPAttentionV1(cfg)
        self.mlp = NoTPFeedForwardV1(cfg.hidden_size, cfg.ffn_hidden_size)
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x + self.self_attn(self.layer_norm1(x))
        return hidden + self.mlp(self.layer_norm2(hidden))


class NoTPTransformerV1(nn.Module):
    def __init__(self, cfg: ConfigNamespace):
        super().__init__()
        self.layers = nn.ModuleList(
            NoTPTransformerBlockV1(cfg, layer_id + 1)
            for layer_id in range(cfg.num_layers)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CLIPLargeVisionTransformerV1(nn.Module):
    """The OCR v1 CLIP-L global-token mixer from the upstream DeepSeek-OCR code."""

    def __init__(self, cfg: ConfigNamespace):
        super().__init__()
        self.cfg = cfg
        self.embeddings = CLIPVisionEmbeddingsV1(
            hidden_size=cfg.hidden_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
        )
        self.transformer = NoTPTransformerV1(cfg)
        self.pre_layrnorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.pre_layernorm_epsilon)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, patch_embeds)
        hidden_states = self.pre_layrnorm(hidden_states)
        return self.transformer(hidden_states)


def build_clip_l_v1(
    num_layers: int = 24,
    hidden_size: int = 1024,
    num_attention_heads: int = 16,
    ffn_hidden_size: int = 4096,
) -> CLIPLargeVisionTransformerV1:
    cfg = ConfigNamespace(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        seq_length=256,
        max_position_embeddings=256,
        use_flash_attn=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        pre_layernorm_epsilon=1e-5,
        image_size=224,
        patch_size=14,
    )
    return CLIPLargeVisionTransformerV1(cfg)


def build_sam_vit_b_v1(checkpoint: Optional[str] = None) -> ImageEncoderViT:
    """Build the OCR v1 SAM-B compressor.

    The module names are intentionally compatible with the upstream v1 checkpoint:
    ``patch_embed``, ``blocks``, ``neck``, ``net_2``, and ``net_3``.
    """

    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=(2, 5, 8, 11),
        window_size=14,
        out_chans=256,
    )
    image_encoder.net_3 = nn.Conv2d(
        512,
        1024,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        image_encoder.load_state_dict(state_dict, strict=True)

    return image_encoder


@dataclass(frozen=True)
class OCRV1FeatureBundle:
    sam_features: torch.Tensor
    clip_features: torch.Tensor
    projected_features: torch.Tensor


class DeepseekOCRV1Model(nn.Module):
    """Standalone OCR v1 vision bridge for interpretability experiments."""

    def __init__(
        self,
        sam_model: Optional[nn.Module] = None,
        vision_model: Optional[nn.Module] = None,
        n_embed: int = _N_EMBED,
    ):
        super().__init__()
        self.sam_model = sam_model if sam_model is not None else build_sam_vit_b_v1()
        self.vision_model = vision_model if vision_model is not None else build_clip_l_v1()
        self.projector = MlpProjector(
            ADict(projector_type="linear", input_dim=2048, n_embed=n_embed)
        )

        embed_std = 1 / math.sqrt(n_embed)
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "DeepseekOCRV1Model":
        model = cls(**kwargs)
        weight_files = cls._find_weight_files(model_path)
        if weight_files:
            state_dict = cls._load_weights(weight_files)
            model._load_vision_weights(state_dict)
        else:
            print(f"Warning: no weight files found at '{model_path}'.")
        return model.to(device=device, dtype=dtype)

    @staticmethod
    def _find_weight_files(model_path: str) -> List[str]:
        import glob

        path = Path(model_path)
        if path.is_dir():
            files = sorted(glob.glob(str(path / "*.safetensors")))
            return files or sorted(glob.glob(str(path / "pytorch_model*.bin")))

        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=model_path)
            files = sorted(glob.glob(f"{local_dir}/*.safetensors"))
            return files or sorted(glob.glob(f"{local_dir}/pytorch_model*.bin"))
        except Exception as exc:
            print(f"Could not resolve weights from Hub: {exc}")
            return []

    @staticmethod
    def _load_weights(weight_files: List[str]) -> Dict[str, torch.Tensor]:
        state_dict: Dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            if weight_file.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict.update(load_file(weight_file))
            else:
                state_dict.update(torch.load(weight_file, map_location="cpu"))
        return state_dict

    def _load_vision_weights(self, raw_state_dict: Dict[str, torch.Tensor]) -> None:
        keep_prefixes = (
            "sam_model.",
            "vision_model.",
            "projector.",
            "image_newline",
            "view_seperator",
        )
        state_dict = {
            key: value
            for key, value in raw_state_dict.items()
            if key.startswith(keep_prefixes)
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Warning: unexpected OCR v1 vision keys: {len(unexpected)}")
        if missing:
            print(f"Warning: missing OCR v1 vision keys: {len(missing)}")

    def encode_view(
        self,
        image: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | OCRV1FeatureBundle:
        sam_features = self.sam_model(image)
        clip_features = self.vision_model(image, sam_features)
        fused_features = torch.cat(
            (
                clip_features[:, 1:],
                sam_features.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        projected_features = self.projector(fused_features)
        if return_intermediates:
            return OCRV1FeatureBundle(
                sam_features=sam_features,
                clip_features=clip_features,
                projected_features=projected_features,
            )
        return projected_features

    def encode_crops(
        self,
        patches: torch.Tensor,
        image_ori: torch.Tensor,
        crop_shape: Tuple[int, int],
    ) -> torch.Tensor:
        local_features = self.encode_view(patches)
        global_features = self.encode_view(image_ori)

        _, local_hw, local_dim = local_features.shape
        local_side = int(math.sqrt(local_hw))
        _, global_hw, global_dim = global_features.shape
        global_side = int(math.sqrt(global_hw))
        width_crop_num, height_crop_num = crop_shape

        global_features = global_features.view(global_side, global_side, global_dim)
        global_features = torch.cat(
            [
                global_features,
                self.image_newline[None, None, :].expand(global_side, 1, global_dim),
            ],
            dim=1,
        ).view(-1, global_dim)

        local_features = (
            local_features.view(
                height_crop_num,
                width_crop_num,
                local_side,
                local_side,
                local_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(height_crop_num * local_side, width_crop_num * local_side, local_dim)
        )
        local_features = torch.cat(
            [
                local_features,
                self.image_newline[None, None, :].expand(
                    height_crop_num * local_side,
                    1,
                    local_dim,
                ),
            ],
            dim=1,
        ).view(-1, local_dim)

        return torch.cat(
            [local_features, global_features, self.view_seperator[None, :]],
            dim=0,
        )

    def forward(
        self,
        image: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | OCRV1FeatureBundle:
        return self.encode_view(image, return_intermediates=return_intermediates)


def build_deepseek_ocr_v1(**kwargs) -> DeepseekOCRV1Model:
    return DeepseekOCRV1Model(**kwargs)

