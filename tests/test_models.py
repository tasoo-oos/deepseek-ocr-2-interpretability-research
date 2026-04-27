"""Unit tests for model components."""

import pytest
import torch


def test_sam_encoder_forward():
    from src.models.sam_encoder import build_sam_vit_b
    model = build_sam_vit_b()
    model.eval()
    x = torch.zeros(1, 3, 1024, 1024)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 896, 16, 16), f"Unexpected shape: {out.shape}"


def test_sam_encoder_768():
    from src.models.sam_encoder import build_sam_vit_b
    model = build_sam_vit_b()
    model.eval()
    x = torch.zeros(1, 3, 768, 768)
    with torch.no_grad():
        out = model(x)
    assert out.shape[1] == 896


def test_qwen2_d2e_forward_1024():
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
    model = build_qwen2_decoder_as_encoder()
    model.eval()
    x = torch.zeros(1, 896, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 256, 896), f"Unexpected shape: {out.shape}"


def test_qwen2_d2e_forward_768():
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
    model = build_qwen2_decoder_as_encoder()
    model.eval()
    x = torch.zeros(1, 896, 12, 12)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 144, 896)


def test_qwen2_d2e_output_attentions():
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
    model = build_qwen2_decoder_as_encoder(output_attentions=True)
    model.eval()
    x = torch.zeros(1, 896, 16, 16)
    with torch.no_grad():
        result = model(x, output_attentions=True)
    assert isinstance(result, tuple)
    query_out, attentions, hidden_states, token_type_ids = result
    assert attentions is not None
    assert len(attentions) == 24
    assert attentions[0].shape[1] == 14  # 14 heads


def test_mlp_projector():
    from src.models.projector import MlpProjector
    from addict import Dict
    proj = MlpProjector(Dict(projector_type="linear", input_dim=896, n_embed=1280))
    x = torch.zeros(1, 256, 896)
    out = proj(x)
    assert out.shape == (1, 256, 1280)


def test_deepseek_ocr_model_init():
    from src.models.deepseek_ocr import DeepseekOCRModel
    model = DeepseekOCRModel()
    assert hasattr(model, 'sam_model')
    assert hasattr(model, 'qwen2_model')
    assert hasattr(model, 'projector')
    assert hasattr(model, 'view_seperator')


def test_deepseek_ocr_v1_model_init_is_separate_from_v2():
    from src.models.deepseek_ocr import DeepseekOCRModel
    from src.models.deepseek_ocr_v1 import DeepseekOCRV1Model

    model = DeepseekOCRV1Model(
        sam_model=torch.nn.Identity(),
        vision_model=torch.nn.Identity(),
    )
    assert isinstance(model, DeepseekOCRV1Model)
    assert not isinstance(model, DeepseekOCRModel)
    assert hasattr(model, "sam_model")
    assert hasattr(model, "vision_model")
    assert not hasattr(model, "qwen2_model")
    assert model.projector.cfg.input_dim == 2048


def test_deepseek_ocr_v1_sam_builder_contract():
    from src.models.deepseek_ocr_v1 import build_sam_vit_b_v1

    model = build_sam_vit_b_v1()
    assert model.net_3.out_channels == 1024


def test_deepseek_ocr_v1_encode_view_contract():
    from src.models.deepseek_ocr_v1 import DeepseekOCRV1Model

    class DummySam(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1024, 2, 2, dtype=x.dtype, device=x.device)

    class DummyVision(torch.nn.Module):
        def forward(self, image, patch_embeds):
            return torch.zeros(image.shape[0], 5, 1024, dtype=image.dtype, device=image.device)

    model = DeepseekOCRV1Model(
        sam_model=DummySam(),
        vision_model=DummyVision(),
    )
    out = model(torch.zeros(1, 3, 32, 32))
    assert out.shape == (1, 4, 1280)
