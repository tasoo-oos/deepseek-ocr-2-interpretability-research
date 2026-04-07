"""Unit tests for attention extraction and visualization."""

import pytest
import torch
from PIL import Image


@pytest.fixture
def d2e_model():
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
    model = build_qwen2_decoder_as_encoder(output_attentions=True)
    model.eval()
    return model


@pytest.fixture
def sam_features_1024():
    return torch.zeros(1, 896, 16, 16)


def test_attention_extractor_shapes(d2e_model, sam_features_1024):
    with torch.no_grad():
        result = d2e_model(sam_features_1024, output_attentions=True)
    query_out, attentions, hidden_states, token_type_ids = result
    assert len(attentions) == 24
    # [B, H, seq_len, seq_len] where seq_len = 256 image + 256 query = 512
    assert attentions[0].shape == (1, 14, 512, 512)
    assert token_type_ids.shape == (1, 512)


def test_token_type_ids_pattern(d2e_model, sam_features_1024):
    with torch.no_grad():
        result = d2e_model(sam_features_1024, output_attentions=True)
    _, _, _, token_type_ids = result
    # First 256 = image (0), next 256 = query (1)
    assert (token_type_ids[0, :256] == 0).all()
    assert (token_type_ids[0, 256:] == 1).all()


def test_attention_visualizer():
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
    from src.visualization.attention_viz import AttentionVisualizer

    model = build_qwen2_decoder_as_encoder(output_attentions=True)
    model.eval()
    x = torch.zeros(1, 896, 16, 16)
    with torch.no_grad():
        _, attentions, _, token_type_ids = model(x, output_attentions=True)

    viz = AttentionVisualizer(
        attention_weights=[a.cpu() for a in attentions],
        token_type_ids=token_type_ids.cpu(),
        spatial_size=16,
    )
    fig = viz.plot_attention_mask(layer=0, show_expected=False)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_attention_entropy():
    from src.visualization.utils import compute_attention_entropy
    # Uniform attention → maximum entropy
    attn = torch.ones(1, 1, 4, 4) / 4
    entropy = compute_attention_entropy(attn)
    assert entropy.shape == (1, 1, 4)
    # Entropy should be log(4) for uniform distribution
    import math
    assert abs(entropy.mean().item() - math.log(4)) < 0.01


def test_attention_analyzer():
    from src.models.qwen2_d2e import build_qwen2_decoder_as_encoder
    from src.analysis.attention_analysis import AttentionAnalyzer

    model = build_qwen2_decoder_as_encoder(output_attentions=True)
    model.eval()
    analyzer = AttentionAnalyzer(model)

    sam_features = torch.zeros(1, 896, 16, 16)
    with torch.no_grad():
        result = analyzer.extract_attention_patterns(sam_features)

    assert "attention_weights" in result
    assert len(result["attention_weights"]) == 24
    assert result["spatial_size"] == 16
