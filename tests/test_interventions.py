"""Unit tests for intervention tools."""

import pytest
import torch
from PIL import Image


@pytest.fixture
def model():
    from src.models.deepseek_ocr import DeepseekOCRModel
    m = DeepseekOCRModel()
    m.eval()
    return m


@pytest.fixture
def dummy_inputs():
    from src.preprocessing.image_transforms import ImageProcessor
    processor = ImageProcessor()
    image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    inputs = processor.process_image(image)
    return inputs


def test_feature_extractor_hooks(model, dummy_inputs):
    from src.analysis.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor(model)
    extractor.register_hooks(sam_layers=[0, 11], d2e_layers=[0, 23], projector=True)

    acts = extractor.extract(
        dummy_inputs["pixel_values"],
        dummy_inputs["images_crop"],
        dummy_inputs["images_spatial_crop"],
    )
    extractor.clear_hooks()

    assert "sam_layer_0" in acts
    assert "sam_layer_11" in acts
    assert "d2e_layer_0" in acts
    assert "d2e_layer_23" in acts
    assert "projector" in acts


def test_feature_extractor_preserves_repeated_d2e_calls(model):
    from src.analysis.feature_extractor import FeatureExtractor
    from src.preprocessing.image_transforms import ImageProcessor

    processor = ImageProcessor(crop_mode=True)
    image = Image.new("RGB", (1536, 768), color=(192, 192, 192))
    inputs = processor.process_image(image)

    extractor = FeatureExtractor(model)
    extractor.register_hooks(d2e_layers=[0], projector=False)
    acts = extractor.extract(
        inputs["pixel_values"],
        inputs["images_crop"],
        inputs["images_spatial_crop"],
    )

    sequence = extractor.get_activation_sequence("d2e_layer_0")
    extractor.clear_hooks()

    assert "d2e_layer_0__call_0" in acts
    assert "d2e_layer_0__call_1" in acts
    assert len(sequence) == 2
    assert sequence[0].shape[0] > 1  # local crops
    assert sequence[1].shape[0] == 1  # global page
    assert sequence[0].shape[1] == 288  # 144 image + 144 query tokens
    assert sequence[1].shape[1] == 512  # 256 image + 256 query tokens


def test_intervention_manager_context(model, dummy_inputs):
    from src.analysis.interventions import InterventionManager

    def get_embedding_norm():
        with torch.no_grad():
            out = model.get_multimodal_embeddings(
                dummy_inputs["pixel_values"],
                dummy_inputs["images_crop"],
                dummy_inputs["images_spatial_crop"],
            )
        return sum(e.norm().item() for e in out)

    baseline_norm = get_embedding_norm()

    # Applying and removing intervention should restore baseline
    with InterventionManager(model) as mgr:
        mgr.ablate_query_tokens()
        ablated_norm = get_embedding_norm()

    restored_norm = get_embedding_norm()

    # Ablation should change the output
    assert ablated_norm != pytest.approx(baseline_norm, abs=1e-3)
    # After context exit, hooks are removed and output is restored
    assert restored_norm == pytest.approx(baseline_norm, abs=1e-3)


def test_head_ablation_changes_output(model, dummy_inputs):
    from src.analysis.interventions import InterventionManager

    with torch.no_grad():
        baseline = model.get_multimodal_embeddings(
            dummy_inputs["pixel_values"],
            dummy_inputs["images_crop"],
            dummy_inputs["images_spatial_crop"],
        )
        baseline_tensor = torch.cat(baseline, dim=0)

    with InterventionManager(model) as mgr:
        mgr.ablate_attention_head(layer=0, head=0, component="d2e")
        with torch.no_grad():
            ablated = model.get_multimodal_embeddings(
                dummy_inputs["pixel_values"],
                dummy_inputs["images_crop"],
                dummy_inputs["images_spatial_crop"],
            )
        ablated_tensor = torch.cat(ablated, dim=0)

    # Output should differ after ablation
    assert not torch.allclose(baseline_tensor, ablated_tensor, atol=1e-6)


def test_layer_query_state_ablation_changes_output(model, dummy_inputs):
    from src.analysis.interventions import InterventionManager

    pixel_values = dummy_inputs["pixel_values"][0].to(dtype=torch.bfloat16)

    with torch.no_grad():
        sam_features = model.sam_model(pixel_values)
        baseline = model.qwen2_model(sam_features)

    with InterventionManager(model) as mgr:
        mgr.ablate_query_states_in_layer(layer=0, start_idx=0, end_idx=16)
        with torch.no_grad():
            ablated = model.qwen2_model(sam_features)

    assert baseline.shape == ablated.shape
    assert not torch.allclose(baseline, ablated, atol=1e-6)


def test_sae_feature_ablation_changes_output(model, dummy_inputs):
    from src.analysis.interventions import InterventionManager
    from src.analysis.sparse_autoencoder import SparseAutoencoder

    pixel_values = dummy_inputs["pixel_values"][0].to(dtype=torch.bfloat16)

    with torch.no_grad():
        sam_features = model.sam_model(pixel_values)
        baseline = model.qwen2_model(sam_features)

    sae = SparseAutoencoder(input_dim=baseline.shape[-1], n_features=16)

    with InterventionManager(model) as mgr:
        mgr.ablate_sae_features_in_query_states(layer=0, sae=sae, feature_indices=[0, 1])
        with torch.no_grad():
            ablated = model.qwen2_model(sam_features)

    assert baseline.shape == ablated.shape
    assert not torch.allclose(baseline, ablated, atol=1e-6)
