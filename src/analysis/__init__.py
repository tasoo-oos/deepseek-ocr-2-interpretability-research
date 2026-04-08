"""Analysis utilities with lazy imports to keep light-weight tools isolated."""

from importlib import import_module


_EXPORTS = {
    "AttentionAnalyzer": ".attention_analysis",
    "FeatureExtractor": ".feature_extractor",
    "InterventionManager": ".interventions",
    "CircuitDiscovery": ".circuits",
    "ProjectorAnalyzer": ".projector_analysis",
    "ProjectorSVD": ".projector_analysis",
    "QueryBankSummary": ".query_analysis",
    "QuerySpecializationAnalyzer": ".query_analysis",
    "ViewAblationAnalyzer": ".view_analysis",
    "ViewAblationResult": ".view_analysis",
    "LinearSpatialProbe": ".spatial_analysis",
    "SpatialProbeMetrics": ".spatial_analysis",
    "SparseAutoencoder": ".sparse_autoencoder",
    "SparseAutoencoderAnalyzer": ".sparse_autoencoder",
    "SparseAutoencoderMetrics": ".sparse_autoencoder",
    "SparseAutoencoderSummary": ".sparse_autoencoder",
    "SparseAutoencoderTrainer": ".sparse_autoencoder",
    "SparseFeatureSummary": ".sparse_autoencoder",
    "ablate_sparse_features": ".sparse_autoencoder",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)

__all__ = [
    "AttentionAnalyzer",
    "FeatureExtractor",
    "InterventionManager",
    "CircuitDiscovery",
    "ProjectorAnalyzer",
    "ProjectorSVD",
    "QueryBankSummary",
    "QuerySpecializationAnalyzer",
    "ViewAblationAnalyzer",
    "ViewAblationResult",
    "LinearSpatialProbe",
    "SpatialProbeMetrics",
    "SparseAutoencoder",
    "SparseAutoencoderAnalyzer",
    "SparseAutoencoderMetrics",
    "SparseAutoencoderSummary",
    "SparseAutoencoderTrainer",
    "SparseFeatureSummary",
    "ablate_sparse_features",
]
