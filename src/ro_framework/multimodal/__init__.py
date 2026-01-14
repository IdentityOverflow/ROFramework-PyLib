"""
Multimodal integration for the Recursive Observer Framework.

This module provides:
- Encoders for different modalities (vision, language, audio)
- Cross-modal fusion mechanisms
- Training protocols and active learning
- Comprehensive uncertainty quantification
"""

from ro_framework.multimodal.encoders import (
    ModalityEncoder,
    VisionEncoder,
    LanguageEncoder,
    AudioEncoder,
    create_vision_encoder,
    create_language_encoder,
    create_audio_encoder,
)

from ro_framework.multimodal.fusion import (
    FusionStrategy,
    ConcatenationFusion,
    AttentionFusion,
    GatedFusion,
    MultimodalObserver,
)

from ro_framework.multimodal.training import (
    TrainingProtocol,
    SupervisedTraining,
    SelfSupervisedTraining,
    ActiveLearningStrategy,
    UncertaintyBasedSampling,
    DiversityBasedSampling,
)

from ro_framework.multimodal.uncertainty import (
    UncertaintyQuantifier,
    EnsembleUncertainty,
    BayesianUncertainty,
    CalibrationMetrics,
)

__all__ = [
    # Encoders
    "ModalityEncoder",
    "VisionEncoder",
    "LanguageEncoder",
    "AudioEncoder",
    "create_vision_encoder",
    "create_language_encoder",
    "create_audio_encoder",
    # Fusion
    "FusionStrategy",
    "ConcatenationFusion",
    "AttentionFusion",
    "GatedFusion",
    "MultimodalObserver",
    # Training
    "TrainingProtocol",
    "SupervisedTraining",
    "SelfSupervisedTraining",
    "ActiveLearningStrategy",
    "UncertaintyBasedSampling",
    "DiversityBasedSampling",
    # Uncertainty
    "UncertaintyQuantifier",
    "EnsembleUncertainty",
    "BayesianUncertainty",
    "CalibrationMetrics",
]
