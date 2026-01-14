"""
Unit tests for multimodal encoders.

Tests encoders for vision, language, and audio modalities.
"""

import pytest
import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    from ro_framework.multimodal.encoders import (
        VisionEncoder,
        LanguageEncoder,
        AudioEncoder,
        create_vision_encoder,
        create_language_encoder,
        create_audio_encoder,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestVisionEncoder:
    """Tests for VisionEncoder."""

    def test_vision_encoder_creation(self):
        """Test creating a vision encoder."""
        output_dofs = [
            PolarDoF(name=f"visual_{i}", description=f"Visual feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16)
        )

        encoder = VisionEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu"
        )

        assert encoder.modality_name == "vision"
        assert len(encoder.output_dofs) == 16
        assert encoder.device == "cpu"

    def test_vision_encoder_encode(self):
        """Test encoding an image."""
        output_dofs = [
            PolarDoF(name=f"visual_{i}", description=f"Visual feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16)
        )

        encoder = VisionEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu"
        )

        # Create test image
        image = np.random.randn(28, 28).astype(np.float32)

        # Encode
        state = encoder.encode(image)

        assert isinstance(state, State)
        assert len(state.dof_values) == 16

        # Check that values are floats
        for dof, value in state.dof_values.items():
            assert isinstance(value, float)

    def test_vision_encoder_uncertainty(self):
        """Test uncertainty estimation."""
        output_dofs = [
            PolarDoF(name=f"visual_{i}", description=f"Visual feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16)
        )

        encoder = VisionEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            use_dropout_uncertainty=False
        )

        image = np.random.randn(28, 28).astype(np.float32)

        # Get uncertainty
        uncertainties = encoder.get_uncertainty(image)

        assert len(uncertainties) == 16
        for dof, unc in uncertainties.items():
            assert isinstance(unc, float)
            assert unc >= 0.0

    def test_vision_encoder_mc_dropout(self):
        """Test MC Dropout uncertainty."""
        output_dofs = [
            PolarDoF(name=f"visual_{i}", description=f"Visual feature {i}")
            for i in range(8)
        ]

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 8)
        )

        encoder = VisionEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            use_dropout_uncertainty=True,
            dropout_samples=5
        )

        image = np.random.randn(28, 28).astype(np.float32)

        # Get uncertainty
        uncertainties = encoder.get_uncertainty(image)

        assert len(uncertainties) == 8
        for unc in uncertainties.values():
            assert unc >= 0.0

    def test_create_vision_encoder(self):
        """Test factory function for vision encoder."""
        encoder = create_vision_encoder(
            output_dim=16,
            architecture="resnet18",
            pretrained=False,
            device="cpu"
        )

        assert encoder is not None
        assert encoder.modality_name == "vision"
        assert len(encoder.output_dofs) == 16


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLanguageEncoder:
    """Tests for LanguageEncoder."""

    def test_language_encoder_creation(self):
        """Test creating a language encoder."""
        output_dofs = [
            PolarDoF(name=f"linguistic_{i}", description=f"Linguistic feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Linear(256, 16)
        )

        encoder = LanguageEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            max_length=256
        )

        assert encoder.modality_name == "language"
        assert len(encoder.output_dofs) == 16
        assert encoder.max_length == 256

    def test_language_encoder_encode(self):
        """Test encoding text."""
        output_dofs = [
            PolarDoF(name=f"linguistic_{i}", description=f"Linguistic feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Linear(256, 16)
        )

        encoder = LanguageEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            max_length=256
        )

        # Encode text
        text = "Hello, this is a test message"
        state = encoder.encode(text)

        assert isinstance(state, State)
        assert len(state.dof_values) == 16

    def test_language_encoder_uncertainty(self):
        """Test uncertainty estimation for language."""
        output_dofs = [
            PolarDoF(name=f"linguistic_{i}", description=f"Linguistic feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Linear(256, 16)
        )

        encoder = LanguageEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            max_length=256,
            use_dropout_uncertainty=False
        )

        text = "Test text for uncertainty"
        uncertainties = encoder.get_uncertainty(text)

        assert len(uncertainties) == 16
        for unc in uncertainties.values():
            assert unc >= 0.0

    def test_create_language_encoder(self):
        """Test factory function for language encoder."""
        encoder = create_language_encoder(
            output_dim=16,
            vocab_size=5000,
            embedding_dim=64,
            hidden_dim=128,
            device="cpu"
        )

        assert encoder is not None
        assert encoder.modality_name == "language"
        assert len(encoder.output_dofs) == 16


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAudioEncoder:
    """Tests for AudioEncoder."""

    def test_audio_encoder_creation(self):
        """Test creating an audio encoder."""
        output_dofs = [
            PolarDoF(name=f"audio_{i}", description=f"Audio feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16)
        )

        encoder = AudioEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            sample_rate=16000
        )

        assert encoder.modality_name == "audio"
        assert len(encoder.output_dofs) == 16
        assert encoder.sample_rate == 16000

    def test_audio_encoder_encode(self):
        """Test encoding audio."""
        output_dofs = [
            PolarDoF(name=f"audio_{i}", description=f"Audio feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16)
        )

        encoder = AudioEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu"
        )

        # Create test audio
        audio = np.random.randn(1000).astype(np.float32)

        # Encode
        state = encoder.encode(audio)

        assert isinstance(state, State)
        assert len(state.dof_values) == 16

    def test_audio_encoder_uncertainty(self):
        """Test uncertainty estimation for audio."""
        output_dofs = [
            PolarDoF(name=f"audio_{i}", description=f"Audio feature {i}")
            for i in range(16)
        ]

        model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16)
        )

        encoder = AudioEncoder(
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            use_dropout_uncertainty=False
        )

        audio = np.random.randn(1000).astype(np.float32)
        uncertainties = encoder.get_uncertainty(audio)

        assert len(uncertainties) == 16
        for unc in uncertainties.values():
            assert unc >= 0.0

    def test_create_audio_encoder(self):
        """Test factory function for audio encoder."""
        encoder = create_audio_encoder(
            output_dim=16,
            input_channels=1,
            device="cpu"
        )

        assert encoder is not None
        assert encoder.modality_name == "audio"
        assert len(encoder.output_dofs) == 16
