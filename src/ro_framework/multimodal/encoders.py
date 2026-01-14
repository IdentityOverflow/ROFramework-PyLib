"""
Multimodal encoders for different sensory modalities.

Each encoder maps raw sensory data to a common representational space
(internal DoFs), allowing the observer to process multiple modalities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ro_framework.core.dof import DoF, PolarDoF, ScalarDoF
from ro_framework.core.state import State


class ModalityEncoder(ABC):
    """
    Abstract base class for modality-specific encoders.

    An encoder transforms raw sensory input from a specific modality
    (vision, language, audio) into a State representation over internal DoFs.
    """

    def __init__(
        self,
        modality_name: str,
        output_dofs: List[DoF],
        resolution: Optional[Dict[DoF, float]] = None,
    ):
        """
        Initialize modality encoder.

        Args:
            modality_name: Name of the modality (e.g., "vision", "language")
            output_dofs: Internal DoFs that represent encoded information
            resolution: Resolution limits for each output DoF
        """
        self.modality_name = modality_name
        self.output_dofs = output_dofs
        self.resolution = resolution or {}

    @abstractmethod
    def encode(self, raw_input: Any) -> State:
        """
        Encode raw sensory input into a State.

        Args:
            raw_input: Raw input data (e.g., image array, text string, audio waveform)

        Returns:
            State representing the encoded information
        """
        pass

    @abstractmethod
    def get_uncertainty(self, raw_input: Any) -> Dict[DoF, float]:
        """
        Estimate encoding uncertainty for each output DoF.

        Args:
            raw_input: Raw input data

        Returns:
            Dictionary mapping DoFs to uncertainty estimates
        """
        pass


if TORCH_AVAILABLE:

    class VisionEncoder(ModalityEncoder):
        """
        Vision encoder using convolutional neural networks.

        Processes images/video frames into internal representations.
        Supports both pre-trained models (transfer learning) and custom architectures.
        """

        def __init__(
            self,
            output_dofs: List[DoF],
            model: nn.Module,
            preprocessor: Optional[Any] = None,
            device: str = "cpu",
            use_dropout_uncertainty: bool = False,
            dropout_samples: int = 10,
            resolution: Optional[Dict[DoF, float]] = None,
        ):
            """
            Initialize vision encoder.

            Args:
                output_dofs: Internal DoFs for visual features
                model: PyTorch model for encoding (e.g., ResNet, ViT)
                preprocessor: Optional preprocessor for input images
                device: Device for computation ("cpu" or "cuda")
                use_dropout_uncertainty: Whether to use MC Dropout for uncertainty
                dropout_samples: Number of samples for MC Dropout
                resolution: Resolution limits for each DoF
            """
            super().__init__("vision", output_dofs, resolution)
            self.model = model.to(device)
            self.preprocessor = preprocessor
            self.device = device
            self.use_dropout_uncertainty = use_dropout_uncertainty
            self.dropout_samples = dropout_samples
            self.model.eval()

        def encode(self, raw_input: np.ndarray) -> State:
            """
            Encode image into State.

            Args:
                raw_input: Image array (H, W, C) or (C, H, W)

            Returns:
                State with visual features
            """
            # Preprocess if needed
            if self.preprocessor is not None:
                processed = self.preprocessor(raw_input)
            else:
                processed = torch.from_numpy(raw_input).float()

            # Ensure batch dimension
            if processed.dim() == 2:  # (H, W) -> (1, H, W) or (1, H*W)
                processed = processed.unsqueeze(0)
            elif processed.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                processed = processed.unsqueeze(0)

            processed = processed.to(self.device)

            # Forward pass
            with torch.no_grad():
                features = self.model(processed)

            # Convert to numpy
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy().flatten()

            # Create state from features
            state_dict = {}
            for i, dof in enumerate(self.output_dofs):
                if i < len(features):
                    state_dict[dof] = float(features[i])

            return State(values=state_dict)

        def get_uncertainty(self, raw_input: np.ndarray) -> Dict[DoF, float]:
            """
            Estimate uncertainty using MC Dropout or other methods.

            Args:
                raw_input: Image array

            Returns:
                Uncertainty estimates for each DoF
            """
            if not self.use_dropout_uncertainty:
                # Simple heuristic: uncertainty proportional to feature magnitude
                state = self.encode(raw_input)
                uncertainties = {}
                for dof in self.output_dofs:
                    value = state.get_value(dof)
                    if value is not None:
                        uncertainties[dof] = abs(float(value)) * 0.1
                    else:
                        uncertainties[dof] = 1.0
                return uncertainties

            # MC Dropout uncertainty
            self.model.train()  # Enable dropout

            if self.preprocessor is not None:
                processed = self.preprocessor(raw_input)
            else:
                processed = torch.from_numpy(raw_input).float()

            if processed.dim() == 2:  # (H, W) -> (1, H, W)
                processed = processed.unsqueeze(0)
            elif processed.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                processed = processed.unsqueeze(0)

            processed = processed.to(self.device)

            # Multiple forward passes
            samples = []
            for _ in range(self.dropout_samples):
                with torch.no_grad():
                    features = self.model(processed)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy().flatten()
                samples.append(features)

            self.model.eval()  # Disable dropout

            # Compute uncertainties as standard deviation
            samples_array = np.array(samples)
            uncertainties = {}
            for i, dof in enumerate(self.output_dofs):
                if i < samples_array.shape[1]:
                    uncertainties[dof] = float(np.std(samples_array[:, i]))
                else:
                    uncertainties[dof] = 1.0

            return uncertainties


    class LanguageEncoder(ModalityEncoder):
        """
        Language encoder using transformer models.

        Processes text into internal representations.
        Supports pre-trained language models (BERT, GPT, etc.) and custom architectures.
        """

        def __init__(
            self,
            output_dofs: List[DoF],
            model: nn.Module,
            tokenizer: Optional[Any] = None,
            device: str = "cpu",
            max_length: int = 512,
            use_dropout_uncertainty: bool = False,
            dropout_samples: int = 10,
            resolution: Optional[Dict[DoF, float]] = None,
        ):
            """
            Initialize language encoder.

            Args:
                output_dofs: Internal DoFs for linguistic features
                model: PyTorch model for encoding (e.g., BERT, GPT)
                tokenizer: Tokenizer for text preprocessing
                device: Device for computation
                max_length: Maximum sequence length
                use_dropout_uncertainty: Whether to use MC Dropout
                dropout_samples: Number of samples for MC Dropout
                resolution: Resolution limits for each DoF
            """
            super().__init__("language", output_dofs, resolution)
            self.model = model.to(device)
            self.tokenizer = tokenizer
            self.device = device
            self.max_length = max_length
            self.use_dropout_uncertainty = use_dropout_uncertainty
            self.dropout_samples = dropout_samples
            self.model.eval()

        def encode(self, raw_input: str) -> State:
            """
            Encode text into State.

            Args:
                raw_input: Text string

            Returns:
                State with linguistic features
            """
            # Tokenize if tokenizer available
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    raw_input,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            else:
                # Simple character-level encoding
                chars = [ord(c) / 255.0 for c in raw_input[:self.max_length]]
                chars += [0.0] * (self.max_length - len(chars))
                tokens = torch.tensor([chars], device=self.device)

            # Forward pass
            with torch.no_grad():
                if isinstance(tokens, dict):
                    features = self.model(**tokens)
                else:
                    features = self.model(tokens)

            # Extract features (may need pooling)
            if isinstance(features, tuple):
                features = features[0]  # Take first output

            if isinstance(features, torch.Tensor):
                # Pool if sequence output
                if features.dim() == 3:
                    features = features.mean(dim=1)  # Average pooling
                features = features.cpu().numpy().flatten()

            # Create state from features
            state_dict = {}
            for i, dof in enumerate(self.output_dofs):
                if i < len(features):
                    state_dict[dof] = float(features[i])

            return State(values=state_dict)

        def get_uncertainty(self, raw_input: str) -> Dict[DoF, float]:
            """
            Estimate uncertainty for text encoding.

            Args:
                raw_input: Text string

            Returns:
                Uncertainty estimates for each DoF
            """
            if not self.use_dropout_uncertainty:
                # Simple heuristic based on text length and feature magnitude
                state = self.encode(raw_input)
                text_len_factor = min(len(raw_input) / self.max_length, 1.0)
                uncertainties = {}
                for dof in self.output_dofs:
                    value = state.get_value(dof)
                    if value is not None:
                        uncertainties[dof] = (1.0 - text_len_factor) * 0.5
                    else:
                        uncertainties[dof] = 1.0
                return uncertainties

            # MC Dropout uncertainty
            self.model.train()

            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    raw_input,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            else:
                chars = [ord(c) / 255.0 for c in raw_input[:self.max_length]]
                chars += [0.0] * (self.max_length - len(chars))
                tokens = torch.tensor([chars], device=self.device)

            samples = []
            for _ in range(self.dropout_samples):
                with torch.no_grad():
                    if isinstance(tokens, dict):
                        features = self.model(**tokens)
                    else:
                        features = self.model(tokens)

                    if isinstance(features, tuple):
                        features = features[0]
                    if isinstance(features, torch.Tensor):
                        if features.dim() == 3:
                            features = features.mean(dim=1)
                        features = features.cpu().numpy().flatten()
                    samples.append(features)

            self.model.eval()

            samples_array = np.array(samples)
            uncertainties = {}
            for i, dof in enumerate(self.output_dofs):
                if i < samples_array.shape[1]:
                    uncertainties[dof] = float(np.std(samples_array[:, i]))
                else:
                    uncertainties[dof] = 1.0

            return uncertainties


    class AudioEncoder(ModalityEncoder):
        """
        Audio encoder using neural networks.

        Processes audio waveforms or spectrograms into internal representations.
        Supports various audio processing architectures (CNNs, RNNs, Transformers).
        """

        def __init__(
            self,
            output_dofs: List[DoF],
            model: nn.Module,
            preprocessor: Optional[Any] = None,
            device: str = "cpu",
            sample_rate: int = 16000,
            use_dropout_uncertainty: bool = False,
            dropout_samples: int = 10,
            resolution: Optional[Dict[DoF, float]] = None,
        ):
            """
            Initialize audio encoder.

            Args:
                output_dofs: Internal DoFs for audio features
                model: PyTorch model for encoding
                preprocessor: Optional preprocessor for audio (e.g., MFCC)
                device: Device for computation
                sample_rate: Audio sample rate
                use_dropout_uncertainty: Whether to use MC Dropout
                dropout_samples: Number of samples for MC Dropout
                resolution: Resolution limits for each DoF
            """
            super().__init__("audio", output_dofs, resolution)
            self.model = model.to(device)
            self.preprocessor = preprocessor
            self.device = device
            self.sample_rate = sample_rate
            self.use_dropout_uncertainty = use_dropout_uncertainty
            self.dropout_samples = dropout_samples
            self.model.eval()

        def encode(self, raw_input: np.ndarray) -> State:
            """
            Encode audio into State.

            Args:
                raw_input: Audio waveform (1D array) or spectrogram (2D array)

            Returns:
                State with audio features
            """
            # Preprocess if needed
            if self.preprocessor is not None:
                processed = self.preprocessor(raw_input, sample_rate=self.sample_rate)
            else:
                processed = torch.from_numpy(raw_input).float()

            # Ensure correct dimensions
            if processed.dim() == 1:
                processed = processed.unsqueeze(0).unsqueeze(0)  # Add batch and channel
            elif processed.dim() == 2:
                processed = processed.unsqueeze(0)  # Add batch

            processed = processed.to(self.device)

            # Forward pass
            with torch.no_grad():
                features = self.model(processed)

            # Convert to numpy
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy().flatten()

            # Create state from features
            state_dict = {}
            for i, dof in enumerate(self.output_dofs):
                if i < len(features):
                    state_dict[dof] = float(features[i])

            return State(values=state_dict)

        def get_uncertainty(self, raw_input: np.ndarray) -> Dict[DoF, float]:
            """
            Estimate uncertainty for audio encoding.

            Args:
                raw_input: Audio waveform or spectrogram

            Returns:
                Uncertainty estimates for each DoF
            """
            if not self.use_dropout_uncertainty:
                # Simple heuristic based on signal energy
                signal_energy = np.mean(raw_input ** 2)
                base_uncertainty = 0.1 + 0.4 * (1.0 - min(signal_energy, 1.0))

                uncertainties = {}
                for dof in self.output_dofs:
                    uncertainties[dof] = base_uncertainty
                return uncertainties

            # MC Dropout uncertainty
            self.model.train()

            if self.preprocessor is not None:
                processed = self.preprocessor(raw_input, sample_rate=self.sample_rate)
            else:
                processed = torch.from_numpy(raw_input).float()

            if processed.dim() == 1:
                processed = processed.unsqueeze(0).unsqueeze(0)
            elif processed.dim() == 2:
                processed = processed.unsqueeze(0)

            processed = processed.to(self.device)

            samples = []
            for _ in range(self.dropout_samples):
                with torch.no_grad():
                    features = self.model(processed)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy().flatten()
                samples.append(features)

            self.model.eval()

            samples_array = np.array(samples)
            uncertainties = {}
            for i, dof in enumerate(self.output_dofs):
                if i < samples_array.shape[1]:
                    uncertainties[dof] = float(np.std(samples_array[:, i]))
                else:
                    uncertainties[dof] = 1.0

            return uncertainties


def create_vision_encoder(
    output_dim: int,
    architecture: str = "resnet18",
    pretrained: bool = True,
    device: str = "cpu",
    **kwargs
) -> Optional["VisionEncoder"]:
    """
    Create a vision encoder with a standard architecture.

    Args:
        output_dim: Dimension of output features
        architecture: Model architecture (resnet18, resnet50, vit, etc.)
        pretrained: Whether to use pretrained weights
        device: Device for computation
        **kwargs: Additional arguments for VisionEncoder

    Returns:
        VisionEncoder instance or None if torch not available
    """
    if not TORCH_AVAILABLE:
        return None

    import torchvision.models as models

    # Create output DoFs
    output_dofs = [
        PolarDoF(name=f"visual_feature_{i}", description=f"Visual feature dimension {i}")
        for i in range(output_dim)
    ]

    # Load model
    if architecture == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return VisionEncoder(
        output_dofs=output_dofs,
        model=model,
        device=device,
        **kwargs
    )


def create_language_encoder(
    output_dim: int,
    vocab_size: int = 10000,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    device: str = "cpu",
    **kwargs
) -> Optional["LanguageEncoder"]:
    """
    Create a simple language encoder.

    Args:
        output_dim: Dimension of output features
        vocab_size: Vocabulary size
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of encoder
        device: Device for computation
        **kwargs: Additional arguments for LanguageEncoder

    Returns:
        LanguageEncoder instance or None if torch not available
    """
    if not TORCH_AVAILABLE:
        return None

    # Create output DoFs
    output_dofs = [
        PolarDoF(name=f"linguistic_feature_{i}", description=f"Linguistic feature dimension {i}")
        for i in range(output_dim)
    ]

    # Simple LSTM-based encoder
    class SimpleLSTMEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)
            output = self.fc(hidden.squeeze(0))
            return output

    model = SimpleLSTMEncoder()

    return LanguageEncoder(
        output_dofs=output_dofs,
        model=model,
        device=device,
        **kwargs
    )


def create_audio_encoder(
    output_dim: int,
    input_channels: int = 1,
    device: str = "cpu",
    **kwargs
) -> Optional["AudioEncoder"]:
    """
    Create a simple audio encoder.

    Args:
        output_dim: Dimension of output features
        input_channels: Number of input channels
        device: Device for computation
        **kwargs: Additional arguments for AudioEncoder

    Returns:
        AudioEncoder instance or None if torch not available
    """
    if not TORCH_AVAILABLE:
        return None

    # Create output DoFs
    output_dofs = [
        PolarDoF(name=f"audio_feature_{i}", description=f"Audio feature dimension {i}")
        for i in range(output_dim)
    ]

    # Simple CNN-based encoder
    class SimpleCNNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, output_dim)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            return x

    model = SimpleCNNEncoder()

    return AudioEncoder(
        output_dofs=output_dofs,
        model=model,
        device=device,
        **kwargs
    )
