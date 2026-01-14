"""
Unit tests for fusion strategies.

Tests concatenation, attention, and gated fusion.
"""

import pytest
import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    from ro_framework.multimodal.fusion import (
        ConcatenationFusion,
        AttentionFusion,
        GatedFusion,
        MultimodalObserver,
    )
    from ro_framework.multimodal.encoders import VisionEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestConcatenationFusion:
    """Tests for ConcatenationFusion."""

    def test_concatenation_fusion_creation(self):
        """Test creating a concatenation fusion."""
        modality_names = ["vision", "language"]
        modality_dofs = {
            "vision": [PolarDoF(name=f"v{i}", description="") for i in range(8)],
            "language": [PolarDoF(name=f"l{i}", description="") for i in range(8)],
        }

        fusion = ConcatenationFusion(
            modality_names=modality_names,
            modality_dofs=modality_dofs
        )

        assert fusion.modality_names == modality_names
        assert len(fusion.output_dofs) == 16  # 8 + 8

    def test_concatenation_fusion_fuse(self):
        """Test fusing states."""
        modality_dofs = {
            "vision": [PolarDoF(name=f"v{i}", description="") for i in range(4)],
            "language": [PolarDoF(name=f"l{i}", description="") for i in range(4)],
        }

        fusion = ConcatenationFusion(
            modality_names=["vision", "language"],
            modality_dofs=modality_dofs
        )

        # Create test states
        vision_state = State(dof_values={
            dof: float(i) for i, dof in enumerate(modality_dofs["vision"])
        })
        language_state = State(dof_values={
            dof: float(i + 10) for i, dof in enumerate(modality_dofs["language"])
        })

        modality_states = {
            "vision": vision_state,
            "language": language_state
        }

        # Fuse
        fused = fusion.fuse(modality_states)

        assert isinstance(fused, State)
        assert len(fused.dof_values) == 8

    def test_concatenation_fusion_weights(self):
        """Test weighted fusion."""
        modality_dofs = {
            "vision": [PolarDoF(name=f"v{i}", description="") for i in range(2)],
            "language": [PolarDoF(name=f"l{i}", description="") for i in range(2)],
        }

        fusion = ConcatenationFusion(
            modality_names=["vision", "language"],
            modality_dofs=modality_dofs,
            weights={"vision": 1.0, "language": 0.5}
        )

        vision_state = State(dof_values={
            modality_dofs["vision"][0]: 1.0,
            modality_dofs["vision"][1]: 1.0,
        })
        language_state = State(dof_values={
            modality_dofs["language"][0]: 1.0,
            modality_dofs["language"][1]: 1.0,
        })

        fused = fusion.fuse({
            "vision": vision_state,
            "language": language_state
        })

        # Language values should be weighted by 0.5
        lang_value = fused.get_value(modality_dofs["language"][0])
        assert lang_value == 0.5

    def test_concatenation_fusion_uncertainties(self):
        """Test fusing uncertainties."""
        modality_dofs = {
            "vision": [PolarDoF(name=f"v{i}", description="") for i in range(4)],
        }

        fusion = ConcatenationFusion(
            modality_names=["vision"],
            modality_dofs=modality_dofs
        )

        uncertainties = {
            "vision": {dof: 0.1 for dof in modality_dofs["vision"]}
        }

        fused_unc = fusion.fuse_uncertainties(uncertainties)

        assert len(fused_unc) == 4
        for unc in fused_unc.values():
            assert unc == 0.1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAttentionFusion:
    """Tests for AttentionFusion."""

    def test_attention_fusion_creation(self):
        """Test creating attention fusion."""
        modality_dims = {"vision": 16, "language": 16}

        fusion = AttentionFusion(
            modality_names=["vision", "language"],
            modality_dims=modality_dims,
            output_dim=32,
            hidden_dim=64,
            device="cpu"
        )

        assert fusion.output_dim == 32
        assert len(fusion.output_dofs) == 32

    def test_attention_fusion_fuse(self):
        """Test attention-based fusion."""
        modality_dims = {"vision": 8, "language": 8}

        fusion = AttentionFusion(
            modality_names=["vision", "language"],
            modality_dims=modality_dims,
            output_dim=16,
            device="cpu"
        )

        # Create test states
        vision_dofs = [PolarDoF(name=f"v{i}", description="") for i in range(8)]
        language_dofs = [PolarDoF(name=f"l{i}", description="") for i in range(8)]

        vision_state = State(dof_values={
            dof: float(np.random.randn()) for dof in vision_dofs
        })
        language_state = State(dof_values={
            dof: float(np.random.randn()) for dof in language_dofs
        })

        modality_states = {
            "vision": vision_state,
            "language": language_state
        }

        # Fuse
        fused = fusion.fuse(modality_states)

        assert isinstance(fused, State)
        assert len(fused.dof_values) == 16

    def test_attention_fusion_uncertainties(self):
        """Test fusing uncertainties with attention."""
        modality_dims = {"vision": 8}

        fusion = AttentionFusion(
            modality_names=["vision"],
            modality_dims=modality_dims,
            output_dim=16,
            device="cpu"
        )

        vision_dofs = [PolarDoF(name=f"v{i}", description="") for i in range(8)]
        uncertainties = {
            "vision": {dof: 0.2 for dof in vision_dofs}
        }

        fused_unc = fusion.fuse_uncertainties(uncertainties)

        assert len(fused_unc) == 16
        for unc in fused_unc.values():
            assert unc >= 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGatedFusion:
    """Tests for GatedFusion."""

    def test_gated_fusion_creation(self):
        """Test creating gated fusion."""
        modality_dims = {"vision": 16, "audio": 16}

        fusion = GatedFusion(
            modality_names=["vision", "audio"],
            modality_dims=modality_dims,
            output_dim=32,
            device="cpu"
        )

        assert fusion.output_dim == 32
        assert len(fusion.output_dofs) == 32

    def test_gated_fusion_fuse(self):
        """Test gated fusion."""
        modality_dims = {"vision": 8}

        fusion = GatedFusion(
            modality_names=["vision"],
            modality_dims=modality_dims,
            output_dim=16,
            device="cpu"
        )

        vision_dofs = [PolarDoF(name=f"v{i}", description="") for i in range(8)]
        vision_state = State(dof_values={
            dof: float(np.random.randn()) for dof in vision_dofs
        })

        fused = fusion.fuse({"vision": vision_state})

        assert isinstance(fused, State)
        assert len(fused.dof_values) == 16


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultimodalObserver:
    """Tests for MultimodalObserver."""

    def test_multimodal_observer_creation(self):
        """Test creating a multimodal observer."""
        # Create simple encoders
        vision_dofs = [PolarDoF(name=f"v{i}", description="") for i in range(8)]
        vision_model = nn.Sequential(nn.Flatten(), nn.Linear(64, 8))
        vision_encoder = VisionEncoder(
            output_dofs=vision_dofs,
            model=vision_model,
            device="cpu"
        )

        # Create fusion
        modality_dofs = {"vision": vision_dofs}
        fusion = ConcatenationFusion(
            modality_names=["vision"],
            modality_dofs=modality_dofs
        )

        # Create observer
        observer = MultimodalObserver(
            name="TestObserver",
            encoders={"vision": vision_encoder},
            fusion_strategy=fusion
        )

        assert observer.name == "TestObserver"
        assert "vision" in observer.encoders
        assert len(observer.external_dofs) == 8

    def test_multimodal_observer_process_input(self):
        """Test processing multimodal input."""
        vision_dofs = [PolarDoF(name=f"v{i}", description="") for i in range(8)]
        vision_model = nn.Sequential(nn.Flatten(), nn.Linear(64, 8))
        vision_encoder = VisionEncoder(
            output_dofs=vision_dofs,
            model=vision_model,
            device="cpu"
        )

        modality_dofs = {"vision": vision_dofs}
        fusion = ConcatenationFusion(
            modality_names=["vision"],
            modality_dofs=modality_dofs
        )

        observer = MultimodalObserver(
            name="TestObserver",
            encoders={"vision": vision_encoder},
            fusion_strategy=fusion
        )

        # Process input
        image = np.random.randn(8, 8).astype(np.float32)
        fused_state, fused_unc = observer.process_multimodal_input(
            {"vision": image}
        )

        assert isinstance(fused_state, State)
        assert len(fused_state.dof_values) == 8
        assert len(fused_unc) == 8

    def test_multimodal_observer_observe(self):
        """Test observing with world model."""
        vision_dofs = [PolarDoF(name=f"v{i}", description="") for i in range(4)]
        vision_model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4))
        vision_encoder = VisionEncoder(
            output_dofs=vision_dofs,
            model=vision_model,
            device="cpu"
        )

        modality_dofs = {"vision": vision_dofs}
        fusion = ConcatenationFusion(
            modality_names=["vision"],
            modality_dofs=modality_dofs
        )

        # Create world model
        from ro_framework.integration.torch import TorchNeuralMapping, create_mlp

        internal_dofs = [PolarDoF(name=f"internal{i}", description="") for i in range(4)]
        world_model_nn = create_mlp(4, 4, [8])

        world_model = TorchNeuralMapping(
            name="world_model",
            input_dofs=vision_dofs,
            output_dofs=internal_dofs,
            model=world_model_nn,
            device="cpu"
        )

        observer = MultimodalObserver(
            name="TestObserver",
            encoders={"vision": vision_encoder},
            fusion_strategy=fusion,
            world_model=world_model
        )

        # Observe
        image = np.random.randn(4, 4).astype(np.float32)
        internal_state = observer.observe_multimodal({"vision": image})

        assert isinstance(internal_state, State)
        assert len(internal_state.dof_values) == 4
