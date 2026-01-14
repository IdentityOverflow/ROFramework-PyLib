"""Unit tests for PyTorch integration."""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF, PolarDoFType
from ro_framework.core.state import State

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    from ro_framework.integration.torch import (
        TorchNeuralMapping,
        TorchObserver,
        create_mlp,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.fixture
def simple_dofs():
    """Create simple DoFs for testing."""
    input_dofs = [
        PolarDoF(
            name=f"input_{i}",
            pole_negative=-1.0,
            pole_positive=1.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(3)
    ]
    output_dofs = [
        PolarDoF(
            name=f"output_{i}",
            pole_negative=-5.0,
            pole_positive=5.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(5)
    ]
    return input_dofs, output_dofs


@pytest.fixture
def simple_model():
    """Create simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )


class TestCreateMLP:
    """Test MLP creation helper."""

    def test_create_basic_mlp(self):
        """Test creating basic MLP."""
        model = create_mlp(
            input_dim=10,
            output_dim=5,
            hidden_dims=[20, 15]
        )

        assert isinstance(model, nn.Module)

        # Test forward pass
        x = torch.randn(1, 10)
        y = model(x)
        assert y.shape == (1, 5)

    def test_create_mlp_with_dropout(self):
        """Test MLP with dropout."""
        model = create_mlp(
            input_dim=10,
            output_dim=5,
            hidden_dims=[20],
            dropout=0.3
        )

        # Check that dropout is present
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
        assert has_dropout

    def test_create_mlp_with_batch_norm(self):
        """Test MLP with batch normalization."""
        model = create_mlp(
            input_dim=10,
            output_dim=5,
            hidden_dims=[20],
            batch_norm=True
        )

        # Check that batch norm is present
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
        assert has_bn

    def test_create_mlp_activation_types(self):
        """Test different activation functions."""
        for activation in ['relu', 'tanh', 'gelu', 'sigmoid']:
            model = create_mlp(
                input_dim=10,
                output_dim=5,
                hidden_dims=[20],
                activation=activation
            )
            assert isinstance(model, nn.Module)


class TestTorchNeuralMapping:
    """Test TorchNeuralMapping implementation."""

    def test_creation(self, simple_dofs, simple_model):
        """Test creating a TorchNeuralMapping."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="test_mapping",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        assert mapping.name == "test_mapping"
        assert len(mapping.input_dofs) == 3
        assert len(mapping.output_dofs) == 5
        assert mapping.device == "cpu"

    def test_forward_pass(self, simple_dofs, simple_model):
        """Test forward pass through mapping."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        # Create input state
        input_state = State(values={
            dof: np.random.uniform(-1.0, 1.0) for dof in input_dofs
        })

        # Forward pass
        output_state = mapping(input_state)

        # Check output
        assert isinstance(output_state, State)
        for dof in output_dofs:
            assert output_state.get_value(dof) is not None

    def test_model_in_eval_mode(self, simple_dofs, simple_model):
        """Test that model is in eval mode by default."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        assert not mapping.model.training

    def test_uncertainty_without_dropout(self, simple_dofs, simple_model):
        """Test uncertainty estimation without dropout."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu",
            use_dropout_uncertainty=False
        )

        input_state = State(values={
            dof: 0.5 for dof in input_dofs
        })

        uncertainties = mapping.compute_uncertainty(input_state)

        # Should return resolution-based uncertainty
        assert len(uncertainties) == len(output_dofs)
        for dof, unc in uncertainties.items():
            assert unc > 0

    def test_uncertainty_with_dropout(self, simple_dofs):
        """Test MC Dropout uncertainty estimation."""
        input_dofs, output_dofs = simple_dofs

        # Create model with dropout
        model = create_mlp(3, 5, [10], dropout=0.3)

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            use_dropout_uncertainty=True,
            dropout_samples=10
        )

        input_state = State(values={
            dof: 0.5 for dof in input_dofs
        })

        uncertainties = mapping.compute_uncertainty(input_state)

        # Should return MC Dropout uncertainty
        assert len(uncertainties) == len(output_dofs)
        for dof, unc in uncertainties.items():
            assert unc >= 0  # Uncertainty should be non-negative

    def test_forward_with_gradients(self, simple_dofs, simple_model):
        """Test forward pass with gradient tracking."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        input_state = State(values={
            dof: 0.5 for dof in input_dofs
        })

        output_state, output_tensor = mapping.forward_with_gradients(input_state)

        # Check state
        assert isinstance(output_state, State)

        # Check tensor has gradients
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.requires_grad or output_tensor.grad_fn is not None

    def test_device_handling(self, simple_dofs, simple_model):
        """Test that model is moved to correct device."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        # Check model is on CPU
        for param in mapping.model.parameters():
            assert param.device.type == "cpu"


class TestTorchObserver:
    """Test TorchObserver implementation."""

    def test_creation(self, simple_dofs, simple_model):
        """Test creating a TorchObserver."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="world",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        observer = TorchObserver(
            name="torch_observer",
            internal_dofs=output_dofs,
            external_dofs=input_dofs,
            world_model=mapping,
            device="cpu"
        )

        assert observer.name == "torch_observer"
        assert observer.device == "cpu"

    def test_observe(self, simple_dofs, simple_model):
        """Test observation with TorchObserver."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="world",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        observer = TorchObserver(
            name="torch_observer",
            internal_dofs=output_dofs,
            external_dofs=input_dofs,
            world_model=mapping,
            device="cpu"
        )

        external_state = State(values={
            dof: 0.5 for dof in input_dofs
        })

        internal_state = observer.observe(external_state)

        assert isinstance(internal_state, State)
        for dof in output_dofs:
            assert internal_state.get_value(dof) is not None

    def test_observe_batch(self, simple_dofs, simple_model):
        """Test batch observation."""
        input_dofs, output_dofs = simple_dofs

        mapping = TorchNeuralMapping(
            name="world",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=simple_model,
            device="cpu"
        )

        observer = TorchObserver(
            name="torch_observer",
            internal_dofs=output_dofs,
            external_dofs=input_dofs,
            world_model=mapping,
            device="cpu"
        )

        # Create batch of external states
        external_states = [
            State(values={dof: np.random.uniform(-1, 1) for dof in input_dofs})
            for _ in range(5)
        ]

        internal_states = observer.observe_batch(external_states)

        assert len(internal_states) == 5
        for state in internal_states:
            assert isinstance(state, State)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_conscious_observer_with_pytorch(self, simple_dofs):
        """Test creating a conscious observer with PyTorch models."""
        input_dofs, output_dofs = simple_dofs

        # World model
        world_model_nn = create_mlp(3, 5, [10], dropout=0.2)
        world_model = TorchNeuralMapping(
            name="world",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=world_model_nn,
            device="cpu",
            use_dropout_uncertainty=True
        )

        # Self model (same architecture!)
        self_model_nn = create_mlp(5, 5, [10], dropout=0.2)
        self_model = TorchNeuralMapping(
            name="self",
            input_dofs=output_dofs,
            output_dofs=output_dofs,
            model=self_model_nn,
            device="cpu",
            use_dropout_uncertainty=True
        )

        # Create conscious observer
        observer = TorchObserver(
            name="conscious",
            internal_dofs=output_dofs,
            external_dofs=input_dofs,
            world_model=world_model,
            self_model=self_model,
            device="cpu"
        )

        # Check consciousness
        assert observer.is_conscious()
        assert observer.recursive_depth() >= 1

        # Test observation and self-observation
        external_state = State(values={dof: 0.5 for dof in input_dofs})
        internal_state = observer.observe(external_state)
        self_repr = observer.self_observe()

        assert internal_state is not None
        assert self_repr is not None

    def test_uncertainty_quantification_pipeline(self, simple_dofs):
        """Test full uncertainty quantification pipeline."""
        input_dofs, output_dofs = simple_dofs

        # Create model with dropout
        model = create_mlp(3, 5, [10], dropout=0.3)

        mapping = TorchNeuralMapping(
            name="test",
            input_dofs=input_dofs,
            output_dofs=output_dofs,
            model=model,
            device="cpu",
            use_dropout_uncertainty=True,
            dropout_samples=20
        )

        # Multiple observations
        for _ in range(5):
            input_state = State(values={
                dof: np.random.uniform(-1, 1) for dof in input_dofs
            })

            output_state = mapping(input_state)
            uncertainties = mapping.compute_uncertainty(input_state)

            # Check that we get valid outputs and uncertainties
            assert output_state is not None
            assert len(uncertainties) == len(output_dofs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
