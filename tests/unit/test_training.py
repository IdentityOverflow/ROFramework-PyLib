"""
Unit tests for training protocols and active learning.

Tests supervised training, self-supervised training, and active learning strategies.
"""

import pytest
import numpy as np

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    from ro_framework.multimodal.training import (
        TrainingMetrics,
        SupervisedTraining,
        SelfSupervisedTraining,
        UncertaintyBasedSampling,
        DiversityBasedSampling,
        train_observer,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestTrainingMetrics:
    """Tests for TrainingMetrics."""

    def test_training_metrics_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            epoch=0,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001
        )

        assert metrics.epoch == 0
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.learning_rate == 0.001
        assert metrics.additional_metrics == {}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSupervisedTraining:
    """Tests for SupervisedTraining."""

    def test_supervised_training_creation(self):
        """Test creating supervised training protocol."""
        protocol = SupervisedTraining(
            learning_rate=0.001,
            batch_size=32,
            device="cpu"
        )

        assert protocol.name == "supervised"
        assert protocol.learning_rate == 0.001
        assert protocol.batch_size == 32

    def test_supervised_training_epoch(self):
        """Test training for one epoch."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5)
        )

        # Create training data
        X = torch.randn(50, 10)
        y = torch.randn(50, 5)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16)

        # Create protocol
        protocol = SupervisedTraining(
            learning_rate=0.01,
            batch_size=16,
            device="cpu"
        )

        # Train one epoch
        metrics = protocol.train_epoch(model, dataloader)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.train_loss >= 0.0
        assert len(protocol.metrics_history) == 1

    def test_supervised_training_with_validation(self):
        """Test training with validation data."""
        model = nn.Sequential(nn.Linear(10, 5))

        X_train = torch.randn(50, 10)
        y_train = torch.randn(50, 5)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)

        X_val = torch.randn(20, 10)
        y_val = torch.randn(20, 5)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

        protocol = SupervisedTraining(learning_rate=0.01, device="cpu")

        metrics = protocol.train_epoch(model, train_loader, val_data=val_loader)

        assert metrics.val_loss is not None
        assert metrics.val_loss >= 0.0

    def test_supervised_training_evaluate(self):
        """Test evaluation method."""
        model = nn.Sequential(nn.Linear(10, 5))

        X_val = torch.randn(30, 10)
        y_val = torch.randn(30, 5)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

        protocol = SupervisedTraining(device="cpu")

        val_loss = protocol.evaluate(model, val_loader)

        assert isinstance(val_loss, float)
        assert val_loss >= 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSelfSupervisedTraining:
    """Tests for SelfSupervisedTraining."""

    def test_self_supervised_training_creation(self):
        """Test creating self-supervised training protocol."""
        protocol = SelfSupervisedTraining(
            pretext_task="reconstruction",
            learning_rate=0.001,
            device="cpu"
        )

        assert protocol.name == "self_supervised"
        assert protocol.pretext_task == "reconstruction"

    def test_self_supervised_reconstruction(self):
        """Test reconstruction pretext task."""
        model = nn.Sequential(
            nn.Linear(10, 10)
        )

        X = torch.randn(50, 10)
        dataloader = DataLoader(X, batch_size=16)

        protocol = SelfSupervisedTraining(
            pretext_task="reconstruction",
            learning_rate=0.01,
            device="cpu"
        )

        metrics = protocol.train_epoch(model, dataloader)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.train_loss >= 0.0

    def test_self_supervised_contrastive(self):
        """Test contrastive pretext task."""
        model = nn.Sequential(
            nn.Linear(10, 8)
        )

        X = torch.randn(50, 10)
        dataloader = DataLoader(X, batch_size=16)

        protocol = SelfSupervisedTraining(
            pretext_task="contrastive",
            learning_rate=0.01,
            device="cpu"
        )

        metrics = protocol.train_epoch(model, dataloader)

        assert metrics.train_loss >= 0.0

    def test_corrupt_input(self):
        """Test input corruption."""
        protocol = SelfSupervisedTraining(pretext_task="reconstruction", device="cpu")

        inputs = torch.randn(10, 5)
        corrupted = protocol._corrupt_input(inputs)

        assert corrupted.shape == inputs.shape
        # Should be different due to noise
        assert not torch.allclose(corrupted, inputs)

    def test_augment_input(self):
        """Test input augmentation."""
        protocol = SelfSupervisedTraining(pretext_task="contrastive", device="cpu")

        inputs = torch.randn(10, 5)
        augmented = protocol._augment_input(inputs)

        assert augmented.shape == inputs.shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestUncertaintyBasedSampling:
    """Tests for UncertaintyBasedSampling."""

    def test_uncertainty_sampling_creation(self):
        """Test creating uncertainty-based sampling strategy."""
        strategy = UncertaintyBasedSampling(uncertainty_measure="entropy")

        assert strategy.name == "uncertainty_based"
        assert strategy.uncertainty_measure == "entropy"

    def test_uncertainty_sampling_select(self):
        """Test selecting uncertain samples."""
        model = nn.Sequential(
            nn.Linear(10, 1)
        )

        unlabeled_data = [torch.randn(10) for _ in range(20)]

        strategy = UncertaintyBasedSampling()

        selected = strategy.select_samples(
            unlabeled_data=unlabeled_data,
            model=model,
            n_samples=5
        )

        assert len(selected) == 5
        assert all(isinstance(idx, int) for idx in selected)
        assert all(0 <= idx < 20 for idx in selected)

    def test_uncertainty_sampling_custom_estimator(self):
        """Test with custom uncertainty estimator."""
        model = nn.Sequential(nn.Linear(10, 1))

        unlabeled_data = [torch.randn(10) for _ in range(10)]

        # Custom estimator returns random uncertainties
        def custom_estimator(sample, model):
            return np.random.rand()

        strategy = UncertaintyBasedSampling()

        selected = strategy.select_samples(
            unlabeled_data=unlabeled_data,
            model=model,
            n_samples=3,
            uncertainty_estimator=custom_estimator
        )

        assert len(selected) == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDiversityBasedSampling:
    """Tests for DiversityBasedSampling."""

    def test_diversity_sampling_creation(self):
        """Test creating diversity-based sampling strategy."""
        strategy = DiversityBasedSampling(diversity_measure="k_means")

        assert strategy.name == "diversity_based"
        assert strategy.diversity_measure == "k_means"

    def test_diversity_sampling_select(self):
        """Test selecting diverse samples."""
        model = nn.Sequential(nn.Linear(10, 5))

        unlabeled_data = [torch.randn(10) for _ in range(20)]

        strategy = DiversityBasedSampling(diversity_measure="k_means")

        selected = strategy.select_samples(
            unlabeled_data=unlabeled_data,
            model=model,
            n_samples=5
        )

        assert len(selected) == 5
        assert all(isinstance(idx, int) for idx in selected)

    def test_k_means_selection(self):
        """Test k-means selection method."""
        strategy = DiversityBasedSampling()

        # Create features with clear clusters
        features = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2],  # Cluster 1
            [5, 5], [5.1, 5.1], [5.2, 5.2],  # Cluster 2
        ])

        selected = strategy._k_means_selection(features, n_samples=2)

        assert len(selected) == 2
        # Should select from different clusters
        assert selected[0] != selected[1]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainObserver:
    """Tests for train_observer function."""

    def test_train_observer_basic(self):
        """Test basic training loop."""
        # Create mock observer with model
        class MockObserver:
            def __init__(self):
                self.world_model = type('obj', (object,), {
                    'model': nn.Sequential(nn.Linear(10, 5))
                })

        observer = MockObserver()

        # Create training data
        X = torch.randn(30, 10)
        y = torch.randn(30, 5)
        train_loader = DataLoader(TensorDataset(X, y), batch_size=10)

        protocol = SupervisedTraining(learning_rate=0.01, device="cpu")

        # Train
        history = train_observer(
            observer=observer,
            training_protocol=protocol,
            train_data=train_loader,
            n_epochs=3,
            verbose=False
        )

        assert len(history) == 3
        assert all(isinstance(m, TrainingMetrics) for m in history)

    def test_train_observer_early_stopping(self):
        """Test early stopping."""
        class MockObserver:
            def __init__(self):
                self.world_model = type('obj', (object,), {
                    'model': nn.Sequential(nn.Linear(10, 5))
                })

        observer = MockObserver()

        X_train = torch.randn(30, 10)
        y_train = torch.randn(30, 5)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10)

        X_val = torch.randn(10, 10)
        y_val = torch.randn(10, 5)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

        protocol = SupervisedTraining(learning_rate=0.01, device="cpu")

        # Train with early stopping
        history = train_observer(
            observer=observer,
            training_protocol=protocol,
            train_data=train_loader,
            n_epochs=20,
            val_data=val_loader,
            early_stopping_patience=2,
            verbose=False
        )

        # Should stop early (less than 20 epochs)
        assert len(history) <= 20
