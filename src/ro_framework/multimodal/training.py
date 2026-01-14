"""
Training protocols and active learning strategies for observers.

Provides supervised and self-supervised training methods, along with
active learning strategies for efficient data collection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ro_framework.core.state import State
from ro_framework.core.dof import DoF


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.001
    additional_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class TrainingProtocol(ABC):
    """
    Abstract base class for training protocols.

    A training protocol defines how an observer learns from data.
    """

    def __init__(self, name: str):
        """
        Initialize training protocol.

        Args:
            name: Name of the protocol
        """
        self.name = name
        self.metrics_history: List[TrainingMetrics] = []

    @abstractmethod
    def train_epoch(
        self,
        model: Any,
        data: Any,
        **kwargs
    ) -> TrainingMetrics:
        """
        Train for one epoch.

        Args:
            model: Model to train
            data: Training data
            **kwargs: Additional arguments

        Returns:
            Training metrics for the epoch
        """
        pass

    def get_metrics_history(self) -> List[TrainingMetrics]:
        """Get history of training metrics."""
        return self.metrics_history


if TORCH_AVAILABLE:

    class SupervisedTraining(TrainingProtocol):
        """
        Supervised training protocol.

        Standard supervised learning with labeled data.
        """

        def __init__(
            self,
            learning_rate: float = 0.001,
            batch_size: int = 32,
            optimizer_class: type = optim.Adam,
            loss_function: Optional[nn.Module] = None,
            device: str = "cpu",
        ):
            """
            Initialize supervised training.

            Args:
                learning_rate: Learning rate for optimizer
                batch_size: Batch size for training
                optimizer_class: Optimizer class (e.g., Adam, SGD)
                loss_function: Loss function (defaults to MSE for regression)
                device: Device for computation
            """
            super().__init__("supervised")
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.optimizer_class = optimizer_class
            self.loss_function = loss_function or nn.MSELoss()
            self.device = device
            self.optimizer = None

        def train_epoch(
            self,
            model: nn.Module,
            data: DataLoader,
            val_data: Optional[DataLoader] = None,
            **kwargs
        ) -> TrainingMetrics:
            """
            Train for one epoch with supervised learning.

            Args:
                model: PyTorch model to train
                data: Training DataLoader
                val_data: Optional validation DataLoader
                **kwargs: Additional arguments

            Returns:
                Training metrics
            """
            model = model.to(self.device)
            model.train()

            # Initialize optimizer if needed
            if self.optimizer is None:
                self.optimizer = self.optimizer_class(
                    model.parameters(),
                    lr=self.learning_rate
                )

            total_loss = 0.0
            num_batches = 0

            for batch in data:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs  # Auto-encoder style

                # Forward pass
                self.optimizer.zero_grad()
                outputs = model(inputs)

                # Compute loss
                loss = self.loss_function(outputs, targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Validation
            val_loss = None
            if val_data is not None:
                val_loss = self.evaluate(model, val_data)

            metrics = TrainingMetrics(
                epoch=len(self.metrics_history),
                train_loss=avg_train_loss,
                val_loss=val_loss,
                learning_rate=self.learning_rate,
            )

            self.metrics_history.append(metrics)
            return metrics

        def evaluate(self, model: nn.Module, data: DataLoader) -> float:
            """
            Evaluate model on validation data.

            Args:
                model: Model to evaluate
                data: Validation DataLoader

            Returns:
                Average validation loss
            """
            model.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in data:
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    else:
                        inputs = batch.to(self.device)
                        targets = inputs

                    outputs = model(inputs)
                    loss = self.loss_function(outputs, targets)

                    total_loss += loss.item()
                    num_batches += 1

            model.train()
            return total_loss / num_batches if num_batches > 0 else 0.0


    class SelfSupervisedTraining(TrainingProtocol):
        """
        Self-supervised training protocol.

        Learning without explicit labels using pretext tasks.
        """

        def __init__(
            self,
            pretext_task: str = "reconstruction",
            learning_rate: float = 0.001,
            batch_size: int = 32,
            optimizer_class: type = optim.Adam,
            device: str = "cpu",
        ):
            """
            Initialize self-supervised training.

            Args:
                pretext_task: Type of pretext task (reconstruction, contrastive, etc.)
                learning_rate: Learning rate
                batch_size: Batch size
                optimizer_class: Optimizer class
                device: Device for computation
            """
            super().__init__("self_supervised")
            self.pretext_task = pretext_task
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.optimizer_class = optimizer_class
            self.device = device
            self.optimizer = None

        def train_epoch(
            self,
            model: nn.Module,
            data: DataLoader,
            **kwargs
        ) -> TrainingMetrics:
            """
            Train for one epoch with self-supervised learning.

            Args:
                model: PyTorch model to train
                data: Training DataLoader (unlabeled)
                **kwargs: Additional arguments

            Returns:
                Training metrics
            """
            model = model.to(self.device)
            model.train()

            if self.optimizer is None:
                self.optimizer = self.optimizer_class(
                    model.parameters(),
                    lr=self.learning_rate
                )

            total_loss = 0.0
            num_batches = 0

            for batch in data:
                inputs = batch.to(self.device) if torch.is_tensor(batch) else batch[0].to(self.device)

                # Generate pretext task
                if self.pretext_task == "reconstruction":
                    # Reconstruction task (e.g., denoising autoencoder)
                    corrupted = self._corrupt_input(inputs)
                    targets = inputs

                    self.optimizer.zero_grad()
                    outputs = model(corrupted)
                    loss = nn.functional.mse_loss(outputs, targets)

                elif self.pretext_task == "contrastive":
                    # Contrastive learning (simplified)
                    augmented1 = self._augment_input(inputs)
                    augmented2 = self._augment_input(inputs)

                    self.optimizer.zero_grad()
                    features1 = model(augmented1)
                    features2 = model(augmented2)

                    # Contrastive loss (simplified InfoNCE)
                    loss = self._contrastive_loss(features1, features2)

                else:
                    raise ValueError(f"Unknown pretext task: {self.pretext_task}")

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            metrics = TrainingMetrics(
                epoch=len(self.metrics_history),
                train_loss=avg_loss,
                learning_rate=self.learning_rate,
            )

            self.metrics_history.append(metrics)
            return metrics

        def _corrupt_input(self, inputs: torch.Tensor) -> torch.Tensor:
            """Add noise to inputs for denoising task."""
            noise = torch.randn_like(inputs) * 0.1
            return inputs + noise

        def _augment_input(self, inputs: torch.Tensor) -> torch.Tensor:
            """Apply random augmentations."""
            # Simple augmentation: random noise
            augmented = inputs + torch.randn_like(inputs) * 0.05
            return augmented

        def _contrastive_loss(
            self,
            features1: torch.Tensor,
            features2: torch.Tensor,
            temperature: float = 0.5
        ) -> torch.Tensor:
            """Simplified contrastive loss."""
            # Normalize features
            features1 = nn.functional.normalize(features1, dim=1)
            features2 = nn.functional.normalize(features2, dim=1)

            # Similarity
            similarity = torch.matmul(features1, features2.T) / temperature

            # Labels: positive pairs are on diagonal
            batch_size = features1.shape[0]
            labels = torch.arange(batch_size, device=features1.device)

            # Cross-entropy loss
            loss = nn.functional.cross_entropy(similarity, labels)
            return loss


class ActiveLearningStrategy(ABC):
    """
    Abstract base class for active learning strategies.

    Active learning selects the most informative samples for labeling.
    """

    def __init__(self, name: str):
        """
        Initialize active learning strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name

    @abstractmethod
    def select_samples(
        self,
        unlabeled_data: List[Any],
        model: Any,
        n_samples: int,
        **kwargs
    ) -> List[int]:
        """
        Select most informative samples from unlabeled data.

        Args:
            unlabeled_data: Pool of unlabeled samples
            model: Current model
            n_samples: Number of samples to select
            **kwargs: Additional arguments

        Returns:
            Indices of selected samples
        """
        pass


class UncertaintyBasedSampling(ActiveLearningStrategy):
    """
    Uncertainty-based active learning.

    Selects samples where the model is most uncertain.
    """

    def __init__(self, uncertainty_measure: str = "entropy"):
        """
        Initialize uncertainty-based sampling.

        Args:
            uncertainty_measure: Type of uncertainty measure (entropy, variance, etc.)
        """
        super().__init__("uncertainty_based")
        self.uncertainty_measure = uncertainty_measure

    def select_samples(
        self,
        unlabeled_data: List[Any],
        model: Any,
        n_samples: int,
        uncertainty_estimator: Optional[Callable] = None,
        **kwargs
    ) -> List[int]:
        """
        Select samples with highest uncertainty.

        Args:
            unlabeled_data: Pool of unlabeled samples
            model: Current model
            n_samples: Number of samples to select
            uncertainty_estimator: Function to estimate uncertainty
            **kwargs: Additional arguments

        Returns:
            Indices of most uncertain samples
        """
        if uncertainty_estimator is None:
            # Default: use model prediction variance
            uncertainties = self._default_uncertainty(unlabeled_data, model)
        else:
            uncertainties = [
                uncertainty_estimator(sample, model)
                for sample in unlabeled_data
            ]

        # Select samples with highest uncertainty
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        return uncertain_indices.tolist()

    def _default_uncertainty(
        self,
        data: List[Any],
        model: Any
    ) -> np.ndarray:
        """Default uncertainty estimation using prediction variance."""
        if not TORCH_AVAILABLE:
            return np.random.rand(len(data))

        if isinstance(model, nn.Module):
            model.eval()
            uncertainties = []

            with torch.no_grad():
                for sample in data:
                    if torch.is_tensor(sample):
                        input_tensor = sample.unsqueeze(0)
                    else:
                        input_tensor = torch.tensor([sample]).float()

                    output = model(input_tensor)

                    # Use output variance as uncertainty
                    if output.dim() > 1:
                        uncertainty = torch.var(output).item()
                    else:
                        uncertainty = abs(output.item())

                    uncertainties.append(uncertainty)

            return np.array(uncertainties)

        # Fallback: random
        return np.random.rand(len(data))


class DiversityBasedSampling(ActiveLearningStrategy):
    """
    Diversity-based active learning.

    Selects diverse samples to cover the input space.
    """

    def __init__(self, diversity_measure: str = "k_means"):
        """
        Initialize diversity-based sampling.

        Args:
            diversity_measure: Type of diversity measure (k_means, coreset, etc.)
        """
        super().__init__("diversity_based")
        self.diversity_measure = diversity_measure

    def select_samples(
        self,
        unlabeled_data: List[Any],
        model: Any,
        n_samples: int,
        **kwargs
    ) -> List[int]:
        """
        Select diverse samples.

        Args:
            unlabeled_data: Pool of unlabeled samples
            model: Current model (may extract features)
            n_samples: Number of samples to select
            **kwargs: Additional arguments

        Returns:
            Indices of diverse samples
        """
        # Extract features if model available
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            features = self._extract_features(unlabeled_data, model)
        else:
            # Use raw data as features
            features = np.array([
                np.array(sample).flatten()
                for sample in unlabeled_data
            ])

        # Select diverse samples
        if self.diversity_measure == "k_means":
            selected_indices = self._k_means_selection(features, n_samples)
        else:
            # Fallback: random selection
            selected_indices = np.random.choice(
                len(unlabeled_data),
                size=n_samples,
                replace=False
            )

        return selected_indices.tolist()

    def _extract_features(
        self,
        data: List[Any],
        model: nn.Module
    ) -> np.ndarray:
        """Extract features from model."""
        model.eval()
        features_list = []

        with torch.no_grad():
            for sample in data:
                if torch.is_tensor(sample):
                    input_tensor = sample.unsqueeze(0)
                else:
                    input_tensor = torch.tensor([sample]).float()

                # Forward pass
                features = model(input_tensor)
                features_list.append(features.cpu().numpy().flatten())

        return np.array(features_list)

    def _k_means_selection(
        self,
        features: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Select samples using k-means clustering."""
        from sklearn.cluster import KMeans

        # Cluster features
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(features)

        # Select sample closest to each cluster center
        selected_indices = []
        for i in range(n_samples):
            cluster_mask = kmeans.labels_ == i
            cluster_features = features[cluster_mask]

            if len(cluster_features) > 0:
                # Find closest to center
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = np.argmin(distances)

                # Get original index
                original_indices = np.where(cluster_mask)[0]
                selected_indices.append(original_indices[closest_idx])

        return np.array(selected_indices)


def train_observer(
    observer: Any,
    training_protocol: TrainingProtocol,
    train_data: Any,
    n_epochs: int,
    val_data: Optional[Any] = None,
    early_stopping_patience: int = 10,
    verbose: bool = True,
) -> List[TrainingMetrics]:
    """
    Train an observer using a training protocol.

    Args:
        observer: Observer to train
        training_protocol: Training protocol to use
        train_data: Training data
        n_epochs: Number of training epochs
        val_data: Optional validation data
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress

    Returns:
        List of training metrics for each epoch
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        # Train one epoch
        metrics = training_protocol.train_epoch(
            model=observer.world_model.model if hasattr(observer.world_model, 'model') else observer,
            data=train_data,
            val_data=val_data,
        )

        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs} - "
                  f"Train Loss: {metrics.train_loss:.4f}", end="")
            if metrics.val_loss is not None:
                print(f" - Val Loss: {metrics.val_loss:.4f}", end="")
            print()

        # Early stopping
        if metrics.val_loss is not None:
            if metrics.val_loss < best_val_loss:
                best_val_loss = metrics.val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    return training_protocol.get_metrics_history()
