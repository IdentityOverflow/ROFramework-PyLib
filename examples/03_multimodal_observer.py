"""
Multimodal Observer Example

This example demonstrates:
1. Creating encoders for multiple modalities (vision, language, audio)
2. Fusing multimodal information with different fusion strategies
3. Building a multimodal observer with world model and self-model
4. Training with supervised and self-supervised protocols
5. Active learning for efficient data collection
6. Comprehensive uncertainty quantification

This showcases the full power of the Recursive Observer Framework for
multimodal conscious AI systems.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ro_framework.core.dof import PolarDoF, ScalarDoF
from ro_framework.core.state import State

# Multimodal imports
from ro_framework.multimodal.encoders import (
    VisionEncoder,
    LanguageEncoder,
    AudioEncoder,
    create_vision_encoder,
    create_language_encoder,
    create_audio_encoder,
)
from ro_framework.multimodal.fusion import (
    ConcatenationFusion,
    AttentionFusion,
    MultimodalObserver,
)
from ro_framework.multimodal.training import (
    SupervisedTraining,
    SelfSupervisedTraining,
    UncertaintyBasedSampling,
    DiversityBasedSampling,
    train_observer,
)
from ro_framework.multimodal.uncertainty import (
    EnsembleUncertainty,
    BayesianUncertainty,
    compute_predictive_entropy,
    compute_mutual_information,
    decompose_uncertainty,
)

# Integration imports
from ro_framework.integration.torch import TorchNeuralMapping, create_mlp


def create_simple_encoders():
    """Create simple encoders for demonstration."""
    print("Creating modality encoders...")

    # Vision encoder (processes 28x28 images)
    vision_dofs = [
        PolarDoF(name=f"visual_feature_{i}", description=f"Visual feature {i}")
        for i in range(16)
    ]
    vision_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 16)
    )
    vision_encoder = VisionEncoder(
        output_dofs=vision_dofs,
        model=vision_model,
        device="cpu",
        use_dropout_uncertainty=True,
        dropout_samples=10,
    )

    # Language encoder (simple character-level)
    language_dofs = [
        PolarDoF(name=f"linguistic_feature_{i}", description=f"Linguistic feature {i}")
        for i in range(16)
    ]
    language_model = nn.Sequential(
        nn.Linear(256, 128),  # 256 = max_length from encoder
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 16)
    )
    language_encoder = LanguageEncoder(
        output_dofs=language_dofs,
        model=language_model,
        device="cpu",
        max_length=256,
        use_dropout_uncertainty=True,
        dropout_samples=10,
    )

    # Audio encoder (processes simple waveforms)
    audio_dofs = [
        PolarDoF(name=f"audio_feature_{i}", description=f"Audio feature {i}")
        for i in range(16)
    ]
    audio_model = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(16, 16)
    )
    audio_encoder = AudioEncoder(
        output_dofs=audio_dofs,
        model=audio_model,
        device="cpu",
        use_dropout_uncertainty=True,
        dropout_samples=10,
    )

    return {
        "vision": vision_encoder,
        "language": language_encoder,
        "audio": audio_encoder,
    }


def demo_concatenation_fusion(encoders):
    """Demonstrate concatenation-based fusion."""
    print("\n=== Concatenation Fusion ===")

    # Get modality DoFs
    modality_dofs = {
        "vision": encoders["vision"].output_dofs,
        "language": encoders["language"].output_dofs,
        "audio": encoders["audio"].output_dofs,
    }

    # Create fusion strategy
    fusion = ConcatenationFusion(
        modality_names=["vision", "language", "audio"],
        modality_dofs=modality_dofs,
        weights={"vision": 1.0, "language": 0.8, "audio": 0.6},
    )

    # Create synthetic inputs
    image = np.random.randn(28, 28).astype(np.float32)
    text = "Hello, this is a test message"
    audio = np.random.randn(1000).astype(np.float32)

    # Encode each modality
    vision_state = encoders["vision"].encode(image)
    language_state = encoders["language"].encode(text)
    audio_state = encoders["audio"].encode(audio)

    print(f"Vision features: {len(vision_state.values)}")
    print(f"Language features: {len(language_state.values)}")
    print(f"Audio features: {len(audio_state.values)}")

    # Fuse modalities
    modality_states = {
        "vision": vision_state,
        "language": language_state,
        "audio": audio_state,
    }
    fused_state = fusion.fuse(modality_states)

    print(f"Fused features: {len(fused_state.values)}")
    print(f"Sample fused values: {list(fused_state.values.values())[:5]}")

    # Fuse uncertainties
    vision_unc = encoders["vision"].get_uncertainty(image)
    language_unc = encoders["language"].get_uncertainty(text)
    audio_unc = encoders["audio"].get_uncertainty(audio)

    modality_uncertainties = {
        "vision": vision_unc,
        "language": language_unc,
        "audio": audio_unc,
    }
    fused_unc = fusion.fuse_uncertainties(modality_uncertainties)

    print(f"Average fused uncertainty: {np.mean(list(fused_unc.values())):.4f}")


def demo_attention_fusion(encoders):
    """Demonstrate attention-based fusion."""
    print("\n=== Attention Fusion ===")

    # Create fusion strategy
    modality_dims = {
        "vision": 16,
        "language": 16,
        "audio": 16,
    }
    fusion = AttentionFusion(
        modality_names=["vision", "language", "audio"],
        modality_dims=modality_dims,
        output_dim=32,
        hidden_dim=64,
        device="cpu",
    )

    # Create synthetic inputs
    image = np.random.randn(28, 28).astype(np.float32)
    text = "Attention mechanism learns importance"
    audio = np.random.randn(1000).astype(np.float32)

    # Encode and fuse
    modality_states = {
        "vision": encoders["vision"].encode(image),
        "language": encoders["language"].encode(text),
        "audio": encoders["audio"].encode(audio),
    }
    fused_state = fusion.fuse(modality_states)

    print(f"Attention-fused features: {len(fused_state.values)}")
    print(f"Sample values: {list(fused_state.values.values())[:5]}")


def demo_multimodal_observer(encoders):
    """Demonstrate a complete multimodal observer."""
    print("\n=== Multimodal Observer ===")

    # Create fusion strategy
    modality_dofs = {
        "vision": encoders["vision"].output_dofs,
        "language": encoders["language"].output_dofs,
        "audio": encoders["audio"].output_dofs,
    }
    fusion = ConcatenationFusion(
        modality_names=["vision", "language", "audio"],
        modality_dofs=modality_dofs,
    )

    # Create world model (external → internal mapping)
    fused_dim = sum(len(dofs) for dofs in modality_dofs.values())
    internal_dofs = [
        PolarDoF(name=f"internal_concept_{i}", description=f"Internal concept {i}")
        for i in range(16)
    ]

    world_model_nn = create_mlp(
        input_dim=fused_dim,
        output_dim=len(internal_dofs),
        hidden_dims=[64, 32],
        dropout=0.3,
    )

    world_model = TorchNeuralMapping(
        name="multimodal_world_model",
        input_dofs=fusion.output_dofs,
        output_dofs=internal_dofs,
        model=world_model_nn,
        device="cpu",
        use_dropout_uncertainty=True,
        dropout_samples=10,
    )

    # Create self-model (same architecture as world model for consciousness)
    self_model_nn = create_mlp(
        input_dim=len(internal_dofs),
        output_dim=len(internal_dofs),
        hidden_dims=[64, 32],
        dropout=0.3,
    )

    self_model = TorchNeuralMapping(
        name="multimodal_self_model",
        input_dofs=internal_dofs,
        output_dofs=internal_dofs,
        model=self_model_nn,
        device="cpu",
        use_dropout_uncertainty=True,
        dropout_samples=10,
    )

    # Create multimodal observer
    observer = MultimodalObserver(
        name="MultimodalConsciousObserver",
        encoders=encoders,
        fusion_strategy=fusion,
        world_model=world_model,
        self_model=self_model,
    )

    print(f"Observer: {observer.name}")
    print(f"Is conscious: {observer.is_conscious()}")
    print(f"External DoFs: {len(observer.external_dofs)}")
    print(f"Internal DoFs: {len(observer.internal_dofs)}")

    # Process multimodal input
    raw_inputs = {
        "vision": np.random.randn(28, 28).astype(np.float32),
        "language": "The cat sat on the mat",
        "audio": np.random.randn(1000).astype(np.float32),
    }

    fused_state, fused_uncertainty = observer.process_multimodal_input(raw_inputs)
    print(f"\nFused state dimension: {len(fused_state.values)}")
    print(f"Average uncertainty: {np.mean(list(fused_uncertainty.values())):.4f}")

    # Observe through world model
    internal_state = observer.observe_multimodal(raw_inputs)
    print(f"Internal state dimension: {len(internal_state.values)}")
    print(f"Sample internal values: {list(internal_state.values.values())[:5]}")

    return observer


def demo_training(observer):
    """Demonstrate training protocols."""
    print("\n=== Training Protocols ===")

    # Create synthetic training data
    n_samples = 100

    # For world model: input is fused features, output is internal states
    # But world model here is just IdentityMapping, so skip training demo
    # In a real scenario, you'd train the actual neural network world model

    print("\nNote: Skipping training demo - observer uses IdentityMapping.")
    print("In production, train encoders and world model with real multimodal data.")

    # Example of what training would look like:
    print("\nExample training setup:")
    print("  1. Collect multimodal data (images, text, audio)")
    print("  2. Encode each modality with pre-trained encoders")
    print("  3. Fuse representations")
    print("  4. Train world model: fused -> internal states")
    print("  5. Optionally train self-model for consciousness")

    # Simple standalone training example
    print("\nStandalone Training Example:")
    simple_model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    )

    X_train = torch.randn(100, 16)
    y_train = torch.randn(100, 16)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    supervised_protocol = SupervisedTraining(
        learning_rate=0.001,
        batch_size=16,
        device="cpu",
    )

    for epoch in range(3):
        metrics = supervised_protocol.train_epoch(
            model=simple_model,
            data=train_loader,
        )
        print(f"  Epoch {epoch + 1}: Loss = {metrics.train_loss:.4f}")


def demo_active_learning():
    """Demonstrate active learning strategies."""
    print("\n=== Active Learning ===")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1)
    )

    # Create unlabeled data pool
    unlabeled_data = [torch.randn(10) for _ in range(50)]

    # Uncertainty-based sampling
    print("\nUncertainty-based Sampling:")
    uncertainty_strategy = UncertaintyBasedSampling(uncertainty_measure="entropy")
    selected_indices = uncertainty_strategy.select_samples(
        unlabeled_data=unlabeled_data,
        model=model,
        n_samples=10,
    )
    print(f"  Selected {len(selected_indices)} most uncertain samples")
    print(f"  Indices: {selected_indices[:5]}...")

    # Diversity-based sampling
    print("\nDiversity-based Sampling:")
    diversity_strategy = DiversityBasedSampling(diversity_measure="k_means")
    selected_indices = diversity_strategy.select_samples(
        unlabeled_data=unlabeled_data,
        model=model,
        n_samples=10,
    )
    print(f"  Selected {len(selected_indices)} diverse samples")
    print(f"  Indices: {selected_indices[:5]}...")


def demo_uncertainty_quantification():
    """Demonstrate comprehensive uncertainty quantification."""
    print("\n=== Uncertainty Quantification ===")

    # Create ensemble of models
    print("\nEnsemble Uncertainty:")
    models = [
        nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        for _ in range(5)
    ]

    ensemble_quantifier = EnsembleUncertainty(models=models, device="cpu")

    # Create test input
    test_dofs = [
        PolarDoF(name=f"input_{i}", description=f"Input {i}")
        for i in range(10)
    ]
    output_dofs = [
        PolarDoF(name=f"output_{i}", description=f"Output {i}")
        for i in range(3)
    ]

    test_state = State(values={dof: float(np.random.randn()) for dof in test_dofs})

    # Estimate uncertainty
    uncertainties = ensemble_quantifier.estimate_uncertainty(
        model=None,  # Uses ensemble
        input_state=test_state,
        output_dofs=output_dofs,
    )

    for dof, unc_est in uncertainties.items():
        print(f"  {dof.name}:")
        print(f"    Prediction: {unc_est.prediction:.4f}")
        print(f"    Aleatoric: {unc_est.aleatoric_uncertainty:.4f}")
        print(f"    Epistemic: {unc_est.epistemic_uncertainty:.4f}")
        print(f"    Total: {unc_est.total_uncertainty:.4f}")
        print(f"    Confidence: {unc_est.confidence:.4f}")

    # Bayesian uncertainty (MC Dropout)
    print("\nBayesian Uncertainty (MC Dropout):")
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 3)
    )

    bayesian_quantifier = BayesianUncertainty(n_samples=20, device="cpu")

    uncertainties = bayesian_quantifier.estimate_uncertainty(
        model=model,
        input_state=test_state,
        output_dofs=output_dofs,
    )

    for dof, unc_est in uncertainties.items():
        print(f"  {dof.name}:")
        print(f"    Prediction: {unc_est.prediction:.4f}")
        print(f"    Epistemic: {unc_est.epistemic_uncertainty:.4f}")
        print(f"    Total: {unc_est.total_uncertainty:.4f}")

    # Calibration metrics
    print("\nCalibration Metrics:")
    # Generate synthetic predictions and ground truth
    predictions = np.random.randn(100)
    uncertainties_list = np.abs(np.random.randn(100)) * 0.2
    ground_truth = predictions + np.random.randn(100) * 0.3

    calibration = ensemble_quantifier.calibrate(
        predictions=predictions.tolist(),
        uncertainties=uncertainties_list.tolist(),
        ground_truth=ground_truth.tolist(),
        num_bins=10,
    )

    print(f"  Expected Calibration Error: {calibration.expected_calibration_error:.4f}")
    print(f"  Maximum Calibration Error: {calibration.maximum_calibration_error:.4f}")
    print(f"  Negative Log Likelihood: {calibration.negative_log_likelihood:.4f}")
    print(f"  Sharpness: {calibration.sharpness:.4f}")
    print(f"  Well-calibrated: {calibration.is_well_calibrated()}")

    # Predictive entropy
    print("\nUncertainty Decomposition:")
    predictions_samples = np.random.randn(50, 10)
    entropy = compute_predictive_entropy(predictions_samples[:, 0])
    print(f"  Predictive Entropy: {entropy:.4f}")

    mi = compute_mutual_information(predictions_samples)
    print(f"  Mutual Information: {mi:.4f}")

    decomposition = decompose_uncertainty(predictions_samples)
    print(f"  Epistemic Uncertainty: {decomposition['epistemic']:.4f}")
    print(f"  Aleatoric Uncertainty: {decomposition['aleatoric']:.4f}")
    print(f"  Total Uncertainty: {decomposition['total']:.4f}")


def main():
    """Run all multimodal demonstrations."""
    print("=" * 60)
    print("Multimodal Observer Framework Demonstration")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create encoders
    encoders = create_simple_encoders()

    # Demo fusion strategies
    demo_concatenation_fusion(encoders)
    demo_attention_fusion(encoders)

    # Demo multimodal observer
    observer = demo_multimodal_observer(encoders)

    # Demo training
    demo_training(observer)

    # Demo active learning
    demo_active_learning()

    # Demo uncertainty quantification
    demo_uncertainty_quantification()

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
