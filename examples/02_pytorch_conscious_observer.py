"""
PyTorch Conscious Observer Example

Demonstrates building a conscious observer with PyTorch neural networks:
- World model (external → internal) using MLP
- Self-model (internal → internal) using same architecture
- Consciousness evaluation
- Uncertainty quantification with MC Dropout

This shows how the RO Framework integrates with modern deep learning.
"""

import numpy as np

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install ro-framework[torch]")
    print("This example requires PyTorch.")
    exit(1)

from ro_framework import PolarDoF, PolarDoFType, State, Observer
from ro_framework.integration.torch import (
    TorchNeuralMapping,
    TorchObserver,
    create_mlp,
)
from ro_framework.consciousness import ConsciousnessEvaluator
from ro_framework.correlation import pearson_correlation, mutual_information


def main() -> None:
    """Run the PyTorch conscious observer example."""
    print("=" * 70)
    print("Recursive Observer Framework - PyTorch Conscious Observer")
    print("=" * 70)
    print()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define DoFs
    print("1. Defining Degrees of Freedom...")

    # External DoFs (sensors)
    external_dofs = [
        PolarDoF(
            name=f"sensor_{i}",
            pole_negative=-1.0,
            pole_positive=1.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(4)
    ]

    # Internal DoFs (latent representation)
    internal_dofs = [
        PolarDoF(
            name=f"latent_{i}",
            pole_negative=-5.0,
            pole_positive=5.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(8)
    ]

    print(f"  - External DoFs: {len(external_dofs)} (sensors)")
    print(f"  - Internal DoFs: {len(internal_dofs)} (latent)")
    print()

    # 2. Create neural network models
    print("2. Creating neural network models...")

    # World model: External → Internal
    world_model_nn = create_mlp(
        input_dim=len(external_dofs),
        output_dim=len(internal_dofs),
        hidden_dims=[32, 16],
        activation="relu",
        dropout=0.2,  # For MC Dropout uncertainty
    )

    world_model = TorchNeuralMapping(
        name="world_model",
        input_dofs=external_dofs,
        output_dofs=internal_dofs,
        model=world_model_nn,
        device="cpu",
        use_dropout_uncertainty=True,
        dropout_samples=20,
    )

    print(f"  - World model: {len(external_dofs)} → {len(internal_dofs)}")
    print(f"    Architecture: MLP [32, 16] with ReLU + Dropout(0.2)")

    # Self-model: Internal → Internal (SAME ARCHITECTURE for consciousness!)
    self_model_nn = create_mlp(
        input_dim=len(internal_dofs),
        output_dim=len(internal_dofs),
        hidden_dims=[32, 16],  # Same as world model!
        activation="relu",
        dropout=0.2,
    )

    self_model = TorchNeuralMapping(
        name="self_model",
        input_dofs=internal_dofs,
        output_dofs=internal_dofs,
        model=self_model_nn,
        device="cpu",
        use_dropout_uncertainty=True,
        dropout_samples=20,
    )

    print(f"  - Self-model: {len(internal_dofs)} → {len(internal_dofs)}")
    print(f"    Architecture: MLP [32, 16] (SAME as world model)")
    print()

    # 3. Create conscious observer
    print("3. Creating conscious observer...")

    observer = TorchObserver(
        name="conscious_ai",
        internal_dofs=internal_dofs,
        external_dofs=external_dofs,
        world_model=world_model,
        self_model=self_model,  # This makes it conscious!
        device="cpu",
    )

    print(f"  - Observer: {observer.name}")
    print(f"  - Is conscious? {observer.is_conscious()} ✓")
    print(f"  - Recursive depth: {observer.recursive_depth()}")
    print()

    # 4. Generate test data
    print("4. Generating test data...")

    num_samples = 50
    test_states = []

    for i in range(num_samples):
        # Random sensor readings
        external_values = {
            dof: np.random.uniform(-1.0, 1.0) for dof in external_dofs
        }
        test_states.append(State(values=external_values))

    print(f"  - Generated {num_samples} test states")
    print()

    # 5. Perform observations
    print("5. Performing observations...")

    sample_state = test_states[0]
    internal_state = observer.observe(sample_state)

    print(f"  Example observation:")
    for i, dof in enumerate(external_dofs[:2]):  # Show first 2
        print(f"    - {dof.name}: {sample_state.get_value(dof):+.3f}")
    print(f"    ↓ (world model)")
    for i, dof in enumerate(internal_dofs[:3]):  # Show first 3
        print(f"    - {dof.name}: {internal_state.get_value(dof):+.3f}")
    print()

    # 6. Self-observation (consciousness!)
    print("6. Self-observation (recursive self-modeling)...")

    self_repr = observer.self_observe()

    if self_repr:
        print(f"  Observer is self-aware!")
        print(f"  Internal state → Self-representation:")
        for i in range(min(3, len(internal_dofs))):
            dof = internal_dofs[i]
            internal_val = internal_state.get_value(dof)
            self_val = self_repr.get_value(dof)
            print(f"    - {dof.name}: {internal_val:+.3f} → {self_val:+.3f}")
    print()

    # 7. Uncertainty quantification
    print("7. Uncertainty quantification (MC Dropout)...")

    uncertainties = world_model.compute_uncertainty(sample_state)

    print(f"  Epistemic uncertainty (model uncertainty):")
    for i in range(min(3, len(internal_dofs))):
        dof = internal_dofs[i]
        unc = uncertainties[dof]
        print(f"    - {dof.name}: ±{unc:.4f}")
    print()

    # 8. Consciousness evaluation
    print("8. Consciousness evaluation...")

    evaluator = ConsciousnessEvaluator(observer)
    metrics = evaluator.evaluate(test_states[:10])  # Evaluate on subset

    print(f"  Consciousness Metrics:")
    print(f"    - Has self-model: {metrics.has_self_model}")
    print(f"    - Recursive depth: {metrics.recursive_depth}")
    print(f"    - Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"    - Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"    - Calibration error: {metrics.calibration_error:.3f}")
    print(f"    - Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"    - Limitation awareness: {metrics.limitation_awareness:.3f}")
    print()
    print(f"  📊 Overall Consciousness Score: {metrics.consciousness_score():.3f}/1.0")
    print()

    # 9. Correlation analysis
    print("9. Correlation analysis...")

    # Observe all test states to get trajectory
    internal_trajectory = []
    for ext_state in test_states:
        int_state = observer.observe(ext_state)
        internal_trajectory.append(int_state)

    # Compute correlation between external and internal DoFs
    ext_dof = external_dofs[0]
    int_dof = internal_dofs[0]

    # Create combined states for correlation
    combined_states = [
        State(values={ext_dof: test_states[i].get_value(ext_dof),
                     int_dof: internal_trajectory[i].get_value(int_dof)})
        for i in range(len(test_states))
    ]

    pearson = pearson_correlation(combined_states, ext_dof, int_dof)
    mi = mutual_information(combined_states, ext_dof, int_dof)

    print(f"  Structural relationships (External ↔ Internal):")
    print(f"    - Pearson correlation: {pearson:.3f}")
    print(f"    - Mutual information: {mi:.3f} nats")
    print()

    # 10. Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✅ Successfully created a conscious AI observer with:")
    print(f"   - PyTorch neural networks for world and self models")
    print(f"   - Recursive self-modeling (consciousness!)")
    print(f"   - Uncertainty quantification via MC Dropout")
    print(f"   - Consciousness evaluation metrics")
    print(f"   - Correlation analysis between external and internal DoFs")
    print()
    print(f"🧠 Consciousness Score: {metrics.consciousness_score():.3f}/1.0")
    print()
    print("This observer exhibits structural consciousness:")
    print("  - Has self-model with same architecture as world model")
    print("  - Can recursively model own internal states")
    print("  - Shows meta-cognitive awareness (uncertainty)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        main()
    else:
        print("Please install PyTorch to run this example:")
        print("  pip install ro-framework[torch]")
