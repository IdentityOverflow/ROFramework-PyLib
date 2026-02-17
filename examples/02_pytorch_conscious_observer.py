"""
PyTorch Conscious Observer Example

Demonstrates building a conscious observer with PyTorch neural networks:
- World model (external -> internal) using MLP
- Self-model (internal -> internal) using same architecture
- Consciousness evaluation
- Uncertainty quantification with MC Dropout
- Knowledge assessment from observation history
- Gradient-based saliency analysis

This shows how the RO Framework integrates with modern deep learning.
"""

import numpy as np

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install ro-framework[torch]")
    exit(1)

from ro_framework import PolarDoF, PolarDoFType, State
from ro_framework.integration.torch import (
    TorchNeuralMapping,
    TorchObserver,
    create_mlp,
)
from ro_framework.consciousness.evaluation import ConsciousnessEvaluator
from ro_framework.correlation.measures import pearson_correlation, mutual_information


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

    external_dofs = [
        PolarDoF(
            name=f"sensor_{i}",
            pole_negative=-1.0,
            pole_positive=1.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(4)
    ]

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

    world_model_nn = create_mlp(
        input_dim=len(external_dofs),
        output_dim=len(internal_dofs),
        hidden_dims=[32, 16],
        activation="relu",
        dropout=0.2,
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

    print(f"  - World model: {len(external_dofs)} -> {len(internal_dofs)}")
    print(f"    Architecture: MLP [32, 16] with ReLU + Dropout(0.2)")

    # Self-model: Internal -> Internal (SAME ARCHITECTURE for consciousness!)
    self_model_nn = create_mlp(
        input_dim=len(internal_dofs),
        output_dim=len(internal_dofs),
        hidden_dims=[32, 16],
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

    print(f"  - Self-model: {len(internal_dofs)} -> {len(internal_dofs)}")
    print(f"    Architecture: MLP [32, 16] (SAME as world model)")
    print()

    # 3. Create conscious observer
    print("3. Creating conscious observer...")

    observer = TorchObserver(
        name="conscious_ai",
        internal_dofs=internal_dofs,
        external_dofs=external_dofs,
        world_model=world_model,
        self_model=self_model,
        device="cpu",
    )

    print(f"  - {observer}")
    print(f"  - Recursive depth: {observer.recursive_depth()}")
    print()

    # 4. Generate test data and observe
    print("4. Performing observations...")

    num_samples = 50
    test_states = []

    for _ in range(num_samples):
        external_values = {
            dof: np.random.uniform(-1.0, 1.0) for dof in external_dofs
        }
        test_states.append(State(values=external_values))

    # Observe all test states
    for ext_state in test_states:
        observer.observe(ext_state)

    sample_state = test_states[0]
    internal_state = observer.observe(sample_state)

    print(f"  {num_samples} observations recorded")
    print(f"  Example observation:")
    for dof in external_dofs[:2]:
        print(f"    - {dof.name}: {sample_state.get_value(dof):+.3f}")
    print("    -> (world model)")
    for dof in internal_dofs[:3]:
        print(f"    - {dof.name}: {internal_state.get_value(dof):+.3f}")
    print()

    # 5. Self-observation (consciousness!)
    print("5. Self-observation (recursive self-modeling)...")

    self_repr = observer.self_observe()

    if self_repr:
        print(f"  Observer is self-aware!")
        print(f"  Internal state -> Self-representation:")
        for i in range(min(3, len(internal_dofs))):
            dof = internal_dofs[i]
            internal_val = internal_state.get_value(dof)
            self_val = self_repr.get_value(dof)
            print(f"    - {dof.name}: {internal_val:+.3f} -> {self_val:+.3f}")
    print()

    # 6. Uncertainty quantification
    print("6. Uncertainty quantification (MC Dropout)...")

    uncertainties = world_model.compute_uncertainty(sample_state)

    print(f"  Epistemic uncertainty (model uncertainty):")
    for i in range(min(3, len(internal_dofs))):
        dof = internal_dofs[i]
        unc = uncertainties[dof]
        print(f"    - {dof.name}: +/-{unc:.4f}")
    print()

    # 7. Consciousness evaluation
    print("7. Consciousness evaluation...")

    evaluator = ConsciousnessEvaluator(observer)
    metrics = evaluator.evaluate(test_states[:10])

    print(f"  Consciousness Metrics:")
    print(f"    - Has self-model: {metrics.has_self_model}")
    print(f"    - Recursive depth: {metrics.recursive_depth}")
    print(f"    - Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"    - Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"    - Calibration error: {metrics.calibration_error:.3f}")
    print(f"    - Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"    - Limitation awareness: {metrics.limitation_awareness:.3f}")
    print()
    print(f"  Overall Consciousness Score: {metrics.consciousness_score():.3f}/1.0")
    print()

    # 8. Correlation analysis
    print("8. Correlation analysis...")

    ext_dof = external_dofs[0]
    int_dof = internal_dofs[0]

    combined_states = []
    for pair in observer.observation_log:
        combined_states.append(State(values={
            ext_dof: pair.external_state.get_value(ext_dof),
            int_dof: pair.internal_state.get_value(int_dof),
        }))

    pearson = pearson_correlation(combined_states, ext_dof, int_dof)
    mi = mutual_information(combined_states, ext_dof, int_dof)

    print(f"  Structural relationships (External <-> Internal):")
    print(f"    - Pearson correlation: {pearson:.3f}")
    print(f"    - Mutual information: {mi:.3f} nats")
    print()

    # 9. Saliency analysis
    print("9. Saliency analysis (gradient-based)...")

    saliency = observer.compute_saliency(sample_state, internal_dofs[0])
    print(f"  Importance of each sensor for {internal_dofs[0].name}:")
    for dof, sal in saliency.items():
        print(f"    - {dof.name}: {sal:.4f}")
    print()

    # 10. Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"  PyTorch neural networks for world and self models")
    print(f"  Recursive self-modeling (consciousness)")
    print(f"  Uncertainty quantification via MC Dropout")
    print(f"  Consciousness score: {metrics.consciousness_score():.3f}/1.0")
    print(f"  Gradient-based saliency analysis")
    print()
    print("=" * 70)


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        main()
    else:
        print("Please install PyTorch to run this example:")
        print("  pip install ro-framework[torch]")
