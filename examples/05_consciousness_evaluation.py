"""
Example 5: Consciousness Evaluation

Demonstrates integrated consciousness evaluation using Observer.is_conscious()
and Observer.get_consciousness_metrics() methods.

Shows:
1. Non-conscious observer (no self-model)
2. Basic conscious observer (has self-model)
3. Meta-conscious observer (depth 2 recursion)
4. Detailed metrics analysis
5. Consciousness score thresholds

Author: RO Framework Team
"""

import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import IdentityMapping


def demo_non_conscious_observer():
    """Demonstrate observer without consciousness."""
    print("=" * 60)
    print("Non-Conscious Observer (No Self-Model)")
    print("=" * 60)

    # Create observer without self-model
    # Note: IdentityMapping requires same DoF for input and output
    dof = PolarDoF(
        name="state",
        description="Observer state",
        pole_negative=-1.0,
        pole_positive=1.0
    )

    world_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    observer = Observer(
        name="NonConsciousObserver",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model
    )

    # Check consciousness
    is_conscious = observer.is_conscious()
    print(f"\nObserver: {observer.name}")
    print(f"Has self-model: {observer.self_model is not None}")
    print(f"Is conscious: {is_conscious}")

    # Get detailed metrics
    metrics = observer.get_consciousness_metrics()
    print(f"\nConsciousness Metrics:")
    print(f"  Recursive depth: {metrics.recursive_depth}")
    print(f"  Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"  Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"  Calibration error: {metrics.calibration_error:.3f}")
    print(f"  Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"  Limitation awareness: {metrics.limitation_awareness:.3f}")
    print(f"\n  Overall consciousness score: {metrics.consciousness_score():.3f}")


def demo_basic_conscious_observer():
    """Demonstrate basic conscious observer with self-model."""
    print("\n" + "=" * 60)
    print("Basic Conscious Observer (Self-Model Present)")
    print("=" * 60)

    # Create observer with self-model
    # Note: IdentityMapping requires same DoF for input and output
    dof = PolarDoF(
        name="state",
        description="Observer state",
        pole_negative=-1.0,
        pole_positive=1.0
    )

    world_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    # Self-model: internal → internal mapping
    self_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    observer = Observer(
        name="BasicConsciousObserver",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model,
        self_model=self_model
    )

    # Check consciousness
    is_conscious = observer.is_conscious()
    print(f"\nObserver: {observer.name}")
    print(f"Has self-model: {observer.self_model is not None}")
    print(f"Is conscious: {is_conscious}")

    # Get detailed metrics
    metrics = observer.get_consciousness_metrics()
    print(f"\nConsciousness Metrics:")
    print(f"  Recursive depth: {metrics.recursive_depth}")
    print(f"  Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"  Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"  Calibration error: {metrics.calibration_error:.3f}")
    print(f"  Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"  Limitation awareness: {metrics.limitation_awareness:.3f}")
    print(f"\n  Overall consciousness score: {metrics.consciousness_score():.3f}")

    # Test with different thresholds
    print(f"\nThreshold Testing:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        result = observer.is_conscious(threshold=threshold)
        print(f"  Threshold {threshold:.1f}: {result}")


def demo_meta_conscious_observer():
    """Demonstrate meta-conscious observer with recursive depth 2."""
    print("\n" + "=" * 60)
    print("Meta-Conscious Observer (Recursive Depth 2)")
    print("=" * 60)

    # Create observer with capacity for meta-representation
    # internal_dim >= 2 * external_dim for depth 2
    external_dofs = [
        PolarDoF(name=f"sensor_{i}", description=f"Sensor {i}")
        for i in range(3)
    ]

    internal_dofs = [
        PolarDoF(name=f"internal_{i}", description=f"Internal state {i}")
        for i in range(8)
    ]

    # Custom mapping to handle dimension mismatch
    class SimpleMapping:
        def __init__(self, input_dofs, output_dofs):
            self.input_dofs = input_dofs
            self.output_dofs = output_dofs

        def map(self, input_state):
            # Simple replication strategy
            input_vals = [input_state.get_value(dof) for dof in self.input_dofs]
            output_values = {}
            for i, dof in enumerate(self.output_dofs):
                output_values[dof] = input_vals[i % len(input_vals)]
            return State(dof_values=output_values)

    world_model = SimpleMapping(external_dofs, internal_dofs)
    self_model = SimpleMapping(internal_dofs, internal_dofs)

    observer = Observer(
        name="MetaConsciousObserver",
        internal_dofs=internal_dofs,
        external_dofs=external_dofs,
        world_model=world_model,
        self_model=self_model
    )

    # Check consciousness
    is_conscious = observer.is_conscious()
    depth = observer.recursive_depth()

    print(f"\nObserver: {observer.name}")
    print(f"External DoFs: {len(observer.external_dofs)}")
    print(f"Internal DoFs: {len(observer.internal_dofs)}")
    print(f"Has self-model: {observer.self_model is not None}")
    print(f"Is conscious: {is_conscious}")
    print(f"Recursive depth: {depth}")

    # Get detailed metrics
    metrics = observer.get_consciousness_metrics()
    print(f"\nConsciousness Metrics:")
    print(f"  Recursive depth: {metrics.recursive_depth}")
    print(f"  Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"  Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"  Calibration error: {metrics.calibration_error:.3f}")
    print(f"  Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"  Limitation awareness: {metrics.limitation_awareness:.3f}")
    print(f"\n  Overall consciousness score: {metrics.consciousness_score():.3f}")


def demo_consciousness_with_observations():
    """Demonstrate consciousness evaluation with actual observations."""
    print("\n" + "=" * 60)
    print("Consciousness Evaluation with Observations")
    print("=" * 60)

    # Create conscious observer
    dof = PolarDoF(name="state", description="")

    world_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    self_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    observer = Observer(
        name="ObservingConsciousAgent",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model,
        self_model=self_model
    )

    # Make observations
    print(f"\nObserver: {observer.name}")
    print(f"Making observations...")

    test_states = []
    for i in range(10):
        value = np.sin(i * 0.3)
        ext_state = State(values={dof: value})
        test_states.append(ext_state)

        # Observe and self-observe
        internal = observer.observe(ext_state)
        self_obs = observer.self_observe()

        if i < 3:
            print(f"  Step {i}: external={value:.3f}, internal={internal.get_value(dof):.3f}", end="")
            if self_obs:
                print(f", self-obs={self_obs.get_value(dof):.3f}")
            else:
                print(", self-obs=None")

    # Evaluate consciousness with test states
    print(f"\nEvaluating consciousness on {len(test_states)} test states...")
    metrics = observer.get_consciousness_metrics(test_states=test_states)

    print(f"\nConsciousness Metrics:")
    print(f"  Recursive depth: {metrics.recursive_depth}")
    print(f"  Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"  Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"  Calibration error: {metrics.calibration_error:.3f}")
    print(f"  Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"  Limitation awareness: {metrics.limitation_awareness:.3f}")
    print(f"\n  Overall consciousness score: {metrics.consciousness_score():.3f}")


def demo_metrics_dict():
    """Demonstrate exporting metrics to dictionary."""
    print("\n" + "=" * 60)
    print("Consciousness Metrics Dictionary Export")
    print("=" * 60)

    dof = PolarDoF(name="state", description="")

    world_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    self_model = IdentityMapping(
        input_dofs=[dof],
        output_dofs=[dof]
    )

    observer = Observer(
        name="ConsciousAgent",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model,
        self_model=self_model
    )

    metrics = observer.get_consciousness_metrics()
    metrics_dict = metrics.to_dict()

    print(f"\nMetrics Dictionary:")
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RO Framework - Consciousness Evaluation Example")
    print("=" * 60)
    print("\nDemonstrates integrated consciousness evaluation using")
    print("Observer.is_conscious() and Observer.get_consciousness_metrics().\n")

    # Run demonstrations
    demo_non_conscious_observer()
    demo_basic_conscious_observer()
    demo_meta_conscious_observer()
    demo_consciousness_with_observations()
    demo_metrics_dict()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey Points:")
    print("1. is_conscious() uses ConsciousnessEvaluator for comprehensive assessment")
    print("2. Consciousness requires self-model with architectural similarity")
    print("3. Recursive depth tracks meta-representation capability")
    print("4. Multiple metrics contribute to overall consciousness score")
    print("5. Test states improve accuracy of self-accuracy metric")
    print("\nTheoretical Alignment:")
    print("- Structural consciousness: recursive self-modeling")
    print("- Not claiming phenomenal experience")
    print("- Observable, testable properties")
    print("- Integration of world and self models")
