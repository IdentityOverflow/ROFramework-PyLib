"""
Example 5: Consciousness Evaluation

Demonstrates integrated consciousness evaluation using Observer.is_conscious()
and Observer.get_consciousness_metrics() methods.

Shows:
1. Non-conscious observer (no self-model)
2. Basic conscious observer (has self-model)
3. Meta-conscious observer (depth 2 via nested self-model)
4. Consciousness evaluation with observations
5. Metrics dictionary export
"""

import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import IdentityMapping


def _print_metrics(metrics):
    """Print consciousness metrics."""
    print(f"\nConsciousness Metrics:")
    print(f"  Recursive depth: {metrics.recursive_depth}")
    print(f"  Self-accuracy: {metrics.self_accuracy:.3f}")
    print(f"  Architectural similarity: {metrics.architectural_similarity:.3f}")
    print(f"  Calibration error: {metrics.calibration_error:.3f}")
    print(f"  Meta-cognitive capability: {metrics.meta_cognitive_capability:.3f}")
    print(f"  Limitation awareness: {metrics.limitation_awareness:.3f}")
    print(f"\n  Overall consciousness score: {metrics.consciousness_score():.3f}")


def demo_non_conscious_observer():
    """Demonstrate observer without consciousness."""
    print("=" * 60)
    print("Non-Conscious Observer (No Self-Model)")
    print("=" * 60)

    dof = PolarDoF(
        name="state",
        description="Observer state",
        pole_negative=-1.0,
        pole_positive=1.0
    )

    observer = Observer(
        name="NonConsciousObserver",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=IdentityMapping(input_dofs=[dof], output_dofs=[dof])
    )

    print(f"\nObserver: {observer.name}")
    print(f"Has self-model: {observer.self_model is not None}")
    print(f"Is conscious: {observer.is_conscious()}")

    _print_metrics(observer.get_consciousness_metrics())


def demo_basic_conscious_observer():
    """Demonstrate basic conscious observer with self-model."""
    print("\n" + "=" * 60)
    print("Basic Conscious Observer (Self-Model Present)")
    print("=" * 60)

    dof = PolarDoF(
        name="state",
        description="Observer state",
        pole_negative=-1.0,
        pole_positive=1.0
    )

    world_model = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
    self_model = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

    observer = Observer(
        name="BasicConsciousObserver",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model,
        self_model=self_model
    )

    # Feed observations so the evaluator has data to work with
    np.random.seed(42)
    for _ in range(20):
        observer.observe(State(values={dof: np.random.uniform(-1, 1)}))

    print(f"\nObserver: {observer.name}")
    print(f"Has self-model: {observer.self_model is not None}")
    print(f"Is conscious: {observer.is_conscious()}")

    _print_metrics(observer.get_consciousness_metrics())

    # Test with different thresholds
    print("\nThreshold Testing:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        result = observer.is_conscious(threshold=threshold)
        print(f"  Threshold {threshold:.1f}: {result}")


def demo_meta_conscious_observer():
    """Demonstrate meta-conscious observer with recursive depth 2."""
    print("\n" + "=" * 60)
    print("Meta-Conscious Observer (Recursive Depth 2)")
    print("=" * 60)

    dof = PolarDoF(
        name="state",
        description="Observer state",
        pole_negative=-1.0,
        pole_positive=1.0
    )

    world_model = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

    # Depth 2: self-model itself has a self_model attribute
    inner_self = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
    outer_self = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
    outer_self.self_model = inner_self  # nested recursion -> depth 2

    observer = Observer(
        name="MetaConsciousObserver",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model,
        self_model=outer_self
    )

    # Feed observations
    np.random.seed(42)
    for _ in range(20):
        observer.observe(State(values={dof: np.random.uniform(-1, 1)}))

    print(f"\nObserver: {observer.name}")
    print(f"Has self-model: {observer.self_model is not None}")
    print(f"Recursive depth: {observer.recursive_depth()}")

    _print_metrics(observer.get_consciousness_metrics())


def demo_consciousness_with_observations():
    """Demonstrate consciousness evaluation with actual observations."""
    print("\n" + "=" * 60)
    print("Consciousness Evaluation with Observations")
    print("=" * 60)

    dof = PolarDoF(name="state", pole_negative=-1.0, pole_positive=1.0)

    world_model = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
    self_model = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

    observer = Observer(
        name="ObservingConsciousAgent",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=world_model,
        self_model=self_model
    )

    print(f"\nObserver: {observer.name}")
    print("Making observations...")

    test_states = []
    for i in range(20):
        value = np.sin(i * 0.3)
        ext_state = State(values={dof: value})
        test_states.append(ext_state)

        internal = observer.observe(ext_state)
        self_obs = observer.self_observe()

        if i < 3:
            print(f"  Step {i}: external={value:.3f}, internal={internal.get_value(dof):.3f}", end="")
            if self_obs:
                print(f", self-obs={self_obs.get_value(dof):.3f}")
            else:
                print(", self-obs=None")

    print(f"\nEvaluating consciousness on {len(test_states)} test states...")
    _print_metrics(observer.get_consciousness_metrics(test_states=test_states))


def demo_metrics_dict():
    """Demonstrate exporting metrics to dictionary."""
    print("\n" + "=" * 60)
    print("Consciousness Metrics Dictionary Export")
    print("=" * 60)

    dof = PolarDoF(name="state", pole_negative=-1.0, pole_positive=1.0)

    observer = Observer(
        name="ConsciousAgent",
        internal_dofs=[dof],
        external_dofs=[dof],
        world_model=IdentityMapping(input_dofs=[dof], output_dofs=[dof]),
        self_model=IdentityMapping(input_dofs=[dof], output_dofs=[dof])
    )

    # Feed observations
    for i in range(20):
        observer.observe(State(values={dof: np.sin(i * 0.2)}))

    metrics = observer.get_consciousness_metrics()
    metrics_dict = metrics.to_dict()

    print("\nMetrics Dictionary:")
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

    demo_non_conscious_observer()
    demo_basic_conscious_observer()
    demo_meta_conscious_observer()
    demo_consciousness_with_observations()
    demo_metrics_dict()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey Points:")
    print("1. is_conscious() uses ConsciousnessEvaluator for behavioral assessment")
    print("2. Consciousness requires self-model with architectural similarity")
    print("3. Recursive depth follows the structural self-model chain")
    print("4. Multiple metrics contribute to overall consciousness score")
    print("5. Observations improve accuracy of behavioral metrics")
