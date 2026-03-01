"""
Knowledge Assessment Example

Demonstrates the graded knowledge system: K(d_ext) = (rho, epsilon, sigma, C).
Shows how an observer builds knowledge through observation history and how
different mapping qualities produce different knowledge types.

Concepts:
- Knowledge is graded, not binary (know/don't know)
- correlation (rho): how strongly internal tracks external
- systematic_error (epsilon): heteroscedasticity — position-dependent error
- random_error (sigma): noise in the mapping (regression residuals)
- calibration (C): whether errors are uniform across the range
- Knowledge types: "strong", "weak", "false", "uncertain"
"""

import numpy as np

from ro_framework.core.dof import PolarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer


def _print_assessment(observer, dof, assessment):
    """Print a knowledge assessment nicely."""
    print(f"\nObserver: {observer.name}")
    print(f"Observations: {observer.get_memory_size()}")
    print(f"\nKnowledge of '{dof.name}':")
    print(f"  Type:             {assessment.knowledge_type}")
    print(f"  Correlation (rho): {assessment.correlation:.3f}")
    print(f"  Bias (epsilon):    {assessment.systematic_error:.3f}")
    print(f"  Noise (sigma):     {assessment.random_error:.3f}")
    print(f"  Calibration (C):   {assessment.calibration:.3f}")
    print(f"  Knows? {observer.know(dof)}")


def demo_strong_knowledge():
    """An accurate observer builds strong knowledge."""
    print("=" * 60)
    print("Strong Knowledge (Accurate Linear Mapping)")
    print("=" * 60)

    sensor = PolarDoF(
        name="temperature",
        pole_negative=-10.0,
        pole_positive=10.0,
        polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
    )
    internal = PolarDoF(
        name="temp_estimate",
        pole_negative=-10.0,
        pole_positive=10.0,
        polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
    )

    class AccurateModel:
        """Near-perfect mapping with small uniform noise."""
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            return State(values={internal: val + np.random.randn() * 0.3})

    observer = Observer(
        name="AccurateObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=AccurateModel(),
    )

    np.random.seed(42)
    for _ in range(200):
        val = np.random.uniform(-10, 10)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    _print_assessment(observer, sensor, assessment)
    print("\n  -> High rho, low epsilon (uniform errors), high C")


def demo_false_knowledge():
    """A heteroscedastic observer has false knowledge — correlated but with
    position-dependent errors (accurate at one end, inaccurate at the other)."""
    print("\n" + "=" * 60)
    print("False Knowledge (Heteroscedastic / Confound Tracking)")
    print("=" * 60)

    sensor = PolarDoF(name="pressure", pole_negative=-10.0, pole_positive=10.0)
    internal = PolarDoF(name="pressure_est", pole_negative=-10.0, pole_positive=10.0)

    class ConfoundModel:
        """Tracks pressure well for positive values but poorly for negative.

        This mimics a feature that correlates with the label through a
        confound: accurate in one region, noisy in another.
        """
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            # Accurate for positive, very noisy for negative
            if val > 0:
                return State(values={internal: val + np.random.randn() * 0.2})
            else:
                return State(values={internal: val + np.random.randn() * 5.0})

    observer = Observer(
        name="ConfoundObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=ConfoundModel(),
    )

    np.random.seed(42)
    for _ in range(200):
        val = np.random.uniform(-10, 10)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    _print_assessment(observer, sensor, assessment)
    print("\n  -> High rho (still correlated overall), high epsilon")
    print("     (error magnitude depends on position = heteroscedasticity)")


def demo_uncertain_knowledge():
    """An uncorrelated but consistently noisy observer has uncertain knowledge."""
    print("\n" + "=" * 60)
    print("Uncertain Knowledge (No Tracking, But Consistent Noise)")
    print("=" * 60)

    sensor = PolarDoF(name="signal", pole_negative=-5.0, pole_positive=5.0)
    internal = PolarDoF(name="signal_est", pole_negative=-5.0, pole_positive=5.0)

    class UncorrelatedModel:
        """No correlation with input — output is pure noise.

        But the noise is uniform: same level everywhere. The observer
        doesn't track the signal, but at least its errors are consistent.
        """
        def __call__(self, ext_state: State) -> State:
            # Ignores input entirely, outputs uniform noise
            return State(values={internal: np.random.randn() * 3.0})

    observer = Observer(
        name="UncorrelatedObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=UncorrelatedModel(),
    )

    np.random.seed(42)
    for _ in range(200):
        val = np.random.uniform(-5, 5)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    _print_assessment(observer, sensor, assessment)
    print("\n  -> Low rho (no tracking), but high C (errors are uniform)")
    print("     'I consistently don't know' = uncertain, not weak")


def demo_weak_knowledge():
    """A noisy observer with position-dependent errors has weak knowledge."""
    print("\n" + "=" * 60)
    print("Weak Knowledge (Moderate Correlation + Heteroscedastic Noise)")
    print("=" * 60)

    sensor = PolarDoF(name="input", pole_negative=-5.0, pole_positive=5.0)
    internal = PolarDoF(name="input_est", pole_negative=-5.0, pole_positive=5.0)

    class WeakModel:
        """Moderate tracking with noise that varies by position."""
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            # Moderate signal + noise that grows with |val|
            noise = np.random.randn() * (abs(val) * 0.5 + 0.5)
            return State(values={internal: val * 0.5 + noise})

    observer = Observer(
        name="WeakObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=WeakModel(),
    )

    np.random.seed(42)
    for _ in range(200):
        val = np.random.uniform(-5, 5)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    _print_assessment(observer, sensor, assessment)
    print("\n  -> Moderate rho, heteroscedastic noise -> low C -> 'weak'")


def demo_insufficient_data():
    """With too few observations, knowledge cannot be assessed."""
    print("\n" + "=" * 60)
    print("Insufficient Data")
    print("=" * 60)

    sensor = PolarDoF(name="x", pole_negative=-1.0, pole_positive=1.0)
    internal = PolarDoF(name="x_est", pole_negative=-1.0, pole_positive=1.0)

    class SimpleModel:
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor) or 0.0
            return State(values={internal: val})

    observer = Observer(
        name="FewObsObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=SimpleModel(),
    )

    # Only 3 observations — below min_samples=10
    for v in [0.1, 0.5, -0.3]:
        observer.observe(State(values={sensor: v}))

    assessment = observer.assess_knowledge(sensor)
    print(f"\nObserver: {observer.name}")
    print(f"Observations: {observer.get_memory_size()}")
    print(f"Assessment: {assessment}")
    print(f"Knows? {observer.know(sensor)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RO Framework - Knowledge Assessment Example")
    print("K(d_ext) = (rho, epsilon, sigma, C)")
    print("=" * 60)
    print()

    demo_strong_knowledge()
    demo_false_knowledge()
    demo_uncertain_knowledge()
    demo_weak_knowledge()
    demo_insufficient_data()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKnowledge Types:")
    print("  strong:    High rho, low epsilon (uniform errors), good calibration C")
    print("  false:     High rho but high epsilon (position-dependent errors)")
    print("  uncertain: Low rho but high C (consistently wrong = well-calibrated noise)")
    print("  weak:      Everything else (moderate rho, or heteroscedastic errors)")
    print("\nKey Insight: Knowledge is observer-relative and graded,")
    print("not a binary property of the external world.")
    print("\nepsilon measures heteroscedasticity: does error magnitude depend on")
    print("where you are in the input range? High epsilon = confound tracking.")
    print("C measures error consistency: are errors uniform across the range?")
