"""
Knowledge Assessment Example

Demonstrates the graded knowledge system: K(d_ext) = (rho, epsilon, sigma, C).
Shows how an observer builds knowledge through observation history and how
different mapping qualities produce different knowledge types.

Concepts:
- Knowledge is graded, not binary (know/don't know)
- correlation (rho): how strongly internal tracks external
- systematic_error (epsilon): consistent bias
- random_error (sigma): noise in the mapping
- calibration (C): whether stated uncertainty matches actual error
- Knowledge types: "strong", "weak", "false", "uncertain"
"""

import numpy as np

from ro_framework.core.dof import PolarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer


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
        """Near-perfect mapping with tiny noise."""
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            # Accurate mapping with small noise
            return State(values={internal: val + np.random.randn() * 0.05})

    observer = Observer(
        name="AccurateObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=AccurateModel(),
    )

    # Build observation history
    np.random.seed(42)
    for _ in range(100):
        val = np.random.uniform(-10, 10)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    print(f"\nObserver: {observer.name}")
    print(f"Observations: {observer.get_memory_size()}")
    print(f"\nKnowledge of '{sensor.name}':")
    print(f"  Type:             {assessment.knowledge_type}")
    print(f"  Correlation (rho): {assessment.correlation:.3f}")
    print(f"  Bias (epsilon):    {assessment.systematic_error:.3f}")
    print(f"  Noise (sigma):     {assessment.random_error:.3f}")
    print(f"  Calibration (C):   {assessment.calibration:.3f}")
    print(f"  Knows? {observer.know(sensor)}")


def demo_weak_knowledge():
    """A noisy observer has weak knowledge."""
    print("\n" + "=" * 60)
    print("Weak Knowledge (Noisy Mapping)")
    print("=" * 60)

    sensor = PolarDoF(name="signal", pole_negative=-5.0, pole_positive=5.0)
    internal = PolarDoF(name="signal_est", pole_negative=-5.0, pole_positive=5.0)

    class NoisyModel:
        """Mapping dominated by noise."""
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            # Signal buried in noise
            return State(values={internal: val * 0.3 + np.random.randn() * 3.0})

    observer = Observer(
        name="NoisyObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=NoisyModel(),
    )

    np.random.seed(42)
    for _ in range(100):
        val = np.random.uniform(-5, 5)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    print(f"\nObserver: {observer.name}")
    print(f"\nKnowledge of '{sensor.name}':")
    print(f"  Type:             {assessment.knowledge_type}")
    print(f"  Correlation (rho): {assessment.correlation:.3f}")
    print(f"  Bias (epsilon):    {assessment.systematic_error:.3f}")
    print(f"  Noise (sigma):     {assessment.random_error:.3f}")
    print(f"  Calibration (C):   {assessment.calibration:.3f}")
    print(f"  Knows? {observer.know(sensor)}")


def demo_false_knowledge():
    """A biased observer has false knowledge — correlated but systematically wrong."""
    print("\n" + "=" * 60)
    print("False Knowledge (Systematic Bias)")
    print("=" * 60)

    sensor = PolarDoF(name="pressure", pole_negative=-10.0, pole_positive=10.0)
    internal = PolarDoF(name="pressure_est", pole_negative=-10.0, pole_positive=10.0)

    class BiasedModel:
        """Correlated but with large systematic offset."""
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            # Tracks well but with large bias
            return State(values={internal: val * 0.8 + 5.0 + np.random.randn() * 0.1})

    observer = Observer(
        name="BiasedObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=BiasedModel(),
    )

    np.random.seed(42)
    for _ in range(100):
        val = np.random.uniform(-10, 10)
        observer.observe(State(values={sensor: val}))

    assessment = observer.assess_knowledge(sensor)
    print(f"\nObserver: {observer.name}")
    print(f"\nKnowledge of '{sensor.name}':")
    print(f"  Type:             {assessment.knowledge_type}")
    print(f"  Correlation (rho): {assessment.correlation:.3f}")
    print(f"  Bias (epsilon):    {assessment.systematic_error:.3f}")
    print(f"  Noise (sigma):     {assessment.random_error:.3f}")
    print(f"  Calibration (C):   {assessment.calibration:.3f}")
    print(f"  Knows? {observer.know(sensor)}")


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


def demo_knowledge_building():
    """Watch knowledge improve as observations accumulate."""
    print("\n" + "=" * 60)
    print("Knowledge Building Over Time")
    print("=" * 60)

    sensor = PolarDoF(name="input", pole_negative=-5.0, pole_positive=5.0)
    internal = PolarDoF(name="output", pole_negative=-5.0, pole_positive=5.0)

    class GoodModel:
        def __call__(self, ext_state: State) -> State:
            val = ext_state.get_value(sensor)
            if val is None:
                return State(values={internal: 0.0})
            return State(values={internal: val * 0.95 + np.random.randn() * 0.2})

    observer = Observer(
        name="LearningObserver",
        internal_dofs=[internal],
        external_dofs=[sensor],
        world_model=GoodModel(),
    )

    np.random.seed(42)
    print(f"\n{'Step':>6}  {'Type':>10}  {'Corr':>6}  {'Bias':>6}  {'Noise':>6}  {'Cal':>6}  Knows?")
    print("-" * 65)

    for step in range(200):
        val = np.random.uniform(-5, 5)
        observer.observe(State(values={sensor: val}))

        if (step + 1) in [10, 25, 50, 100, 200]:
            a = observer.assess_knowledge(sensor)
            if a is not None:
                knows = observer.know(sensor)
                print(f"{step+1:>6}  {a.knowledge_type:>10}  {a.correlation:>6.3f}  "
                      f"{a.systematic_error:>6.3f}  {a.random_error:>6.3f}  "
                      f"{a.calibration:>6.3f}  {knows}")
            else:
                print(f"{step+1:>6}  {'N/A':>10}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RO Framework - Knowledge Assessment Example")
    print("K(d_ext) = (rho, epsilon, sigma, C)")
    print("=" * 60)
    print()

    demo_strong_knowledge()
    demo_weak_knowledge()
    demo_false_knowledge()
    demo_insufficient_data()
    demo_knowledge_building()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKnowledge Types:")
    print("  strong:    High correlation, low bias, good calibration")
    print("  weak:      Low correlation or poor calibration")
    print("  false:     High correlation but systematic bias")
    print("  uncertain: Low correlation but well-calibrated uncertainty")
    print("\nKey Insight: Knowledge is observer-relative and graded,")
    print("not a binary property of the external world.")
