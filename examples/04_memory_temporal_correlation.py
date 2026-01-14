"""
Memory and Temporal Correlation Example

Demonstrates the integrated memory system using proper temporal correlation
analysis from the correlation module. Shows how memory is now a structural
property based on correlation across temporal DoF, not just a buffer.
"""

import numpy as np
from ro_framework.core.dof import PolarDoF, ScalarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import IdentityMapping


def demo_memory_detection():
    """Demonstrate memory detection using temporal correlation."""
    print("=" * 60)
    print("Memory Detection via Temporal Correlation")
    print("=" * 60)

    # Create DoFs
    temporal_dof = ScalarDoF(name="time", min_value=0, max_value=1000)
    sensor_dof = PolarDoF(name="sensor", pole_negative=-10, pole_positive=10)

    # Create observer with temporal DoF
    observer = Observer(
        name="MemoryObserver",
        internal_dofs=[sensor_dof],
        external_dofs=[sensor_dof],
        world_model=IdentityMapping(
            input_dofs=[sensor_dof],
            output_dofs=[sensor_dof]
        ),
        temporal_dof=temporal_dof
    )

    print(f"\nObserver: {observer.name}")
    print(f"Temporal DoF: {temporal_dof.name}")

    # Scenario 1: Strong temporal correlation (memory present)
    print("\n--- Scenario 1: Autocorrelated sequence (Memory Present) ---")
    np.random.seed(42)

    # Generate AR(1) process: x_t = 0.8 * x_{t-1} + noise
    x = 0
    for t in range(50):
        x = 0.8 * x + np.random.randn() * 0.5
        state = State(values={sensor_dof: float(np.clip(x, -10, 10))})
        observer.memory_buffer.append(state)

    has_memory = observer.has_memory(threshold=0.5)
    print(f"Has memory: {has_memory}")
    print(f"Memory buffer size: {observer.get_memory_size()}")

    # Analyze memory structure
    analysis = observer.analyze_memory_structure(max_lag=5)
    print(f"\nTemporal correlation profile (lags 1-5):")
    for dof, correlations in analysis.items():
        print(f"  {dof.name}: {[f'{c:.3f}' for c in correlations]}")

    # Scenario 2: Random uncorrelated sequence (no memory)
    print("\n--- Scenario 2: Random sequence (No Memory) ---")
    observer.clear_memory()

    for t in range(50):
        x = np.random.randn() * 5
        state = State(values={sensor_dof: float(np.clip(x, -10, 10))})
        observer.memory_buffer.append(state)

    has_memory = observer.has_memory(threshold=0.5)
    print(f"Has memory: {has_memory}")

    analysis = observer.analyze_memory_structure(max_lag=5)
    print(f"\nTemporal correlation profile:")
    for dof, correlations in analysis.items():
        print(f"  {dof.name}: {[f'{c:.3f}' for c in correlations]}")


def demo_memory_with_multiple_dofs():
    """Demonstrate memory analysis with multiple internal DoFs."""
    print("\n" + "=" * 60)
    print("Multi-DoF Memory Analysis")
    print("=" * 60)

    # Create DoFs
    temporal_dof = ScalarDoF(name="time", min_value=0, max_value=1000)
    latent1 = PolarDoF(name="latent1", pole_negative=-5, pole_positive=5)
    latent2 = PolarDoF(name="latent2", pole_negative=-5, pole_positive=5)

    observer = Observer(
        name="MultiDofObserver",
        internal_dofs=[latent1, latent2],
        external_dofs=[latent1, latent2],
        world_model=IdentityMapping(
            input_dofs=[latent1, latent2],
            output_dofs=[latent1, latent2]
        ),
        temporal_dof=temporal_dof
    )

    # Generate correlated sequences
    np.random.seed(42)
    x1 = 0
    x2 = 0

    for t in range(40):
        # x1 has strong autocorrelation
        x1 = 0.9 * x1 + np.random.randn() * 0.3

        # x2 is influenced by x1 (cross-correlation)
        x2 = 0.5 * x1 + 0.3 * x2 + np.random.randn() * 0.5

        state = State(values={
            latent1: float(np.clip(x1, -5, 5)),
            latent2: float(np.clip(x2, -5, 5))
        })
        observer.memory_buffer.append(state)

    print(f"\nMemory buffer size: {observer.get_memory_size()}")
    print(f"Has memory: {observer.has_memory(threshold=0.5)}")

    # Analyze individual DoF temporal correlations
    print("\nTemporal Correlation Analysis:")
    analysis = observer.analyze_memory_structure(max_lag=5)
    for dof, correlations in analysis.items():
        print(f"\n{dof.name}:")
        for lag, corr in enumerate(correlations, 1):
            print(f"  Lag {lag}: {corr:+.3f}")

    # Analyze cross-correlations between DoFs
    print("\nCross-correlation between DoFs:")
    corr = observer.get_memory_correlations(latent1, latent2)
    print(f"  {latent1.name} <-> {latent2.name}: {corr:.3f}")


def demo_observation_with_memory():
    """Demonstrate memory building through observations."""
    print("\n" + "=" * 60)
    print("Memory Building Through Observations")
    print("=" * 60)

    # Create DoFs
    temporal_dof = ScalarDoF(name="time", min_value=0, max_value=1000)
    sensor_dof = PolarDoF(name="sensor", pole_negative=-1, pole_positive=1)

    observer = Observer(
        name="ObservingMemorySystem",
        internal_dofs=[sensor_dof],
        external_dofs=[sensor_dof],
        world_model=IdentityMapping(
            input_dofs=[sensor_dof],
            output_dofs=[sensor_dof]
        ),
        temporal_dof=temporal_dof
    )

    print(f"Observer: {observer.name}")
    print("\nPerforming observations...")

    # Observe a sinusoidal pattern (smooth temporal structure)
    for t in range(30):
        value = np.sin(t * 0.2)
        external_state = State(values={sensor_dof: value})
        observer.observe(external_state)

        # Check memory every 10 steps
        if (t + 1) % 10 == 0:
            has_memory = observer.has_memory(threshold=0.5)
            print(f"  Step {t+1}: Memory={has_memory}, Buffer size={observer.get_memory_size()}")

    # Final analysis
    print("\nFinal Memory Analysis:")
    analysis = observer.analyze_memory_structure(max_lag=8)
    for dof, correlations in analysis.items():
        print(f"\n{dof.name} autocorrelation:")
        for lag, corr in enumerate(correlations[:5], 1):
            print(f"  Lag {lag}: {corr:+.3f}")


def main():
    """Run all memory demonstrations."""
    print("\n" + "=" * 60)
    print("Integrated Memory System Demonstration")
    print("Memory as Correlation Across Temporal DoF")
    print("=" * 60)

    demo_memory_detection()
    demo_memory_with_multiple_dofs()
    demo_observation_with_memory()

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nKey Points:")
    print("- Memory is now detected via temporal correlation (not just buffer)")
    print("- Uses correlation module's temporal_correlation() function")
    print("- analyze_memory_structure() provides detailed correlation profiles")
    print("- get_memory_correlations() checks cross-correlations between DoFs")
    print("- Properly integrates with Observer's temporal DoF")


if __name__ == "__main__":
    main()
