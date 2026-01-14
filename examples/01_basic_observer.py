"""
Basic Observer Example

Demonstrates creating a simple observer with a world model.
This example shows the core concepts of the RO Framework:
- Defining Degrees of Freedom (DoFs)
- Creating States
- Building an Observer with a world model
- Observing external states
"""

from ro_framework import PolarDoF, PolarDoFType, State, Observer


def main() -> None:
    """Run the basic observer example."""
    print("=" * 60)
    print("Recursive Observer Framework - Basic Observer Example")
    print("=" * 60)
    print()

    # 1. Define Degrees of Freedom
    print("1. Defining Degrees of Freedom...")
    sensor_dof = PolarDoF(
        name="sensor_reading",
        description="External sensor input",
        pole_negative=-1.0,
        pole_positive=1.0,
        polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
    )

    latent_dof = PolarDoF(
        name="latent_state",
        description="Internal latent representation",
        pole_negative=-10.0,
        pole_positive=10.0,
        polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
    )

    print(f"  - External DoF: {sensor_dof.name}")
    print(f"    Domain: {sensor_dof.domain()}")
    print(f"  - Internal DoF: {latent_dof.name}")
    print(f"    Domain: {latent_dof.domain()}")
    print()

    # 2. Create a simple world model (mapping)
    print("2. Creating world model...")

    class SimpleWorldModel:
        """Simple linear mapping from sensor to latent space."""

        def __call__(self, external_state: State) -> State:
            sensor_value = external_state.get_value(sensor_dof)

            if sensor_value is None:
                latent_value = 0.0
            else:
                # Simple scaling: multiply by 10
                latent_value = sensor_value * 10.0

            return State(values={latent_dof: latent_value})

    world_model = SimpleWorldModel()
    print("  - World model created (external → internal mapping)")
    print()

    # 3. Create an observer
    print("3. Creating observer...")
    observer = Observer(
        name="basic_observer",
        internal_dofs=[latent_dof],
        external_dofs=[sensor_dof],
        world_model=world_model,
    )

    print(f"  - Observer: {observer.name}")
    print(f"  - Internal DoFs: {len(observer.internal_dofs)}")
    print(f"  - External DoFs: {len(observer.external_dofs)}")
    print(f"  - Is conscious? {observer.is_conscious()}")
    print()

    # 4. Perform observations
    print("4. Performing observations...")
    test_values = [-0.8, -0.3, 0.0, 0.5, 1.0]

    for sensor_value in test_values:
        # Create external state
        external_state = State(values={sensor_dof: sensor_value})

        # Observe
        internal_state = observer.observe(external_state)
        latent_value = internal_state.get_value(latent_dof)

        print(f"  - Sensor: {sensor_value:+.2f} → Latent: {latent_value:+.2f}")

    print()

    # 5. Demonstrate state distance
    print("5. Computing state distances...")
    state1 = State(values={sensor_dof: -1.0})
    state2 = State(values={sensor_dof: 1.0})

    distance = state1.distance_to(state2)
    print(f"  - Distance from {state1.get_value(sensor_dof)} to " f"{state2.get_value(sensor_dof)}: {distance:.2f}")
    print()

    # 6. Demonstrate DoF normalization
    print("6. DoF normalization (for neural networks)...")
    for value in [-1.0, 0.0, 1.0]:
        normalized = sensor_dof.normalize(value)
        denormalized = sensor_dof.denormalize(normalized)
        print(f"  - Value: {value:+.2f} → Normalized: {normalized:+.2f} → " f"Denormalized: {denormalized:+.2f}")

    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
