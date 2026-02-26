"""
Serialization Example

Demonstrates saving and loading an Observer with its observation history.
This enables workflows that span multiple sessions — observe in one run,
analyze knowledge in a later run without re-observing.

Key points:
- Observer.save() writes DoFs, observation log, and metadata to JSON
- Observer.load() restores everything except the world model (which must be re-supplied)
- All DoF types (Polar, Scalar, Categorical, Derived) serialize correctly
"""

import tempfile
from pathlib import Path

import numpy as np

from ro_framework import Observer, PolarDoF, PolarDoFType, State
from ro_framework.integration.wrappers import wrap_callable


def make_world_model():
    """A simple linear mapping: doubles the input."""
    def fn(x: np.ndarray) -> np.ndarray:
        return x * 2.0
    return fn


def main() -> None:
    """Run the serialization example."""
    print("=" * 60)
    print("Recursive Observer Framework - Serialization Example")
    print("=" * 60)
    print()

    # 1. Set up DoFs and observer
    input_dofs = [
        PolarDoF(name="sensor", pole_negative=-10.0, pole_positive=10.0,
                 polar_type=PolarDoFType.CONTINUOUS_BOUNDED),
    ]
    output_dofs = [
        PolarDoF(name="estimate", pole_negative=-np.inf, pole_positive=np.inf,
                 polar_type=PolarDoFType.CONTINUOUS_REAL),
    ]

    fn = make_world_model()
    observer = wrap_callable(fn, input_dofs, output_dofs, name="doubler")

    # 2. Collect observations
    print("1. Collecting 50 observations...")
    rng = np.random.default_rng(42)
    for _ in range(50):
        val = rng.uniform(-10, 10)
        ext = State(values={input_dofs[0]: val})
        observer.observe(ext)

    print(f"   Observation log size: {len(observer.observation_log)}")

    # 3. Assess knowledge before saving
    k = observer.assess_knowledge(input_dofs[0])
    print(f"   Knowledge: ρ={k.correlation:.3f}, ε={k.systematic_error:.3f}")
    print()

    # 4. Save to disk
    save_path = Path(tempfile.gettempdir()) / "observer_checkpoint.json"
    print(f"2. Saving observer to {save_path}...")
    observer.save(save_path)
    print(f"   File size: {save_path.stat().st_size:,} bytes")
    print()

    # 5. Load in a "new session" — re-supply the world model
    print("3. Loading observer (simulating a new session)...")
    fn2 = make_world_model()  # same function, new instance
    from ro_framework.integration.wrappers import _CallableMapping
    world_model = _CallableMapping(fn2, input_dofs, output_dofs)

    loaded = Observer.load(save_path, world_model=world_model)
    print(f"   Name: {loaded.name}")
    print(f"   Observation log size: {len(loaded.observation_log)}")
    print(f"   External DoFs: {[d.name for d in loaded.external_dofs]}")
    print(f"   Internal DoFs: {[d.name for d in loaded.internal_dofs]}")
    print()

    # 6. Verify knowledge is the same
    k2 = loaded.assess_knowledge(loaded.external_dofs[0])
    print("4. Verifying knowledge matches...")
    print(f"   Original:  ρ={k.correlation:.3f}, ε={k.systematic_error:.3f}")
    print(f"   Loaded:    ρ={k2.correlation:.3f}, ε={k2.systematic_error:.3f}")
    print()

    # 7. Continue observing on the loaded observer
    print("5. Continuing observation on loaded observer...")
    for _ in range(20):
        val = rng.uniform(-10, 10)
        ext = State(values={input_dofs[0]: val})
        loaded.observe(ext)
    print(f"   Observation log size: {len(loaded.observation_log)} (was 50)")
    print()

    # Cleanup
    save_path.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
