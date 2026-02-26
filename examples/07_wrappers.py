"""
Wrappers Example

Demonstrates the convenience functions for quickly wrapping existing
models as Observers without manually constructing DoFs and mappings.

Covers:
- wrap_callable: wrap any fn(ndarray) -> ndarray
- wrap_torch_model: wrap a PyTorch nn.Module
- create_dofs_for_vector: quick DoF creation for flat vectors
- Batch observation via observe_batch()
"""

import numpy as np

from ro_framework import State
from ro_framework.integration.wrappers import create_dofs_for_vector, wrap_callable


def main() -> None:
    """Run the wrappers example."""
    print("=" * 60)
    print("Recursive Observer Framework - Wrappers Example")
    print("=" * 60)
    print()

    # ---- 1. wrap_callable with a numpy function ----
    print("1. wrap_callable — wrapping a plain numpy function")
    print("-" * 50)

    # A simple function: 3 inputs -> 2 outputs
    # Note: wrap_callable normalizes inputs/outputs through DoF.normalize/denormalize.
    # For unbounded PolarDoFs this is a tanh-like mapping, so the function operates
    # on normalized values in [-1, 1], not raw values.
    def my_model(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + x[1], x[1] - x[2]])

    input_dofs = create_dofs_for_vector(3, prefix="in")
    output_dofs = create_dofs_for_vector(2, prefix="out")

    observer = wrap_callable(my_model, input_dofs, output_dofs, name="numpy_model")

    print(f"   Observer: {observer.name}")
    print(f"   Input DoFs:  {[d.name for d in observer.external_dofs]}")
    print(f"   Output DoFs: {[d.name for d in observer.internal_dofs]}")

    # Observe
    ext = State(values={input_dofs[0]: 1.0, input_dofs[1]: 2.0, input_dofs[2]: 3.0})
    result = observer.observe(ext)
    for d in output_dofs:
        print(f"   {d.name} = {result.get_value(d):.4f}")
    print()

    # ---- 2. Batch observation ----
    print("2. Batch observation — vectorized numpy path")
    print("-" * 50)

    # The wrapped function also accepts (N, 3) arrays for batching
    def batched_model(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([x[0] + x[1], x[1] * x[2]])
        return np.column_stack([x[:, 0] + x[:, 1], x[:, 1] * x[:, 2]])

    observer2 = wrap_callable(batched_model, input_dofs, output_dofs, name="batched")

    rng = np.random.default_rng(42)
    states = [
        State(values={d: rng.uniform(-5, 5) for d in input_dofs})
        for _ in range(20)
    ]

    observer2.observe_batch(states)
    print(f"   Batched {len(states)} states in one call")
    print(f"   Observation log: {len(observer2.observation_log)} entries")
    print()

    # ---- 3. Knowledge assessment on wrapped model ----
    print("3. Knowledge assessment on wrapped callable")
    print("-" * 50)

    # A model with a clear linear relationship plus noise
    def noisy_linear(x: np.ndarray) -> np.ndarray:
        return x * 2.5 + np.random.default_rng(0).normal(0, 0.1, size=x.shape)

    in_dofs = create_dofs_for_vector(1, prefix="x")
    out_dofs = create_dofs_for_vector(1, prefix="y")
    obs = wrap_callable(noisy_linear, in_dofs, out_dofs, name="noisy_linear")

    for _ in range(100):
        val = rng.uniform(-5, 5)
        obs.observe(State(values={in_dofs[0]: val}))

    k = obs.assess_knowledge(in_dofs[0])
    print(f"   Correlation (ρ): {k.correlation:.3f}")
    print(f"   Systematic error (ε): {k.systematic_error:.3f}")
    print(f"   Random noise (σ): {k.random_error:.3f}")
    print(f"   Knowledge type: {k.knowledge_type}")
    print()

    # ---- 4. PyTorch wrapper (if available) ----
    try:
        import torch.nn as nn
        from ro_framework.integration.wrappers import wrap_torch_model

        print("4. wrap_torch_model — wrapping a PyTorch nn.Module")
        print("-" * 50)

        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
        t_in = create_dofs_for_vector(4, prefix="feat")
        t_out = create_dofs_for_vector(2, prefix="pred")

        torch_obs = wrap_torch_model(model, t_in, t_out, name="mlp")
        print(f"   Observer: {torch_obs.name}")
        print(f"   Type: {type(torch_obs).__name__}")

        ext = State(values={d: rng.uniform(-1, 1) for d in t_in})
        result = torch_obs.observe(ext)
        print(f"   Output: [{result.get_value(t_out[0]):.4f}, "
              f"{result.get_value(t_out[1]):.4f}]")
        print()

    except ImportError:
        print("4. (Skipped — PyTorch not installed)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
