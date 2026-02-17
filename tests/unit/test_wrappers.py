"""Unit tests for integration/wrappers.py."""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.integration.wrappers import (
    create_dofs_for_vector,
    wrap_callable,
)
from ro_framework.observer.observer import Observer

try:
    import torch
    import torch.nn as nn
    from ro_framework.integration.wrappers import wrap_torch_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# create_dofs_for_vector
# ---------------------------------------------------------------------------

class TestCreateDofsForVector:

    def test_basic(self):
        dofs = create_dofs_for_vector(5, prefix="x")
        assert len(dofs) == 5
        assert all(isinstance(d, PolarDoF) for d in dofs)
        assert dofs[0].name == "x_0"
        assert dofs[4].name == "x_4"

    def test_unbounded(self):
        dofs = create_dofs_for_vector(3)
        assert dofs[0].polar_type == PolarDoFType.CONTINUOUS_REAL

    def test_bounded(self):
        dofs = create_dofs_for_vector(2, pole_negative=-1.0, pole_positive=1.0)
        assert dofs[0].polar_type == PolarDoFType.CONTINUOUS_BOUNDED

    def test_zero_dims(self):
        dofs = create_dofs_for_vector(0)
        assert dofs == []


# ---------------------------------------------------------------------------
# wrap_callable
# ---------------------------------------------------------------------------

class TestWrapCallable:

    def test_identity_function(self):
        # Use bounded DoFs to keep normalize/denormalize linear and exact
        dofs = create_dofs_for_vector(3, pole_negative=-10.0, pole_positive=10.0)
        obs = wrap_callable(lambda x: x, input_dofs=dofs, output_dofs=dofs)
        assert isinstance(obs, Observer)
        assert obs.name == "callable_observer"

        ext = State(values={dofs[0]: 1.0, dofs[1]: 2.0, dofs[2]: 3.0})
        result = obs.observe(ext)
        for d in dofs:
            assert abs(result.get_value(d) - ext.get_value(d)) < 1e-6

    def test_linear_function(self):
        """Callable receives/returns normalized vectors; verify roundtrip."""
        in_dofs = create_dofs_for_vector(2, prefix="in", pole_negative=-10.0, pole_positive=10.0)
        out_dofs = create_dofs_for_vector(3, prefix="out", pole_negative=-10.0, pole_positive=10.0)

        # fn operates on normalized [-1, 1] vectors and returns same scale
        def linear(x):
            return np.array([x[0] + x[1], x[0] - x[1], x[0] * 0.5])

        obs = wrap_callable(linear, in_dofs, out_dofs, name="linear_obs")
        assert obs.name == "linear_obs"

        # Input: 3.0 and 1.0 in [-10, 10] → normalized to 0.3 and 0.1
        ext = State(values={in_dofs[0]: 3.0, in_dofs[1]: 1.0})
        result = obs.observe(ext)

        # fn(0.3, 0.1) = [0.4, 0.2, 0.15] → denormalized in [-10, 10]: [4, 2, 1.5]
        assert abs(result.get_value(out_dofs[0]) - 4.0) < 1e-4
        assert abs(result.get_value(out_dofs[1]) - 2.0) < 1e-4
        assert abs(result.get_value(out_dofs[2]) - 1.5) < 1e-4

    def test_with_self_model(self):
        dofs = create_dofs_for_vector(2)
        obs = wrap_callable(
            lambda x: x * 2,
            input_dofs=dofs,
            output_dofs=dofs,
            self_model_fn=lambda x: x,  # identity self-model
        )
        assert obs.self_model is not None
        assert obs.recursive_depth() >= 1

    def test_observation_log_populated(self):
        dofs = create_dofs_for_vector(2)
        obs = wrap_callable(lambda x: x, dofs, dofs)
        for i in range(5):
            obs.observe(State(values={dofs[0]: float(i), dofs[1]: 0.0}))
        assert len(obs.observation_log) == 5

    def test_know_after_observations(self):
        in_dofs = create_dofs_for_vector(1, prefix="ext")
        out_dofs = create_dofs_for_vector(1, prefix="int")
        obs = wrap_callable(lambda x: x, in_dofs, out_dofs)

        rng = np.random.default_rng(42)
        for _ in range(50):
            obs.observe(State(values={in_dofs[0]: float(rng.uniform(-5, 5))}))

        assert obs.know(in_dofs[0], threshold=0.7)


# ---------------------------------------------------------------------------
# wrap_torch_model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestWrapTorchModel:

    def test_basic_wrap(self):
        in_dofs = create_dofs_for_vector(4, prefix="in")
        out_dofs = create_dofs_for_vector(2, prefix="out")
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        obs = wrap_torch_model(model, in_dofs, out_dofs)
        assert obs.name == "torch_observer"

        ext = State(values={d: 0.5 for d in in_dofs})
        result = obs.observe(ext)
        for d in out_dofs:
            assert result.get_value(d) is not None

    def test_with_self_model(self):
        dofs = create_dofs_for_vector(3, prefix="d")
        world = nn.Linear(3, 3)
        self_m = nn.Linear(3, 3)
        obs = wrap_torch_model(world, dofs, dofs, self_model=self_m)
        assert obs.self_model is not None
        assert obs.recursive_depth() >= 1

    def test_batch_observe(self):
        in_dofs = create_dofs_for_vector(2, prefix="in")
        out_dofs = create_dofs_for_vector(2, prefix="out")
        model = nn.Linear(2, 2)
        obs = wrap_torch_model(model, in_dofs, out_dofs)

        rng = np.random.default_rng(0)
        states = [
            State(values={d: float(rng.uniform(-1, 1)) for d in in_dofs})
            for _ in range(8)
        ]
        results = obs.observe_batch(states)
        assert len(results) == 8
        assert len(obs.observation_log) == 8

    def test_saliency(self):
        in_dofs = create_dofs_for_vector(3, prefix="in")
        out_dofs = create_dofs_for_vector(2, prefix="out")
        model = nn.Linear(3, 2)
        obs = wrap_torch_model(model, in_dofs, out_dofs)

        ext = State(values={d: 0.5 for d in in_dofs})
        sal = obs.compute_saliency(ext, out_dofs[0])
        assert len(sal) == 3
        # Gradients through nn.Linear should be non-zero
        assert any(v > 0.0 for v in sal.values())
