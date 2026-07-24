"""Tests for Closed(O) — d_meta, the consumption loop, and closure recognition.

Covers v2-migration phases A (d_meta + ClosureAssessment) and B
(consumption_gain in observe()). See docs/v2_migration.md and
docs/ro_framework.md §5.3.
"""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import ClosureAssessment, Observer


EXT = PolarDoF(name="ext_x")
INT = PolarDoF(name="int_y")


class ExtOnlyModel:
    """Probe-grade world model: y = 0.5 * x. Ignores any feedback."""

    name = "ext_only"
    input_dofs = [EXT]
    output_dofs = [INT]

    def __call__(self, state: State) -> State:
        return State(values={INT: 0.5 * float(state.get_value(EXT))})


class RecurrentModel:
    """Loop-capable world model: y' = 0.5*x + feedback (if routed).

    Its domain includes INT — the d_meta channel — so the observer's
    consumption injection has somewhere to land. Records the input state
    it received (spy) so tests can check what was actually routed.
    """

    name = "recurrent"
    input_dofs = [EXT, INT]
    output_dofs = [INT]

    def __init__(self):
        self.last_input: State | None = None

    def __call__(self, state: State) -> State:
        self.last_input = state
        fb = state.get_value(INT)
        fb = float(fb) if fb is not None else 0.0
        return State(values={INT: 0.5 * float(state.get_value(EXT)) + fb})


class IdentitySelfModel:
    """meta = internal value, declared on the same DoF (d_meta = [INT])."""

    name = "self_id"
    input_dofs = [INT]
    output_dofs = [INT]

    def __call__(self, state: State) -> State:
        return State(values={INT: float(state.get_value(INT))})


def _drive(observer: Observer, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        observer.observe(State(values={EXT: float(rng.normal())}))


class TestDMeta:
    def test_empty_without_self_model(self):
        obs = Observer("o", [INT], [EXT], ExtOnlyModel())
        assert obs.d_meta == []

    def test_reflects_self_model_output_dofs(self):
        obs = Observer("o", [INT], [EXT], ExtOnlyModel(),
                       self_model=IdentitySelfModel())
        assert obs.d_meta == [INT]


class TestConsumptionLoop:
    def test_gain_zero_is_pure_probe(self):
        """At g=0 no meta value is injected — exact pre-v2 behavior."""
        wm = RecurrentModel()
        obs = Observer("o", [INT], [EXT], wm,
                       self_model=IdentitySelfModel(), consumption_gain=0.0)
        _drive(obs, 3)
        assert wm.last_input.get_value(INT) is None

    def test_injection_scales_with_gain(self):
        wm = RecurrentModel()
        obs = Observer("o", [INT], [EXT], wm,
                       self_model=IdentitySelfModel(), consumption_gain=0.5)
        obs.observe(State(values={EXT: 1.0}))          # y = 0.5, no prior state
        obs.observe(State(values={EXT: 0.0}))          # meta(0.5) routed at g=0.5
        assert wm.last_input.get_value(INT) == pytest.approx(0.25)

    def test_first_observation_has_no_feedback(self):
        wm = RecurrentModel()
        obs = Observer("o", [INT], [EXT], wm,
                       self_model=IdentitySelfModel(), consumption_gain=1.0)
        obs.observe(State(values={EXT: 1.0}))
        assert wm.last_input.get_value(INT) is None

    def test_batch_falls_back_to_sequential_when_consuming(self):
        wm = RecurrentModel()
        wm.batch_call = lambda states: pytest.fail(
            "batch_call must not be used while the loop is closed")
        obs = Observer("o", [INT], [EXT], wm,
                       self_model=IdentitySelfModel(), consumption_gain=0.8)
        results = obs.observe_batch(
            [State(values={EXT: 1.0}), State(values={EXT: -1.0})])
        assert len(results) == 2
        assert len(obs.observation_log) == 2


class TestClosureRecognition:
    def test_probe_is_not_closed(self):
        """Structural gate: no consumption path -> never closed."""
        obs = Observer("o", [INT], [EXT], ExtOnlyModel(),
                       self_model=IdentitySelfModel(), consumption_gain=0.0)
        _drive(obs, 60)
        a = obs.closure_assessment()
        assert isinstance(a, ClosureAssessment)
        assert a.structural is False
        assert a.closed is False

    def test_closed_loop_is_recognized(self):
        """g>0 + recurrent domain -> AR(1) dynamics: d_meta(t) correlates
        with internal(t+1) far above external(t+1)."""
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=IdentitySelfModel(), consumption_gain=0.8)
        _drive(obs, 80)
        a = obs.closure_assessment()
        assert a.structural is True
        assert a.corr_internal > a.corr_external
        assert a.closed is True

    def test_open_dynamics_not_closed_despite_gain(self):
        """g>0 but the world model ignores the feedback: structurally wired,
        correlationally open — internal is driven by iid external only, so
        the consumption correlation cannot beat the external one decisively."""
        obs = Observer("o", [INT], [EXT], ExtOnlyModel(),
                       self_model=IdentitySelfModel(), consumption_gain=0.8)
        _drive(obs, 80)
        a = obs.closure_assessment()
        assert a.structural is True
        # y_t = 0.5 x_t with iid x: meta(t) predicts NEITHER side at lag 1;
        # closure must not be granted on structure alone.
        assert a.corr_internal < 0.4
        assert a.closed is False

    def test_insufficient_history(self):
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=IdentitySelfModel(), consumption_gain=0.8)
        _drive(obs, 4)
        a = obs.closure_assessment(min_samples=10)
        assert a.closed is False

    def test_is_closed_convenience(self):
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=IdentitySelfModel(), consumption_gain=0.8)
        _drive(obs, 80)
        assert obs.is_closed() is True


class TestSerialization:
    def test_consumption_gain_roundtrip(self):
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=IdentitySelfModel(), consumption_gain=0.7)
        _drive(obs, 5)
        d = obs.to_dict()
        assert d["consumption_gain"] == 0.7
        obs2 = Observer.from_dict(d, world_model=RecurrentModel(),
                                  self_model=IdentitySelfModel())
        assert obs2.consumption_gain == 0.7

    def test_legacy_dict_defaults_to_probe(self):
        obs = Observer("o", [INT], [EXT], ExtOnlyModel())
        d = obs.to_dict()
        del d["consumption_gain"]
        obs2 = Observer.from_dict(d, world_model=ExtOnlyModel())
        assert obs2.consumption_gain == 0.0
