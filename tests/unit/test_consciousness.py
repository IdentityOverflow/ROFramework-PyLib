"""Unit tests for consciousness evaluation.

Tests ConsciousnessMetrics, ConsciousnessEvaluator (all four evaluation
methods with behavioral verification), and comparison/ranking utilities.
"""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import IdentityMapping, NeuralMapping
from ro_framework.consciousness.evaluation import (
    ConsciousnessEvaluator,
    ConsciousnessMetrics,
    _binned_ece,
    compare_observers,
    rank_by_consciousness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dof(name="d", lo=-1.0, hi=1.0):
    return PolarDoF(
        name=name,
        pole_negative=lo,
        pole_positive=hi,
        polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
    )


def _make_observer(*, self_model_fn=None, n_ext=1, n_int=1, name="obs"):
    """Build an Observer with optional self_model callable."""
    ext = [_make_dof(f"ext_{i}") for i in range(n_ext)]
    intl = [_make_dof(f"int_{i}", -5.0, 5.0) for i in range(n_int)]

    class WorldModel:
        def __call__(self, state: State) -> State:
            vals = {}
            for i, d in enumerate(intl):
                v = state.get_value(ext[i % len(ext)])
                vals[d] = float(v) if v is not None else 0.0
            return State(values=vals)

    self_model = None
    if self_model_fn is not None:
        self_model = self_model_fn(intl)

    return Observer(
        name=name,
        internal_dofs=intl,
        external_dofs=ext,
        world_model=WorldModel(),
        self_model=self_model,
    )


class _IdentitySelf:
    """Self-model that returns state unchanged (perfect self-knowledge)."""
    def __init__(self, dofs):
        self._dofs = dofs
    def __call__(self, state: State) -> State:
        return state


class _NoisySelf:
    """Self-model that adds noise (imperfect self-knowledge)."""
    def __init__(self, dofs, noise=2.0):
        self._dofs = dofs
        self._noise = noise
        self._rng = np.random.default_rng(0)
    def __call__(self, state: State) -> State:
        vals = {}
        for d in self._dofs:
            v = state.get_value(d)
            vals[d] = (float(v) if v is not None else 0.0) + self._rng.normal(0, self._noise)
        return State(values=vals)


class _ConstantSelf:
    """Self-model that always returns zeros (bad self-knowledge)."""
    def __init__(self, dofs):
        self._dofs = dofs
    def __call__(self, state: State) -> State:
        return State(values={d: 0.0 for d in self._dofs})


def _observe_many(obs, n=20, rng=None):
    """Feed n random external states through the observer."""
    if rng is None:
        rng = np.random.default_rng(42)
    for _ in range(n):
        vals = {d: float(rng.uniform(-1, 1)) for d in obs.external_dofs}
        obs.observe(State(values=vals))


# ---------------------------------------------------------------------------
# ConsciousnessMetrics
# ---------------------------------------------------------------------------

class TestConsciousnessMetrics:

    def test_metrics_creation(self):
        m = ConsciousnessMetrics(
            has_self_model=True, recursive_depth=1, self_accuracy=0.9,
            architectural_similarity=1.0, calibration_error=0.1,
            meta_cognitive_capability=0.8, limitation_awareness=0.7,
        )
        assert m.has_self_model
        assert m.recursive_depth == 1

    def test_consciousness_score_no_self_model(self):
        m = ConsciousnessMetrics(
            has_self_model=False, recursive_depth=0, self_accuracy=0.0,
            architectural_similarity=0.0, calibration_error=1.0,
            meta_cognitive_capability=0.0, limitation_awareness=0.0,
        )
        assert m.consciousness_score() == 0.0

    def test_consciousness_score_perfect(self):
        m = ConsciousnessMetrics(
            has_self_model=True, recursive_depth=3, self_accuracy=1.0,
            architectural_similarity=1.0, calibration_error=0.0,
            meta_cognitive_capability=1.0, limitation_awareness=1.0,
        )
        assert m.consciousness_score() > 0.8

    def test_consciousness_score_weighted(self):
        m = ConsciousnessMetrics(
            has_self_model=True, recursive_depth=1, self_accuracy=0.5,
            architectural_similarity=0.5, calibration_error=0.5,
            meta_cognitive_capability=0.5, limitation_awareness=0.5,
        )
        assert 0.0 < m.consciousness_score() < 1.0

    def test_to_dict(self):
        m = ConsciousnessMetrics(
            has_self_model=True, recursive_depth=1, self_accuracy=0.8,
            architectural_similarity=0.9, calibration_error=0.2,
            meta_cognitive_capability=0.7, limitation_awareness=0.6,
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["has_self_model"] is True
        assert isinstance(d["overall_score"], float)
        assert "recursive_depth" in d


# ---------------------------------------------------------------------------
# _binned_ece helper
# ---------------------------------------------------------------------------

class TestBinnedECE:

    def test_perfect_calibration(self):
        """Uncertainty == error everywhere → ECE ≈ 0."""
        vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ece = _binned_ece(vals, vals)
        assert ece < 0.05

    def test_bad_calibration(self):
        """Uncertainty far from error → high ECE."""
        unc = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        err = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        ece = _binned_ece(unc, err)
        assert ece > 0.3

    def test_too_few_samples(self):
        ece = _binned_ece(np.array([0.1]), np.array([0.1]))
        assert ece == 0.5  # fallback


# ---------------------------------------------------------------------------
# ConsciousnessEvaluator
# ---------------------------------------------------------------------------

class TestConsciousnessEvaluator:

    def test_creation(self):
        obs = _make_observer()
        evaluator = ConsciousnessEvaluator(obs)
        assert evaluator.observer is obs

    def test_evaluate_unconscious(self):
        obs = _make_observer()
        metrics = ConsciousnessEvaluator(obs).evaluate()
        assert not metrics.has_self_model
        assert metrics.recursive_depth == 0
        assert metrics.consciousness_score() == 0.0

    def test_evaluate_conscious(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        _observe_many(obs)
        metrics = ConsciousnessEvaluator(obs).evaluate()
        assert metrics.has_self_model
        assert metrics.recursive_depth >= 1
        assert metrics.consciousness_score() > 0.0

    def test_evaluate_with_test_states(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        ext = obs.external_dofs
        test_states = [State(values={d: float(i * 0.1) for d in ext}) for i in range(10)]
        metrics = ConsciousnessEvaluator(obs).evaluate(test_states)
        assert isinstance(metrics, ConsciousnessMetrics)
        assert 0.0 <= metrics.self_accuracy <= 1.0


class TestSelfAccuracy:

    def test_identity_self_model_high_accuracy(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        _observe_many(obs)
        acc = ConsciousnessEvaluator(obs)._evaluate_self_accuracy()
        assert acc > 0.9  # identity → near-perfect

    def test_constant_self_model_lower_accuracy(self):
        obs = _make_observer(self_model_fn=_ConstantSelf)
        _observe_many(obs)
        acc_bad = ConsciousnessEvaluator(obs)._evaluate_self_accuracy()

        obs2 = _make_observer(self_model_fn=_IdentitySelf)
        _observe_many(obs2)
        acc_good = ConsciousnessEvaluator(obs2)._evaluate_self_accuracy()

        assert acc_good > acc_bad

    def test_no_internal_state_returns_half(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        # No observations yet → internal_state is None
        acc = ConsciousnessEvaluator(obs)._evaluate_self_accuracy()
        assert acc == 0.5


class TestArchitecturalSimilarity:

    def test_same_type_high_similarity(self):
        dof = _make_dof()
        world = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        self_m = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        obs = Observer(
            name="t", internal_dofs=[dof], external_dofs=[dof],
            world_model=world, self_model=self_m,
        )
        sim = ConsciousnessEvaluator(obs)._evaluate_architectural_similarity()
        assert sim > 0.7  # same type → high

    def test_different_type_lower_similarity(self):
        dof = _make_dof()
        world = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

        class CustomSelf:
            def __call__(self, state):
                return state

        obs = Observer(
            name="t", internal_dofs=[dof], external_dofs=[dof],
            world_model=world, self_model=CustomSelf(),
        )
        sim = ConsciousnessEvaluator(obs)._evaluate_architectural_similarity()
        assert sim < 0.7  # different type → lower

    def test_neural_mappings_with_same_dims(self):
        dofs = [_make_dof(f"d{i}") for i in range(3)]
        world = NeuralMapping(name="w", input_dofs=dofs, output_dofs=dofs, model=None)
        self_m = NeuralMapping(name="s", input_dofs=dofs, output_dofs=dofs, model=None)
        obs = Observer(
            name="t", internal_dofs=dofs, external_dofs=dofs,
            world_model=world, self_model=self_m,
        )
        sim = ConsciousnessEvaluator(obs)._evaluate_architectural_similarity()
        # same type + same dims + same attrs → very high
        assert sim > 0.9

    def test_no_self_model(self):
        obs = _make_observer()
        sim = ConsciousnessEvaluator(obs)._evaluate_architectural_similarity()
        assert sim == 0.0


class TestCalibration:

    def test_identity_self_model_low_calibration_error(self):
        """Identity self-model → errors ≈ 0 → ECE should be low."""
        obs = _make_observer(self_model_fn=_IdentitySelf)
        _observe_many(obs, n=30)
        err = ConsciousnessEvaluator(obs)._evaluate_calibration()
        assert 0.0 <= err <= 1.0
        # Identity self-model: actual error is tiny, uncertainties are small
        # So ECE = |small_unc - ~0_err| — should be small
        assert err < 0.5

    def test_no_self_model_returns_one(self):
        obs = _make_observer()
        err = ConsciousnessEvaluator(obs)._evaluate_calibration()
        assert err == 1.0

    def test_insufficient_data_returns_half(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        # Only 1 observation — not enough
        obs.observe(State(values={obs.external_dofs[0]: 0.5}))
        err = ConsciousnessEvaluator(obs)._evaluate_calibration()
        assert err == 0.5

    def test_with_test_states(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        ext = obs.external_dofs
        test_states = [State(values={d: float(i * 0.1) for d in ext}) for i in range(10)]
        err = ConsciousnessEvaluator(obs)._evaluate_calibration(test_states)
        assert 0.0 <= err <= 1.0


class TestMetacognition:

    def test_identity_self_model_with_observations(self):
        """Identity self → high accuracy + stability → decent metacognition."""
        obs = _make_observer(self_model_fn=_IdentitySelf)
        _observe_many(obs, n=20)
        score = ConsciousnessEvaluator(obs)._evaluate_metacognition()
        assert score > 0.3  # accuracy and stability should be high

    def test_no_self_model(self):
        obs = _make_observer()
        score = ConsciousnessEvaluator(obs)._evaluate_metacognition()
        assert score == 0.0

    def test_depth_2_increases_score(self):
        """Nested self-model (depth 2) should boost metacognition score."""
        dof = _make_dof()
        world = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        inner = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        outer = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        outer.self_model = inner

        obs_d1 = Observer(
            name="d1", internal_dofs=[dof], external_dofs=[dof],
            world_model=world,
            self_model=IdentityMapping(input_dofs=[dof], output_dofs=[dof]),
        )
        obs_d2 = Observer(
            name="d2", internal_dofs=[dof], external_dofs=[dof],
            world_model=world, self_model=outer,
        )
        # Observe the same data
        for obs in (obs_d1, obs_d2):
            _observe_many(obs, n=10)

        s1 = ConsciousnessEvaluator(obs_d1)._evaluate_metacognition()
        s2 = ConsciousnessEvaluator(obs_d2)._evaluate_metacognition()
        assert s2 > s1  # depth 2 should score higher

    def test_noisy_self_model_lower_score(self):
        """Noisy self-model → lower accuracy → lower metacognition."""
        obs_good = _make_observer(self_model_fn=_IdentitySelf)
        obs_bad = _make_observer(self_model_fn=_NoisySelf)
        _observe_many(obs_good, n=20)
        _observe_many(obs_bad, n=20)
        s_good = ConsciousnessEvaluator(obs_good)._evaluate_metacognition()
        s_bad = ConsciousnessEvaluator(obs_bad)._evaluate_metacognition()
        assert s_good > s_bad


class TestLimitationAwareness:

    def test_no_self_model(self):
        obs = _make_observer()
        score = ConsciousnessEvaluator(obs)._evaluate_limitation_awareness()
        assert score == 0.0

    def test_insufficient_data_returns_half(self):
        obs = _make_observer(self_model_fn=_IdentitySelf)
        # No observations → empty log → can't split easy/hard
        score = ConsciousnessEvaluator(obs)._evaluate_limitation_awareness()
        assert score == 0.5

    def test_with_spread_test_states(self):
        """With a mix of central and extreme states, score should be in [0,1]."""
        obs = _make_observer(self_model_fn=_IdentitySelf)
        _observe_many(obs, n=10)
        ext = obs.external_dofs
        rng = np.random.default_rng(42)
        # Mix of central and extreme states
        test_states = (
            [State(values={d: float(rng.uniform(-0.1, 0.1)) for d in ext}) for _ in range(5)]
            + [State(values={d: float(rng.uniform(0.8, 1.0)) for d in ext}) for _ in range(5)]
        )
        score = ConsciousnessEvaluator(obs)._evaluate_limitation_awareness(test_states)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Comparison / ranking utilities
# ---------------------------------------------------------------------------

class TestComparisonFunctions:

    def test_compare_observers(self):
        obs1 = _make_observer(name="unconscious")
        obs2 = _make_observer(self_model_fn=_IdentitySelf, name="conscious")
        _observe_many(obs2, n=10)

        comparison = compare_observers([obs1, obs2])
        assert len(comparison) == 2
        assert comparison["conscious"].consciousness_score() > comparison["unconscious"].consciousness_score()

    def test_rank_by_consciousness(self):
        obs1 = _make_observer(name="obs1")
        obs2 = _make_observer(self_model_fn=_IdentitySelf, name="obs2")
        _observe_many(obs2, n=10)

        ranked = rank_by_consciousness([obs1, obs2])
        assert ranked[0][0].name == "obs2"
        assert ranked[0][1] >= ranked[1][1]

    def test_rank_empty_list(self):
        assert rank_by_consciousness([]) == []

    def test_rank_single_observer(self):
        obs = _make_observer(self_model_fn=_IdentitySelf, name="solo")
        ranked = rank_by_consciousness([obs])
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_evaluation_pipeline(self):
        obs = _make_observer(self_model_fn=_IdentitySelf, n_ext=3, n_int=3)
        _observe_many(obs, n=30)
        ext = obs.external_dofs
        test_states = [
            State(values={d: float(np.random.uniform(-1, 1)) for d in ext})
            for _ in range(20)
        ]
        metrics = ConsciousnessEvaluator(obs).evaluate(test_states)
        assert metrics.has_self_model
        for attr in ("self_accuracy", "architectural_similarity",
                     "calibration_error", "meta_cognitive_capability",
                     "limitation_awareness"):
            val = getattr(metrics, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of range"
        assert 0.0 <= metrics.consciousness_score() <= 1.0

    def test_better_self_model_scores_higher(self):
        obs_bad = _make_observer(self_model_fn=_ConstantSelf, name="bad")
        obs_good = _make_observer(self_model_fn=_IdentitySelf, name="good")
        _observe_many(obs_bad, n=20)
        _observe_many(obs_good, n=20)

        m_bad = ConsciousnessEvaluator(obs_bad).evaluate()
        m_good = ConsciousnessEvaluator(obs_good).evaluate()

        assert m_good.self_accuracy > m_bad.self_accuracy
        assert m_good.consciousness_score() > m_bad.consciousness_score()
