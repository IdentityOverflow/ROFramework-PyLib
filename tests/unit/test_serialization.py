"""Unit tests for serialization: DoF, State, ObservationLog, Observer save/load."""

import json
import numpy as np
import pytest
from pathlib import Path

from ro_framework.core.dof import (
    CategoricalDoF,
    DerivedDoF,
    DoF,
    PolarDoF,
    PolarDoFType,
    ScalarDoF,
)
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer, ObservationLog, ObservationPair


# ---------------------------------------------------------------------------
# DoF serialization
# ---------------------------------------------------------------------------


class TestDoFSerialization:

    def test_polar_dof_roundtrip(self):
        dof = PolarDoF(
            name="temp",
            description="temperature",
            pole_negative=-100.0,
            pole_positive=100.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
            resolution=0.01,
        )
        d = dof.to_dict()
        restored = DoF.from_dict(d)

        assert isinstance(restored, PolarDoF)
        assert restored.name == "temp"
        assert restored.description == "temperature"
        assert restored.pole_negative == -100.0
        assert restored.pole_positive == 100.0
        assert restored.polar_type == PolarDoFType.CONTINUOUS_BOUNDED
        assert restored.resolution == 0.01

    def test_polar_dof_unbounded_roundtrip(self):
        dof = PolarDoF(name="x")
        d = dof.to_dict()
        restored = DoF.from_dict(d)

        assert isinstance(restored, PolarDoF)
        assert restored.pole_negative == -np.inf
        assert restored.pole_positive == np.inf
        assert restored.polar_type == PolarDoFType.CONTINUOUS_REAL

    def test_scalar_dof_roundtrip(self):
        dof = ScalarDoF(name="mass", min_value=0.0, max_value=1000.0, resolution=0.1)
        d = dof.to_dict()
        restored = DoF.from_dict(d)

        assert isinstance(restored, ScalarDoF)
        assert restored.name == "mass"
        assert restored.min_value == 0.0
        assert restored.max_value == 1000.0

    def test_categorical_dof_roundtrip(self):
        dof = CategoricalDoF(name="color", categories={"red", "green", "blue"})
        d = dof.to_dict()
        restored = DoF.from_dict(d)

        assert isinstance(restored, CategoricalDoF)
        assert restored.name == "color"
        assert restored.categories == {"red", "green", "blue"}

    def test_derived_dof_roundtrip(self):
        base = PolarDoF(name="pos")
        dof = DerivedDoF(name="vel", constituent_dofs=[base])
        d = dof.to_dict()
        restored = DoF.from_dict(d)

        assert isinstance(restored, DerivedDoF)
        assert restored.name == "vel"
        assert len(restored.constituent_dofs) == 1
        assert restored.constituent_dofs[0].name == "pos"
        assert restored.derivation_function is None  # callables not serialized

    def test_json_compatible(self):
        """to_dict output must be JSON-serializable."""
        dof = PolarDoF(name="x", pole_negative=-10.0, pole_positive=10.0)
        s = json.dumps(dof.to_dict())
        assert isinstance(s, str)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown DoF type"):
            DoF.from_dict({"type": "FooDoF", "name": "bad"})


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


class TestStateSerialization:

    def test_roundtrip(self):
        d1 = PolarDoF(name="x", pole_negative=-10.0, pole_positive=10.0)
        d2 = PolarDoF(name="y", pole_negative=-10.0, pole_positive=10.0)
        state = State(values={d1: 3.5, d2: -1.0})

        restored = State.from_dict(state.to_dict())

        assert restored.get_value(d1) == 3.5
        assert restored.get_value(d2) == -1.0

    def test_categorical_state_roundtrip(self):
        dof = CategoricalDoF(name="label", categories={"cat", "dog"})
        state = State(values={dof: "cat"})

        restored = State.from_dict(state.to_dict())
        # DoF reconstructed by name
        restored_dof = [d for d in restored.values if d.name == "label"][0]
        assert restored.get_value(restored_dof) == "cat"

    def test_json_compatible(self):
        dof = PolarDoF(name="x")
        state = State(values={dof: 1.0})
        s = json.dumps(state.to_dict())
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# ObservationPair / ObservationLog serialization
# ---------------------------------------------------------------------------


class TestObservationLogSerialization:

    def test_pair_roundtrip(self):
        ext_dof = PolarDoF(name="ext")
        int_dof = PolarDoF(name="int")
        pair = ObservationPair(
            external_state=State(values={ext_dof: 1.0}),
            internal_state=State(values={int_dof: 2.0}),
            timestamp=0.0,
        )
        restored = ObservationPair.from_dict(pair.to_dict())
        assert restored.timestamp == 0.0

    def test_log_roundtrip(self):
        ext_dof = PolarDoF(name="ext")
        int_dof = PolarDoF(name="int")
        log = ObservationLog(capacity=50)
        for i in range(10):
            log.append(ObservationPair(
                external_state=State(values={ext_dof: float(i)}),
                internal_state=State(values={int_dof: float(i * 2)}),
                timestamp=float(i),
            ))

        restored = ObservationLog.from_dict(log.to_dict())
        assert len(restored) == 10
        assert restored.capacity == 50


# ---------------------------------------------------------------------------
# Observer save / load
# ---------------------------------------------------------------------------


class TestObserverSerialization:

    def _make_observer_and_mapping(self):
        ext_dof = PolarDoF(name="sensor", pole_negative=-10.0, pole_positive=10.0)
        int_dof = PolarDoF(name="latent", pole_negative=-10.0, pole_positive=10.0)

        class WorldMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(ext_dof)
                return State(values={int_dof: val * 2.0 if val is not None else 0.0})

        mapping = WorldMapping()
        obs = Observer(
            name="test_obs",
            internal_dofs=[int_dof],
            external_dofs=[ext_dof],
            world_model=mapping,
            log_capacity=100,
        )
        return obs, mapping, ext_dof, int_dof

    def test_to_dict_and_from_dict(self):
        obs, mapping, ext_dof, int_dof = self._make_observer_and_mapping()

        rng = np.random.default_rng(42)
        for _ in range(20):
            obs.observe(State(values={ext_dof: float(rng.uniform(-5, 5))}))

        d = obs.to_dict()
        restored = Observer.from_dict(d, world_model=mapping)

        assert restored.name == "test_obs"
        assert len(restored.internal_dofs) == 1
        assert len(restored.external_dofs) == 1
        assert len(restored.observation_log) == 20
        assert restored.internal_dofs[0].name == "latent"

    def test_save_and_load(self, tmp_path):
        obs, mapping, ext_dof, _ = self._make_observer_and_mapping()

        rng = np.random.default_rng(42)
        for _ in range(15):
            obs.observe(State(values={ext_dof: float(rng.uniform(-5, 5))}))

        path = tmp_path / "observer.json"
        obs.save(path)

        assert path.exists()

        restored = Observer.load(path, world_model=mapping)
        assert restored.name == "test_obs"
        assert len(restored.observation_log) == 15

    def test_loaded_observer_can_observe(self, tmp_path):
        """After loading, the observer should work normally with the re-supplied model."""
        obs, mapping, ext_dof, int_dof = self._make_observer_and_mapping()

        for i in range(5):
            obs.observe(State(values={ext_dof: float(i)}))

        path = tmp_path / "obs.json"
        obs.save(path)

        restored = Observer.load(path, world_model=mapping)
        # Make new observations on the restored observer
        result = restored.observe(State(values={ext_dof: 7.0}))
        # World model doubles the input
        assert abs(result.get_value(int_dof) - 14.0) < 1e-6
        assert len(restored.observation_log) == 6  # 5 loaded + 1 new

    def test_resolution_preserved(self, tmp_path):
        ext_dof = PolarDoF(name="ext")
        int_dof = PolarDoF(name="int")

        class M:
            def __call__(self, state):
                return State(values={int_dof: 0.0})

        obs = Observer(
            name="res_test",
            internal_dofs=[int_dof],
            external_dofs=[ext_dof],
            world_model=M(),
            resolution={int_dof: 0.05},
        )

        path = tmp_path / "obs.json"
        obs.save(path)
        restored = Observer.load(path, world_model=M())

        assert restored.get_resolution(restored.internal_dofs[0]) == 0.05

    def test_json_file_is_valid(self, tmp_path):
        obs, _, ext_dof, _ = self._make_observer_and_mapping()
        obs.observe(State(values={ext_dof: 1.0}))

        path = tmp_path / "obs.json"
        obs.save(path)

        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "test_obs"
        assert "observation_log" in data
