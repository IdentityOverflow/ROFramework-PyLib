"""Unit tests for Observer and Mapping implementations."""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.mapping import ComposedMapping, IdentityMapping, NeuralMapping
from ro_framework.observer.observer import Observer


class TestIdentityMapping:
    """Test IdentityMapping implementation."""

    def test_creation(self) -> None:
        """Test creating an identity mapping."""
        dof = PolarDoF(name="test")
        mapping = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

        assert mapping.name == "identity"
        assert mapping.input_dofs == [dof]
        assert mapping.output_dofs == [dof]

    def test_mismatched_dofs(self) -> None:
        """Test that mismatched DoFs raise ValueError."""
        dof1 = PolarDoF(name="x")
        dof2 = PolarDoF(name="y")

        with pytest.raises(ValueError, match="input_dofs == output_dofs"):
            IdentityMapping(input_dofs=[dof1], output_dofs=[dof2])

    def test_call(self) -> None:
        """Test calling identity mapping."""
        dof = PolarDoF(name="test")
        mapping = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

        state = State(values={dof: 5.0})
        result = mapping(state)

        assert result == state
        assert result.get_value(dof) == 5.0


class TestComposedMapping:
    """Test ComposedMapping implementation."""

    def test_creation(self) -> None:
        """Test creating a composed mapping."""
        dof = PolarDoF(name="test")
        m1 = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        m2 = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

        composed = ComposedMapping(name="composed", mappings=[m1, m2])

        assert composed.name == "composed"
        assert len(composed.mappings) == 2

    def test_too_few_mappings(self) -> None:
        """Test that too few mappings raise ValueError."""
        dof = PolarDoF(name="test")
        m1 = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

        with pytest.raises(ValueError, match="at least 2 mappings"):
            ComposedMapping(name="invalid", mappings=[m1])

    def test_call(self) -> None:
        """Test calling composed mapping."""
        dof = PolarDoF(name="test")
        m1 = IdentityMapping(input_dofs=[dof], output_dofs=[dof])
        m2 = IdentityMapping(input_dofs=[dof], output_dofs=[dof])

        composed = ComposedMapping(name="composed", mappings=[m1, m2])

        state = State(values={dof: 5.0})
        result = composed(state)

        # Should be unchanged after two identity mappings
        assert result.get_value(dof) == 5.0


class TestNeuralMapping:
    """Test NeuralMapping implementation."""

    def test_creation(self) -> None:
        """Test creating a neural mapping."""
        input_dof = PolarDoF(name="input")
        output_dof = PolarDoF(name="output")

        mapping = NeuralMapping(
            name="test_mapping",
            input_dofs=[input_dof],
            output_dofs=[output_dof],
            model=None,  # Framework-agnostic
        )

        assert mapping.name == "test_mapping"
        assert mapping.input_dofs == [input_dof]
        assert mapping.output_dofs == [output_dof]

    def test_resolution_initialization(self) -> None:
        """Test that resolution dict is initialized."""
        input_dof = PolarDoF(name="input")
        output_dof = PolarDoF(name="output")

        mapping = NeuralMapping(
            name="test",
            input_dofs=[input_dof],
            output_dofs=[output_dof],
            model=None,
        )

        assert output_dof in mapping.resolution
        assert mapping.resolution[output_dof] == 1e-3

    def test_compute_uncertainty(self) -> None:
        """Test uncertainty computation."""
        input_dof = PolarDoF(name="input")
        output_dof = PolarDoF(name="output")

        mapping = NeuralMapping(
            name="test",
            input_dofs=[input_dof],
            output_dofs=[output_dof],
            model=None,
        )

        state = State(values={input_dof: 5.0})
        uncertainties = mapping.compute_uncertainty(state)

        assert output_dof in uncertainties
        assert uncertainties[output_dof] > 0


class TestObserver:
    """Test Observer implementation."""

    def test_creation(self) -> None:
        """Test creating an observer."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        # Use a simple callable mapping
        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof)
                return State(values={internal_dof: val if val is not None else 0.0})

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
        )

        assert observer.name == "test_observer"
        assert len(observer.internal_dofs) == 1
        assert len(observer.external_dofs) == 1

    def test_observe(self) -> None:
        """Test observation process."""
        external_dof = PolarDoF(name="sensor")
        internal_dof = PolarDoF(name="latent")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof)
                return State(values={internal_dof: val * 2 if val is not None else 0.0})

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
        )

        external_state = State(values={external_dof: 5.0})
        internal_state = observer.observe(external_state)

        assert internal_state.get_value(internal_dof) == 10.0
        assert observer.internal_state == internal_state

    def test_self_observe_without_model(self) -> None:
        """Test self-observation when no self-model exists."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof)
                return State(values={internal_dof: val if val is not None else 0.0})

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            self_model=None,
        )

        # Observe first to set internal state
        observer.observe(State(values={external_dof: 5.0}))

        # Self-observe should return None
        result = observer.self_observe()
        assert result is None

    def test_self_observe_with_model(self) -> None:
        """Test self-observation when self-model exists."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof if external_dof in state.values else internal_dof)
                return State(values={internal_dof: val if val is not None else 0.0})

        class SelfMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(internal_dof)
                # Self-model just copies internal state
                return State(values={internal_dof: val if val is not None else 0.0})

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            self_model=SelfMapping(),
        )

        # Observe first
        observer.observe(State(values={external_dof: 5.0}))

        # Self-observe
        self_repr = observer.self_observe()
        assert self_repr is not None
        assert self_repr.get_value(internal_dof) == 5.0

    def test_memory_storage(self) -> None:
        """Test that observations are stored in memory."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")
        temporal_dof = PolarDoF(name="time")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof)
                return State(values={internal_dof: val if val is not None else 0.0})

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            temporal_dof=temporal_dof,
        )

        # Make multiple observations
        for i in range(5):
            observer.observe(State(values={external_dof: float(i)}))

        assert len(observer.memory_buffer) == 5

    def test_memory_capacity(self) -> None:
        """Test that memory respects capacity limit."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")
        temporal_dof = PolarDoF(name="time")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof)
                return State(values={internal_dof: val if val is not None else 0.0})

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            temporal_dof=temporal_dof,
            memory_capacity=3,
        )

        # Make more observations than capacity
        for i in range(5):
            observer.observe(State(values={external_dof: float(i)}))

        # Should only keep last 3
        assert len(observer.memory_buffer) == 3

    def test_is_conscious(self) -> None:
        """Test consciousness detection."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                return state

        # Observer without self-model
        observer1 = Observer(
            name="unconscious",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            self_model=None,
        )
        assert not observer1.is_conscious()

        # Observer with self-model
        observer2 = Observer(
            name="conscious",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            self_model=SimpleMapping(),
        )
        assert observer2.is_conscious()

    def test_recursive_depth(self) -> None:
        """Test recursive depth computation."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                return state

        # No self-model: depth 0
        observer1 = Observer(
            name="depth0",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            self_model=None,
        )
        assert observer1.recursive_depth() == 0

        # With self-model: depth 1
        observer2 = Observer(
            name="depth1",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            self_model=SimpleMapping(),
        )
        assert observer2.recursive_depth() == 1

    def test_clear_memory(self) -> None:
        """Test clearing memory buffer."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")
        temporal_dof = PolarDoF(name="time")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(external_dof)
                return State(values={internal_dof: val if val is not None else 0.0})

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
            temporal_dof=temporal_dof,
        )

        # Add some observations
        for i in range(5):
            observer.observe(State(values={external_dof: float(i)}))

        assert len(observer.memory_buffer) > 0

        # Clear memory
        observer.clear_memory()
        assert len(observer.memory_buffer) == 0

    def test_get_memory_size(self) -> None:
        """Test getting memory size."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                return state

        observer = Observer(
            name="test",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
        )

        assert observer.get_memory_size() == 0

    def test_repr(self) -> None:
        """Test string representation."""
        external_dof = PolarDoF(name="external")
        internal_dof = PolarDoF(name="internal")

        class SimpleMapping:
            def __call__(self, state: State) -> State:
                return state

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[external_dof],
            world_model=SimpleMapping(),
        )

        repr_str = repr(observer)
        assert "test_observer" in repr_str
        assert "Observer" in repr_str
