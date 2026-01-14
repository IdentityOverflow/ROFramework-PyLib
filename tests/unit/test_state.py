"""Unit tests for Value and State implementations."""

import numpy as np
import pytest

from ro_framework.core.dof import CategoricalDoF, PolarDoF, ScalarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.core.value import Value


class TestValue:
    """Test Value implementation."""

    def test_creation(self) -> None:
        """Test creating a value."""
        dof = PolarDoF(name="position", pole_negative=-10.0, pole_positive=10.0)
        value = Value(dof=dof, value=5.0)

        assert value.dof == dof
        assert value.value == 5.0

    def test_invalid_value(self) -> None:
        """Test that invalid values raise ValueError."""
        dof = PolarDoF(
            name="bounded",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        with pytest.raises(ValueError, match="Invalid value"):
            Value(dof=dof, value=20.0)  # Outside domain

    def test_immutability(self) -> None:
        """Test that values are immutable."""
        dof = PolarDoF(name="test")
        value = Value(dof=dof, value=5.0)

        with pytest.raises(Exception):  # dataclass frozen
            value.value = 10.0  # type: ignore

    def test_distance_to(self) -> None:
        """Test distance computation between values."""
        dof = PolarDoF(name="position")
        v1 = Value(dof=dof, value=3.0)
        v2 = Value(dof=dof, value=7.0)

        assert v1.distance_to(v2) == 4.0
        assert v2.distance_to(v1) == 4.0

    def test_distance_different_dofs(self) -> None:
        """Test that distance between values on different DoFs raises error."""
        dof1 = PolarDoF(name="x")
        dof2 = PolarDoF(name="y")

        v1 = Value(dof=dof1, value=5.0)
        v2 = Value(dof=dof2, value=5.0)

        with pytest.raises(ValueError, match="different DoFs"):
            v1.distance_to(v2)

    def test_repr(self) -> None:
        """Test string representation."""
        dof = PolarDoF(name="test")
        value = Value(dof=dof, value=5.0)

        assert "test" in repr(value)
        assert "5.0" in repr(value)


class TestState:
    """Test State implementation."""

    def test_creation(self) -> None:
        """Test creating a state."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")

        state = State(values={x: 3.0, y: 4.0})

        assert state.get_value(x) == 3.0
        assert state.get_value(y) == 4.0

    def test_invalid_value_in_state(self) -> None:
        """Test that invalid values raise ValueError."""
        dof = PolarDoF(
            name="bounded",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        with pytest.raises(ValueError, match="Invalid value"):
            State(values={dof: 20.0})

    def test_get_value_missing(self) -> None:
        """Test getting value for DoF not in state."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")
        state = State(values={x: 3.0})

        assert state.get_value(x) == 3.0
        assert state.get_value(y) is None

    def test_set_value(self) -> None:
        """Test setting value (returns new state)."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")

        state1 = State(values={x: 3.0, y: 4.0})
        state2 = state1.set_value(x, 5.0)

        # Original unchanged
        assert state1.get_value(x) == 3.0

        # New state has updated value
        assert state2.get_value(x) == 5.0
        assert state2.get_value(y) == 4.0

    def test_project(self) -> None:
        """Test projecting state onto subset of DoFs."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")
        z = PolarDoF(name="z")

        state = State(values={x: 1.0, y: 2.0, z: 3.0})
        projected = state.project([x, y])

        assert projected.get_value(x) == 1.0
        assert projected.get_value(y) == 2.0
        assert projected.get_value(z) is None

    def test_distance_to(self) -> None:
        """Test distance computation between states."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")

        state1 = State(values={x: 0.0, y: 0.0})
        state2 = State(values={x: 3.0, y: 4.0})

        # Should compute Euclidean distance: sqrt(3^2 + 4^2) = 5
        distance = state1.distance_to(state2)
        assert distance == pytest.approx(5.0)

    def test_distance_to_subset(self) -> None:
        """Test distance computation on subset of DoFs."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")
        z = PolarDoF(name="z")

        state1 = State(values={x: 0.0, y: 0.0, z: 0.0})
        state2 = State(values={x: 3.0, y: 4.0, z: 100.0})

        # Distance considering only x and y
        distance = state1.distance_to(state2, dofs=[x, y])
        assert distance == pytest.approx(5.0)

    def test_to_vector_polar(self) -> None:
        """Test converting state to vector (polar DoFs)."""
        x = PolarDoF(
            name="x",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        y = PolarDoF(
            name="y",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        state = State(values={x: 0.0, y: 5.0})
        vector = state.to_vector([x, y])

        assert vector.shape == (2,)
        assert vector[0] == pytest.approx(0.0)  # Normalized 0 is 0
        # y=5 normalized to [-1,1] from [-10,10] should be 0.5
        assert vector[1] == pytest.approx(0.5)

    def test_to_vector_scalar(self) -> None:
        """Test converting state to vector (scalar DoFs)."""
        mass = ScalarDoF(name="mass", min_value=0.0, max_value=10.0)
        prob = ScalarDoF(name="probability", min_value=0.0, max_value=1.0)

        state = State(values={mass: 5.0, prob: 0.5})
        vector = state.to_vector([mass, prob])

        assert vector.shape == (2,)
        assert vector[0] == pytest.approx(0.5)  # 5/10
        assert vector[1] == pytest.approx(0.5)  # 0.5/1

    def test_to_vector_categorical(self) -> None:
        """Test converting state to vector (categorical DoFs)."""
        color = CategoricalDoF(name="color", categories={"red", "green", "blue"})

        state = State(values={color: "green"})
        vector = state.to_vector([color])

        # Should be one-hot encoded
        assert vector.shape == (3,)
        assert np.sum(vector) == pytest.approx(1.0)
        assert np.max(vector) == pytest.approx(1.0)

    def test_to_vector_missing_dof(self) -> None:
        """Test converting state with missing DoF."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")

        state = State(values={x: 3.0})  # y is missing
        vector = state.to_vector([x, y])

        assert vector.shape == (2,)
        assert vector[1] == pytest.approx(0.0)  # Missing value becomes 0

    def test_from_vector_polar(self) -> None:
        """Test reconstructing state from vector (polar DoFs)."""
        x = PolarDoF(
            name="x",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        y = PolarDoF(
            name="y",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        vector = np.array([0.0, 0.5])
        state = State.from_vector(vector, [x, y])

        assert state.get_value(x) == pytest.approx(0.0)
        assert state.get_value(y) == pytest.approx(5.0)

    def test_from_vector_categorical(self) -> None:
        """Test reconstructing state from vector (categorical DoFs)."""
        color = CategoricalDoF(name="color", categories={"red", "green", "blue"})

        # Create one-hot for "green"
        categories_sorted = sorted(color.categories)
        green_idx = categories_sorted.index("green")
        one_hot = np.zeros(3)
        one_hot[green_idx] = 1.0

        state = State.from_vector(one_hot, [color])

        assert state.get_value(color) == "green"

    def test_round_trip_conversion(self) -> None:
        """Test that to_vector and from_vector are inverses."""
        x = PolarDoF(
            name="x",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        y = PolarDoF(
            name="y",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        original_state = State(values={x: 3.5, y: -7.2})
        vector = original_state.to_vector([x, y])
        reconstructed_state = State.from_vector(vector, [x, y])

        assert reconstructed_state.get_value(x) == pytest.approx(3.5, abs=0.01)
        assert reconstructed_state.get_value(y) == pytest.approx(-7.2, abs=0.01)

    def test_repr(self) -> None:
        """Test string representation."""
        x = PolarDoF(name="x")
        state = State(values={x: 5.0})

        repr_str = repr(state)
        assert "State" in repr_str
        assert "x" in repr_str
        assert "5.0" in repr_str

    def test_equality(self) -> None:
        """Test state equality."""
        x = PolarDoF(name="x")
        y = PolarDoF(name="y")

        state1 = State(values={x: 3.0, y: 4.0})
        state2 = State(values={x: 3.0, y: 4.0})
        state3 = State(values={x: 3.0, y: 5.0})

        assert state1 == state2
        assert state1 != state3
