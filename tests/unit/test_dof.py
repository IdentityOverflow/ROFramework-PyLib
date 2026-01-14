"""Unit tests for DoF implementations."""

import numpy as np
import pytest

from ro_framework.core.dof import (
    CategoricalDoF,
    DerivedDoF,
    PolarDoF,
    PolarDoFType,
    ScalarDoF,
)


class TestPolarDoF:
    """Test PolarDoF implementation."""

    def test_creation_bounded(self) -> None:
        """Test creating a bounded polar DoF."""
        dof = PolarDoF(
            name="position",
            description="Spatial position",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        assert dof.name == "position"
        assert dof.pole_negative == -10.0
        assert dof.pole_positive == 10.0
        assert dof.domain() == (-10.0, 10.0)

    def test_creation_unbounded(self) -> None:
        """Test creating an unbounded polar DoF."""
        dof = PolarDoF(
            name="real_axis",
            polar_type=PolarDoFType.CONTINUOUS_REAL,
        )

        assert dof.domain() == (-np.inf, np.inf)

    def test_invalid_poles(self) -> None:
        """Test that invalid poles raise ValueError."""
        with pytest.raises(ValueError, match="pole_negative.*must be < pole_positive"):
            PolarDoF(name="invalid", pole_negative=10.0, pole_positive=5.0)

    def test_validate_value(self) -> None:
        """Test value validation."""
        dof = PolarDoF(
            name="bounded",
            pole_negative=-1.0,
            pole_positive=1.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        assert dof.validate_value(0.0)
        assert dof.validate_value(-1.0)
        assert dof.validate_value(1.0)
        assert not dof.validate_value(2.0)
        assert not dof.validate_value(-2.0)

    def test_distance(self) -> None:
        """Test distance computation."""
        dof = PolarDoF(name="test")

        assert dof.distance(0.0, 5.0) == 5.0
        assert dof.distance(5.0, 0.0) == 5.0
        assert dof.distance(-3.0, 3.0) == 6.0

    def test_normalize_bounded(self) -> None:
        """Test normalization for bounded polar DoF."""
        dof = PolarDoF(
            name="bounded",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        assert dof.normalize(-10.0) == pytest.approx(-1.0)
        assert dof.normalize(0.0) == pytest.approx(0.0)
        assert dof.normalize(10.0) == pytest.approx(1.0)
        assert dof.normalize(5.0) == pytest.approx(0.5)

    def test_normalize_unbounded(self) -> None:
        """Test normalization for unbounded polar DoF."""
        dof = PolarDoF(name="unbounded", polar_type=PolarDoFType.CONTINUOUS_REAL)

        # Should use tanh
        assert -1.0 < dof.normalize(0.0) < 1.0
        assert dof.normalize(100.0) == pytest.approx(1.0, abs=0.01)
        assert dof.normalize(-100.0) == pytest.approx(-1.0, abs=0.01)

    def test_denormalize_bounded(self) -> None:
        """Test denormalization for bounded polar DoF."""
        dof = PolarDoF(
            name="bounded",
            pole_negative=-10.0,
            pole_positive=10.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )

        assert dof.denormalize(-1.0) == pytest.approx(-10.0)
        assert dof.denormalize(0.0) == pytest.approx(0.0)
        assert dof.denormalize(1.0) == pytest.approx(10.0)

    def test_gradient(self) -> None:
        """Test gradient computation."""
        dof = PolarDoF(name="test")

        assert dof.gradient(0.0, 5.0) == 5.0
        assert dof.gradient(5.0, 0.0) == -5.0
        assert dof.gradient(-3.0, 3.0) == 6.0

    def test_measure(self) -> None:
        """Test measure type."""
        dof = PolarDoF(name="test")
        assert dof.measure() == "lebesgue"

    def test_hashable(self) -> None:
        """Test that DoFs are hashable."""
        dof1 = PolarDoF(name="x")
        dof2 = PolarDoF(name="x")
        dof3 = PolarDoF(name="y")

        assert hash(dof1) == hash(dof2)
        assert hash(dof1) != hash(dof3)

        # Can be used in sets and dicts
        dof_set = {dof1, dof2, dof3}
        assert len(dof_set) == 2  # dof1 and dof2 are equal


class TestScalarDoF:
    """Test ScalarDoF implementation."""

    def test_creation(self) -> None:
        """Test creating a scalar DoF."""
        dof = ScalarDoF(
            name="mass",
            description="Rest mass",
            min_value=0.0,
            max_value=100.0,
        )

        assert dof.name == "mass"
        assert dof.min_value == 0.0
        assert dof.max_value == 100.0
        assert dof.domain() == (0.0, 100.0)

    def test_invalid_bounds(self) -> None:
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="min_value.*must be < max_value"):
            ScalarDoF(name="invalid", min_value=10.0, max_value=5.0)

    def test_validate_value(self) -> None:
        """Test value validation."""
        dof = ScalarDoF(name="probability", min_value=0.0, max_value=1.0)

        assert dof.validate_value(0.5)
        assert dof.validate_value(0.0)
        assert dof.validate_value(1.0)
        assert not dof.validate_value(-0.1)
        assert not dof.validate_value(1.1)

    def test_distance(self) -> None:
        """Test distance computation (non-directional)."""
        dof = ScalarDoF(name="test")

        assert dof.distance(0.0, 5.0) == 5.0
        assert dof.distance(5.0, 0.0) == 5.0  # Non-directional

    def test_normalize_bounded(self) -> None:
        """Test normalization for bounded scalar DoF."""
        dof = ScalarDoF(name="bounded", min_value=0.0, max_value=10.0)

        assert dof.normalize(0.0) == pytest.approx(0.0)
        assert dof.normalize(5.0) == pytest.approx(0.5)
        assert dof.normalize(10.0) == pytest.approx(1.0)

    def test_normalize_unbounded(self) -> None:
        """Test normalization for unbounded scalar DoF."""
        dof = ScalarDoF(name="unbounded", min_value=0.0, max_value=np.inf)

        # Should use sigmoid-like normalization
        assert 0.0 < dof.normalize(0.0) < 1.0
        assert dof.normalize(100.0) == pytest.approx(1.0, abs=0.01)

    def test_measure(self) -> None:
        """Test measure type."""
        dof = ScalarDoF(name="test")
        assert dof.measure() == "lebesgue"


class TestCategoricalDoF:
    """Test CategoricalDoF implementation."""

    def test_creation(self) -> None:
        """Test creating a categorical DoF."""
        categories = {"red", "green", "blue"}
        dof = CategoricalDoF(
            name="color",
            description="Color names",
            categories=categories,
        )

        assert dof.name == "color"
        assert dof.categories == categories
        assert dof.domain() == categories

    def test_empty_categories(self) -> None:
        """Test that empty categories raise ValueError."""
        with pytest.raises(ValueError, match="at least one category"):
            CategoricalDoF(name="invalid", categories=set())

    def test_uniform_weights(self) -> None:
        """Test that weights are initialized uniformly."""
        categories = {"a", "b", "c"}
        dof = CategoricalDoF(name="test", categories=categories)

        assert dof.weights is not None
        assert all(w == pytest.approx(1.0 / 3.0) for w in dof.weights.values())

    def test_custom_weights(self) -> None:
        """Test creating DoF with custom weights."""
        categories = {"a", "b"}
        weights = {"a": 0.7, "b": 0.3}
        dof = CategoricalDoF(name="test", categories=categories, weights=weights)

        assert dof.weights == weights

    def test_invalid_weights(self) -> None:
        """Test that invalid weights raise ValueError."""
        categories = {"a", "b"}

        # Weights don't sum to 1
        with pytest.raises(ValueError, match="sum to 1.0"):
            CategoricalDoF(name="test", categories=categories, weights={"a": 0.5, "b": 0.6})

        # Weights keys don't match categories
        with pytest.raises(ValueError, match="must match categories"):
            CategoricalDoF(name="test", categories=categories, weights={"a": 1.0})

    def test_validate_value(self) -> None:
        """Test value validation."""
        dof = CategoricalDoF(name="color", categories={"red", "green", "blue"})

        assert dof.validate_value("red")
        assert dof.validate_value("green")
        assert not dof.validate_value("yellow")

    def test_distance(self) -> None:
        """Test distance computation (binary)."""
        dof = CategoricalDoF(name="test", categories={"a", "b", "c"})

        assert dof.distance("a", "a") == 0.0
        assert dof.distance("a", "b") == 1.0
        assert dof.distance("b", "c") == 1.0

    def test_one_hot_encoding(self) -> None:
        """Test one-hot encoding."""
        dof = CategoricalDoF(name="color", categories={"red", "green", "blue"})

        one_hot = dof.to_one_hot("green")
        assert len(one_hot) == 3
        assert np.sum(one_hot) == 1.0
        assert np.max(one_hot) == 1.0

    def test_one_hot_invalid(self) -> None:
        """Test one-hot encoding with invalid value."""
        dof = CategoricalDoF(name="color", categories={"red", "green", "blue"})

        with pytest.raises(ValueError, match="not in categories"):
            dof.to_one_hot("yellow")

    def test_from_one_hot(self) -> None:
        """Test decoding one-hot to category."""
        dof = CategoricalDoF(name="color", categories={"red", "green", "blue"})

        # Encode and decode
        original = "green"
        one_hot = dof.to_one_hot(original)
        decoded = dof.from_one_hot(one_hot)

        assert decoded == original

    def test_measure(self) -> None:
        """Test measure type."""
        # Uniform weights
        dof1 = CategoricalDoF(name="test1", categories={"a", "b", "c"})
        assert dof1.measure() == "counting"

        # Non-uniform weights
        dof2 = CategoricalDoF(
            name="test2", categories={"a", "b"}, weights={"a": 0.7, "b": 0.3}
        )
        assert dof2.measure() == "weighted_counting"


class TestDerivedDoF:
    """Test DerivedDoF implementation."""

    def test_creation(self) -> None:
        """Test creating a derived DoF."""
        position = PolarDoF(name="position")
        time = PolarDoF(name="time")

        def compute_velocity(position: float, time: float) -> float:
            return position / time if time != 0 else 0.0

        dof = DerivedDoF(
            name="velocity",
            constituent_dofs=[position, time],
            derivation_function=compute_velocity,
        )

        assert dof.name == "velocity"
        assert len(dof.constituent_dofs) == 2

    def test_empty_constituents(self) -> None:
        """Test that empty constituents raise ValueError."""
        with pytest.raises(ValueError, match="at least one constituent"):
            DerivedDoF(name="invalid", constituent_dofs=[])

    def test_compute(self) -> None:
        """Test computing derived value."""
        position = PolarDoF(name="position")
        time = PolarDoF(name="time")

        def compute_velocity(position: float, time: float) -> float:
            return position / time

        dof = DerivedDoF(
            name="velocity",
            constituent_dofs=[position, time],
            derivation_function=compute_velocity,
        )

        result = dof.compute(position=10.0, time=2.0)
        assert result == 5.0

    def test_compute_missing_args(self) -> None:
        """Test that missing arguments raise ValueError."""
        position = PolarDoF(name="position")
        time = PolarDoF(name="time")

        dof = DerivedDoF(
            name="velocity",
            constituent_dofs=[position, time],
            derivation_function=lambda position, time: position / time,
        )

        with pytest.raises(ValueError, match="Missing values"):
            dof.compute(position=10.0)  # Missing 'time'

    def test_validate_value(self) -> None:
        """Test value validation."""
        dof = DerivedDoF(
            name="test", constituent_dofs=[PolarDoF(name="x")], derivation_function=lambda x: x
        )

        assert dof.validate_value(5.0)
        assert dof.validate_value(-5.0)
        assert not dof.validate_value(np.nan)

    def test_measure(self) -> None:
        """Test measure type."""
        dof = DerivedDoF(
            name="test", constituent_dofs=[PolarDoF(name="x")], derivation_function=lambda x: x
        )
        assert dof.measure() == "derived"
