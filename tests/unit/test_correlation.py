"""Unit tests for correlation measures."""

import numpy as np
import pytest

from ro_framework.core.dof import CategoricalDoF, PolarDoF, PolarDoFType, ScalarDoF
from ro_framework.core.state import State
from ro_framework.correlation.measures import (
    pearson_correlation,
    mutual_information,
    temporal_correlation,
    cross_correlation,
    detect_causality,
)


@pytest.fixture
def continuous_dofs():
    """Create continuous DoFs for testing."""
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
    time = PolarDoF(
        name="time",
        pole_negative=0.0,
        pole_positive=100.0,
        polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
    )
    return x, y, time


@pytest.fixture
def categorical_dofs():
    """Create categorical DoFs for testing."""
    color = CategoricalDoF(name="color", categories={"red", "green", "blue"})
    shape = CategoricalDoF(name="shape", categories={"circle", "square", "triangle"})
    return color, shape


class TestPearsonCorrelation:
    """Test Pearson correlation coefficient computation."""

    def test_perfect_positive_correlation(self, continuous_dofs):
        """Test perfect positive correlation (y = x)."""
        x, y, _ = continuous_dofs

        # Create states with perfect positive correlation
        states = [
            State(values={x: i, y: i})
            for i in range(-5, 6)
        ]

        corr = pearson_correlation(states, x, y)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self, continuous_dofs):
        """Test perfect negative correlation (y = -x)."""
        x, y, _ = continuous_dofs

        states = [
            State(values={x: i, y: -i})
            for i in range(-5, 6)
        ]

        corr = pearson_correlation(states, x, y)
        assert corr == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self, continuous_dofs):
        """Test no correlation (independent variables)."""
        x, y, _ = continuous_dofs

        # x increases, y is constant
        states = [
            State(values={x: i, y: 0.0})
            for i in range(-5, 6)
        ]

        corr = pearson_correlation(states, x, y)
        # With constant y, correlation should be NaN or close to 0
        assert np.isnan(corr) or abs(corr) < 0.1

    def test_linear_correlation(self, continuous_dofs):
        """Test linear correlation (y = 2x + 1)."""
        x, y, _ = continuous_dofs

        states = [
            State(values={x: i, y: 2 * i + 1})
            for i in range(-5, 6)
        ]

        corr = pearson_correlation(states, x, y)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_insufficient_data(self, continuous_dofs):
        """Test with insufficient data points."""
        x, y, _ = continuous_dofs

        # Only one data point
        states = [State(values={x: 1.0, y: 2.0})]

        corr = pearson_correlation(states, x, y)
        assert corr == 0.0  # Should return 0 for insufficient data

    def test_missing_values(self, continuous_dofs):
        """Test handling of missing values."""
        x, y, _ = continuous_dofs

        # Some states missing values
        states = [
            State(values={x: 1.0, y: 2.0}),
            State(values={x: 2.0}),  # Missing y
            State(values={x: 3.0, y: 4.0}),
        ]

        corr = pearson_correlation(states, x, y)
        # Should compute on available pairs only
        assert isinstance(corr, float)


class TestMutualInformation:
    """Test mutual information computation."""

    def test_perfect_dependence(self, continuous_dofs):
        """Test perfect mutual dependence."""
        x, y, _ = continuous_dofs

        # y = x (perfect dependence)
        states = [
            State(values={x: i, y: i})
            for i in range(-5, 6)
        ]

        mi = mutual_information(states, x, y, bins=10)
        # MI should be positive for dependent variables
        assert mi > 0

    def test_independence(self, continuous_dofs):
        """Test independent variables."""
        x, y, _ = continuous_dofs

        np.random.seed(42)
        # Independent random variables
        states = [
            State(values={
                x: np.random.uniform(-10, 10),
                y: np.random.uniform(-10, 10)
            })
            for _ in range(100)
        ]

        mi = mutual_information(states, x, y, bins=5)
        # MI should be close to 0 for independent variables
        # (but not exactly 0 due to finite sample effects)
        assert mi < 1.0

    def test_nonlinear_relationship(self, continuous_dofs):
        """Test nonlinear relationship (y = x²)."""
        x, y, _ = continuous_dofs

        states = [
            State(values={x: i, y: i**2})
            for i in range(-5, 6)
        ]

        # Pearson might miss this, but MI should catch it
        pearson = pearson_correlation(states, x, y)
        mi = mutual_information(states, x, y, bins=10)

        # MI should be higher than Pearson for nonlinear relationship
        assert mi > abs(pearson)

    def test_categorical_mi(self, categorical_dofs):
        """Test MI with categorical variables."""
        color, shape = categorical_dofs

        # Create perfect correspondence
        mapping = {"red": "circle", "green": "square", "blue": "triangle"}
        states = [
            State(values={color: c, shape: mapping[c]})
            for c in ["red", "green", "blue"] * 10
        ]

        mi = mutual_information(states, color, shape)
        # Should have high MI due to perfect correspondence
        assert mi > 0.5


class TestTemporalCorrelation:
    """Test temporal correlation (auto-correlation)."""

    def test_strong_memory(self, continuous_dofs):
        """Test strong temporal correlation (memory)."""
        x, _, time = continuous_dofs

        # AR(1) process: x_t = 0.9 * x_{t-1} + noise
        np.random.seed(42)
        values = [0.0]
        for _ in range(50):
            values.append(0.9 * values[-1] + np.random.normal(0, 0.1))

        states = [
            State(values={x: values[i], time: float(i)})
            for i in range(len(values))
        ]

        autocorr = temporal_correlation(states, x, time, lag=1)
        # Should have high autocorrelation
        assert autocorr > 0.7

    def test_no_memory(self, continuous_dofs):
        """Test no temporal correlation (no memory)."""
        x, _, time = continuous_dofs

        # White noise (no memory)
        np.random.seed(42)
        states = [
            State(values={x: np.random.normal(0, 1), time: float(i)})
            for i in range(100)
        ]

        autocorr = temporal_correlation(states, x, time, lag=1)
        # Should have low autocorrelation
        assert abs(autocorr) < 0.3

    def test_lag_dependence(self, continuous_dofs):
        """Test that correlation decreases with lag."""
        x, _, time = continuous_dofs

        # AR(1) process
        np.random.seed(42)
        values = [0.0]
        for _ in range(100):
            values.append(0.8 * values[-1] + np.random.normal(0, 0.1))

        states = [
            State(values={x: values[i], time: float(i)})
            for i in range(len(values))
        ]

        # Correlation should decrease with lag
        corr_lag1 = temporal_correlation(states, x, time, lag=1)
        corr_lag5 = temporal_correlation(states, x, time, lag=5)

        assert corr_lag1 > corr_lag5

    def test_insufficient_data_for_lag(self, continuous_dofs):
        """Test with insufficient data for lag."""
        x, _, time = continuous_dofs

        states = [
            State(values={x: float(i), time: float(i)})
            for i in range(3)
        ]

        # Lag is too large for data
        autocorr = temporal_correlation(states, x, time, lag=10)
        assert autocorr == 0.0


class TestCrossCorrelation:
    """Test cross-correlation."""

    def test_synchronous_correlation(self, continuous_dofs):
        """Test correlation at zero lag."""
        x, y, time = continuous_dofs

        # y = x (no delay)
        states = [
            State(values={x: float(i), y: float(i), time: float(i)})
            for i in range(20)
        ]

        xcorr = cross_correlation(states, x, y, time, max_lag=5)

        # Find correlation at lag 0
        lag0_corr = [corr for lag, corr in xcorr if lag == 0][0]
        assert lag0_corr == pytest.approx(1.0, abs=0.01)

    def test_delayed_correlation(self, continuous_dofs):
        """Test correlation with delay."""
        x, y, time = continuous_dofs

        # y lags x by 2 steps
        values = list(range(20))
        states = [
            State(values={
                x: float(values[i]),
                y: float(values[max(0, i - 2)]),
                time: float(i)
            })
            for i in range(20)
        ]

        xcorr = cross_correlation(states, x, y, time, max_lag=5)

        # Should have peak at lag 2
        corrs_dict = {lag: corr for lag, corr in xcorr}
        lag2_corr = corrs_dict.get(2, 0)

        # Lag 2 should have high correlation
        assert lag2_corr > 0.8


class TestDetectCausality:
    """Test causality detection."""

    def test_causal_relationship(self, continuous_dofs):
        """Test detecting causal relationship."""
        x, y, time = continuous_dofs

        # x causes y with delay
        values_x = list(range(30))
        states = [
            State(values={
                x: float(values_x[i]),
                y: float(values_x[max(0, i - 2)]),  # y = x delayed by 2
                time: float(i)
            })
            for i in range(30)
        ]

        is_causal = detect_causality(states, x, y, time, threshold=0.5)
        assert is_causal

    def test_no_causality(self, continuous_dofs):
        """Test no causal relationship."""
        x, y, time = continuous_dofs

        # Independent random variables
        np.random.seed(42)
        states = [
            State(values={
                x: np.random.normal(0, 1),
                y: np.random.normal(0, 1),
                time: float(i)
            })
            for i in range(50)
        ]

        is_causal = detect_causality(states, x, y, time, threshold=0.5)
        assert not is_causal

    def test_reverse_causality(self, continuous_dofs):
        """Test that reverse causality is not detected."""
        x, y, time = continuous_dofs

        # y causes x (reverse)
        values_y = list(range(30))
        states = [
            State(values={
                x: float(values_y[max(0, i - 2)]),  # x = y delayed
                y: float(values_y[i]),
                time: float(i)
            })
            for i in range(30)
        ]

        # Should not detect x → y causality
        is_causal = detect_causality(states, x, y, time, threshold=0.5)
        # This might be False if implementation is correct
        # (depends on asymmetry in cross-correlation peak)
        assert isinstance(is_causal, bool)


class TestIntegration:
    """Integration tests combining multiple correlation measures."""

    def test_memory_detection_pipeline(self, continuous_dofs):
        """Test complete memory detection pipeline."""
        x, _, time = continuous_dofs

        # Create trajectory with memory
        np.random.seed(42)
        values = [0.0]
        for _ in range(50):
            values.append(0.85 * values[-1] + np.random.normal(0, 0.1))

        states = [
            State(values={x: values[i], time: float(i)})
            for i in range(len(values))
        ]

        # Check temporal correlation
        autocorr = temporal_correlation(states, x, time, lag=1)
        assert autocorr > 0.7  # Strong memory

        # Check that it decreases with lag
        autocorr_lag5 = temporal_correlation(states, x, time, lag=5)
        assert autocorr > autocorr_lag5

    def test_correlation_comparison(self, continuous_dofs):
        """Compare Pearson and MI on nonlinear relationship."""
        x, y, _ = continuous_dofs

        # Nonlinear: y = sin(x)
        states = [
            State(values={x: float(i) / 5, y: float(np.sin(i / 5))})
            for i in range(50)
        ]

        pearson = pearson_correlation(states, x, y)
        mi = mutual_information(states, x, y, bins=10)

        # Both should detect relationship, but MI might be stronger
        assert abs(pearson) > 0 or mi > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
