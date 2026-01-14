"""
Correlation measures for detecting structural relationships.

All correlation is computed relative to observer's accessible DoFs
and their induced measure structure.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

from ro_framework.core.dof import CategoricalDoF, DoF, PolarDoF, ScalarDoF
from ro_framework.core.state import State


class CorrelationMeasure:
    """
    Static methods for computing various correlation measures.

    Correlation measures detect structural relationships between DoFs
    across a collection of states (trajectory through DoF-space).
    """

    @staticmethod
    def extract_values(
        states: List[State], dof: DoF, handle_missing: str = "drop"
    ) -> List[float]:
        """
        Extract values for a DoF from a list of states.

        Args:
            states: List of states
            dof: DoF to extract values for
            handle_missing: How to handle missing values ('drop', 'zero', 'mean')

        Returns:
            List of values
        """
        values = []
        for state in states:
            val = state.get_value(dof)
            if val is not None:
                values.append(float(val))
            elif handle_missing == "zero":
                values.append(0.0)
            elif handle_missing == "drop":
                continue  # Will be handled later
            # 'mean' would require two-pass

        return values


def pearson_correlation(states: List[State], dof1: DoF, dof2: DoF) -> float:
    """
    Compute Pearson correlation coefficient between two DoFs.

    ρ(d₁, d₂) = Cov(d₁, d₂) / (σ(d₁) · σ(d₂))

    This measures linear correlation. Values range from -1 to 1:
    - 1: Perfect positive correlation
    - 0: No linear correlation
    - -1: Perfect negative correlation

    Args:
        states: List of states containing observations
        dof1: First DoF
        dof2: Second DoF

    Returns:
        Pearson correlation coefficient

    Example:
        >>> states = [State(values={x: i, y: 2*i}) for i in range(10)]
        >>> corr = pearson_correlation(states, x, y)
        >>> print(f"Correlation: {corr:.3f}")  # → 1.000 (perfect)
    """
    # Extract values
    values1, values2 = [], []

    for state in states:
        v1 = state.get_value(dof1)
        v2 = state.get_value(dof2)

        # Only include if both values present
        if v1 is not None and v2 is not None:
            values1.append(float(v1))
            values2.append(float(v2))

    if len(values1) < 2:
        return 0.0

    # Compute Pearson correlation
    corr, _ = pearsonr(values1, values2)

    return float(corr)


def mutual_information(
    states: List[State],
    dof1: DoF,
    dof2: DoF,
    bins: int = 10,
    normalize: bool = False,
) -> float:
    """
    Compute mutual information between two DoFs.

    I(d₁; d₂) = H(d₁) + H(d₂) - H(d₁, d₂)

    Where H is entropy. This measures how much knowing one DoF
    reduces uncertainty about the other. Unlike Pearson correlation,
    this captures non-linear relationships.

    Args:
        states: List of states containing observations
        dof1: First DoF
        dof2: Second DoF
        bins: Number of bins for discretization (for continuous DoFs)
        normalize: If True, normalize by min(H(d₁), H(d₂)) to get [0,1]

    Returns:
        Mutual information in nats (or normalized)

    Example:
        >>> # Non-linear relationship: y = x²
        >>> states = [State(values={x: i, y: i**2}) for i in range(-5, 6)]
        >>> mi = mutual_information(states, x, y)
        >>> print(f"MI: {mi:.3f}")  # High MI despite zero Pearson
    """
    # Extract values
    values1, values2 = [], []

    for state in states:
        v1 = state.get_value(dof1)
        v2 = state.get_value(dof2)

        if v1 is not None and v2 is not None:
            values1.append(v1)
            values2.append(v2)

    if len(values1) < 2:
        return 0.0

    # Discretize continuous values
    if isinstance(dof1, (PolarDoF, ScalarDoF)):
        discretizer1 = KBinsDiscretizer(n_bins=min(bins, len(set(values1))), encode="ordinal")
        values1 = discretizer1.fit_transform(np.array(values1).reshape(-1, 1)).ravel()

    if isinstance(dof2, (PolarDoF, ScalarDoF)):
        discretizer2 = KBinsDiscretizer(n_bins=min(bins, len(set(values2))), encode="ordinal")
        values2 = discretizer2.fit_transform(np.array(values2).reshape(-1, 1)).ravel()

    # Compute mutual information
    mi = mutual_info_score(values1, values2)

    if normalize:
        # Normalize by minimum entropy
        h1 = -np.sum(np.unique(values1, return_counts=True)[1] / len(values1) * np.log(np.unique(values1, return_counts=True)[1] / len(values1)))
        h2 = -np.sum(np.unique(values2, return_counts=True)[1] / len(values2) * np.log(np.unique(values2, return_counts=True)[1] / len(values2)))
        mi = mi / min(h1, h2) if min(h1, h2) > 0 else 0.0

    return float(mi)


def temporal_correlation(
    states: List[State], dof: DoF, temporal_dof: DoF, lag: int = 1
) -> float:
    """
    Compute auto-correlation across temporal DoF (for memory detection).

    Correlation(S_internal(t₁), S_internal(t₂))

    This measures how much a DoF at time t predicts itself at time t+lag.
    High temporal correlation indicates memory structure.

    Args:
        states: List of states (should be temporally ordered)
        dof: DoF to compute auto-correlation for
        temporal_dof: DoF representing time
        lag: Temporal lag (in units of temporal_dof)

    Returns:
        Auto-correlation coefficient

    Example:
        >>> # Sequence with memory: x_t = 0.9 * x_{t-1} + noise
        >>> states = generate_ar1_states(x, time, rho=0.9)
        >>> autocorr = temporal_correlation(states, x, time, lag=1)
        >>> print(f"Autocorr: {autocorr:.3f}")  # → ~0.9
    """
    # Sort states by temporal position
    sorted_states = sorted(
        states, key=lambda s: s.get_value(temporal_dof) if s.get_value(temporal_dof) else 0.0
    )

    # Extract values
    values = []
    for state in sorted_states:
        val = state.get_value(dof)
        if val is not None:
            values.append(float(val))

    if len(values) < lag + 2:
        return 0.0

    # Compute lagged correlation
    v1 = np.array(values[:-lag])
    v2 = np.array(values[lag:])

    if len(v1) < 2:
        return 0.0

    corr, _ = pearsonr(v1, v2)

    return float(corr)


def cross_correlation(
    states: List[State],
    dof1: DoF,
    dof2: DoF,
    temporal_dof: DoF,
    max_lag: int = 10,
) -> List[Tuple[int, float]]:
    """
    Compute cross-correlation at multiple lags.

    This can detect causal relationships: if dof1 at time t correlates
    with dof2 at time t+k (k>0), suggests dof1 may cause dof2.

    Args:
        states: List of states
        dof1: First DoF
        dof2: Second DoF
        temporal_dof: DoF representing time
        max_lag: Maximum lag to compute

    Returns:
        List of (lag, correlation) tuples

    Example:
        >>> # Causal: x causes y with 2-step delay
        >>> xcorr = cross_correlation(states, x, y, time, max_lag=5)
        >>> # Peak at lag=2 indicates causal relationship
    """
    # Sort states by time
    sorted_states = sorted(
        states, key=lambda s: s.get_value(temporal_dof) if s.get_value(temporal_dof) else 0.0
    )

    # Extract values
    values1, values2 = [], []
    for state in sorted_states:
        v1 = state.get_value(dof1)
        v2 = state.get_value(dof2)
        if v1 is not None and v2 is not None:
            values1.append(float(v1))
            values2.append(float(v2))

    if len(values1) < max_lag + 2:
        return [(0, 0.0)]

    # Compute cross-correlation at each lag
    cross_corrs = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Negative lag: dof2 leads dof1
            v1 = np.array(values1[: lag if lag != 0 else None])
            v2 = np.array(values2[-lag:])
        elif lag > 0:
            # Positive lag: dof1 leads dof2
            v1 = np.array(values1[lag:])
            v2 = np.array(values2[:-lag])
        else:
            # Zero lag
            v1 = np.array(values1)
            v2 = np.array(values2)

        if len(v1) < 2:
            cross_corrs.append((lag, 0.0))
            continue

        corr, _ = pearsonr(v1, v2)
        cross_corrs.append((lag, float(corr)))

    return cross_corrs


def detect_causality(
    states: List[State],
    cause_dof: DoF,
    effect_dof: DoF,
    temporal_dof: DoF,
    threshold: float = 0.5,
) -> bool:
    """
    Detect if cause_dof likely causes effect_dof.

    Uses cross-correlation to check if cause precedes effect.

    Args:
        states: List of states
        cause_dof: Potential cause DoF
        effect_dof: Potential effect DoF
        temporal_dof: DoF representing time
        threshold: Minimum correlation to consider causal

    Returns:
        True if causal relationship detected

    Example:
        >>> is_causal = detect_causality(states, sensor_input, motor_output, time)
        >>> if is_causal:
        ...     print("Sensor input causes motor output")
    """
    xcorr = cross_correlation(states, cause_dof, effect_dof, temporal_dof, max_lag=10)

    # Check for peak at positive lag (cause precedes effect)
    for lag, corr in xcorr:
        if lag > 0 and abs(corr) > threshold:
            return True

    return False
