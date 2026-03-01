"""
Knowledge assessment: K(d_ext) = (ρ, ε, σ, C).

Implements the framework's graded knowledge definition (Section 4.4):
- ρ (correlation): How strongly an internal DoF tracks an external DoF
- ε (systematic_error): Consistent bias between internal and external
- σ (random_error): Noise / inconsistency in the mapping
- C (calibration): Whether stated uncertainty matches actual error
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ro_framework.core.dof import DoF

if TYPE_CHECKING:
    from ro_framework.observer.observer import ObservationLog


@dataclass(frozen=True)
class KnowledgeAssessment:
    """Graded knowledge of an external DoF: K(d_ext) = (ρ, ε, σ, C).

    Attributes:
        external_dof: The external DoF being assessed.
        best_internal_dof: The internal DoF most correlated with external_dof.
        correlation: ρ — absolute correlation strength [0, 1].
        systematic_error: ε — mean signed normalized residual (bias).
        random_error: σ — std of normalized residuals (noise).
        calibration: C — calibration quality [0, 1], higher is better.
        n_samples: Number of observation pairs used in computation.
    """

    external_dof: DoF
    best_internal_dof: Optional[DoF]
    correlation: float
    systematic_error: float
    random_error: float
    calibration: float
    n_samples: int
    contributing_dofs: Tuple[DoF, ...] = ()

    # Classification thresholds — override on the class to change defaults.
    THRESHOLDS: ClassVar[Dict[str, float]] = {
        "strong_correlation": 0.7,
        "strong_max_bias": 0.3,
        "strong_min_calibration": 0.5,
        "uncertain_max_correlation": 0.5,
        "uncertain_min_calibration": 0.5,
    }

    @property
    def knowledge_type(self) -> str:
        """Classify knowledge per framework Section 4.4.

        - "strong": High ρ, low ε, good C
        - "false": High ρ but high ε (correlated with wrong thing)
        - "uncertain": Low ρ but correctly calibrated
        - "weak": Everything else

        Thresholds are set via the class-level ``THRESHOLDS`` dict::

            KnowledgeAssessment.THRESHOLDS["strong_correlation"] = 0.8
        """
        t = self.THRESHOLDS
        if (self.correlation >= t["strong_correlation"]
                and abs(self.systematic_error) < t["strong_max_bias"]
                and self.calibration >= t["strong_min_calibration"]):
            return "strong"
        if (self.correlation >= t["strong_correlation"]
                and abs(self.systematic_error) >= t["strong_max_bias"]):
            return "false"
        if (self.correlation < t["uncertain_max_correlation"]
                and self.calibration >= t["uncertain_min_calibration"]):
            return "uncertain"
        return "weak"


def compute_knowledge(
    observation_log: "ObservationLog",
    external_dof: DoF,
    internal_dofs: List[DoF],
    min_samples: int = 10,
    max_features: int = 1,
) -> Optional[KnowledgeAssessment]:
    """Compute K(d_ext) from observation history.

    For each internal DoF, computes Pearson correlation with the external DoF.
    When ``max_features=1`` (default), selects the single best-correlated internal
    DoF and computes error metrics from it.  When ``max_features > 1``, uses
    multiple regression with the top-k features jointly, so ρ reflects the
    observer's *combined* knowledge of the external DoF.

    Algorithm (single-feature, max_features=1):
        1. Extract paired (external_value, internal_value) sequences from log
        2. For each internal DoF, compute |Pearson correlation| with external DoF
        3. Select the internal DoF with highest |correlation|
        4. Z-score both series, align sign for negative correlations
        5. Compute OLS regression residuals (predicted = ρ * ext_norm)
        6. σ = std(residuals) — regression noise
        7. ε = |Spearman(ext, |residuals|)| — heteroscedasticity
        8. C = 1 - CV(binned |residuals|) — residual consistency across range

    Algorithm (multi-feature, max_features > 1):
        1-2. Same: compute individual |Pearson ρ| for each internal DoF
        3. Select top-k features by |ρ| (k capped at n_samples // 10)
        4. Build design matrix X of z-scored feature values (n × k)
        5. OLS multiple regression: ext_norm = X @ β + residuals
        6-8. Same: σ, ε, C from residuals

    Args:
        observation_log: Paired observation history.
        external_dof: The external DoF to assess knowledge of.
        internal_dofs: List of internal DoFs to check correlation against.
        min_samples: Minimum observation pairs required.
        max_features: Maximum number of internal DoFs to use jointly.
            1 = single best feature (default, backward compatible).
            >1 = multiple regression with top-k features.

    Returns:
        KnowledgeAssessment, or None if insufficient data.
    """
    if len(observation_log) < min_samples:
        return None

    # Phase 1: compute individual correlations for all internal DoFs
    candidates: List[Tuple[DoF, float, List[float], List[float]]] = []

    for int_dof in internal_dofs:
        ext_vals, int_vals = observation_log.get_paired_values(external_dof, int_dof)

        if len(ext_vals) < min_samples:
            continue

        # Need variance in both series for meaningful correlation
        if np.std(ext_vals) < 1e-12 or np.std(int_vals) < 1e-12:
            continue

        corr, _ = pearsonr(ext_vals, int_vals)
        candidates.append((int_dof, float(corr), ext_vals, int_vals))

    if not candidates:
        return KnowledgeAssessment(
            external_dof=external_dof,
            best_internal_dof=None,
            correlation=0.0,
            systematic_error=0.0,
            random_error=1.0,
            calibration=0.0,
            n_samples=len(observation_log),
        )

    # Sort by |correlation| descending
    candidates.sort(key=lambda c: abs(c[1]), reverse=True)
    best_int_dof = candidates[0][0]

    # Phase 2: single-feature or multi-feature path
    if max_features <= 1 or len(candidates) == 1:
        return _single_feature_assessment(
            external_dof, candidates[0],
        )

    return _multi_feature_assessment(
        external_dof, candidates, max_features,
    )


def _single_feature_assessment(
    external_dof: DoF,
    candidate: Tuple[DoF, float, List[float], List[float]],
) -> KnowledgeAssessment:
    """Compute K from a single best-correlated internal DoF."""
    int_dof, corr, ext_vals, int_vals = candidate
    ext_arr = np.array(ext_vals)
    int_arr = np.array(int_vals)
    abs_corr = abs(corr)

    # Z-score normalize for scale-independent error computation
    ext_norm = _zscore(ext_arr)
    int_norm = _zscore(int_arr)

    # Align sign for negative correlations
    if corr < 0:
        int_norm = -int_norm

    # OLS regression residuals: predicted = ρ * ext_norm (for z-scored data)
    predicted = abs_corr * ext_norm
    residuals = int_norm - predicted

    systematic_error, random_error, calibration = _compute_error_metrics(
        ext_norm, residuals
    )

    return KnowledgeAssessment(
        external_dof=external_dof,
        best_internal_dof=int_dof,
        correlation=abs_corr,
        systematic_error=systematic_error,
        random_error=random_error,
        calibration=calibration,
        n_samples=len(ext_vals),
    )


def _multi_feature_assessment(
    external_dof: DoF,
    candidates: List[Tuple[DoF, float, List[float], List[float]]],
    max_features: int,
) -> KnowledgeAssessment:
    """Compute K using multiple regression with top-k features jointly."""
    # Use the ext_vals from the first candidate (all should be the same length
    # since they come from the same observation log paired with the same ext DoF)
    ext_vals = candidates[0][2]
    n = len(ext_vals)
    ext_arr = np.array(ext_vals)
    ext_norm = _zscore(ext_arr)

    # Cap k to avoid overfitting: at most n_samples // 10 features
    k = min(max_features, len(candidates), max(1, n // 10))

    top_candidates = candidates[:k]
    contributing_dofs = tuple(c[0] for c in top_candidates)

    # Build design matrix X (n × k) of z-scored feature values
    X = np.column_stack([
        _zscore(np.array(c[3])) * (1.0 if c[1] >= 0 else -1.0)
        for c in top_candidates
    ])

    # OLS multiple regression: ext_norm = X @ β + residuals
    beta, _, _, _ = np.linalg.lstsq(X, ext_norm, rcond=None)
    predicted = X @ beta
    residuals = ext_norm - predicted

    # Multiple correlation coefficient
    pred_corr, _ = pearsonr(ext_norm, predicted)
    abs_corr = abs(float(pred_corr))

    systematic_error, random_error, calibration = _compute_error_metrics(
        ext_norm, residuals
    )

    return KnowledgeAssessment(
        external_dof=external_dof,
        best_internal_dof=candidates[0][0],
        correlation=abs_corr,
        systematic_error=systematic_error,
        random_error=random_error,
        calibration=calibration,
        n_samples=n,
        contributing_dofs=contributing_dofs,
    )


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize an array. Returns zero-centered if std is negligible."""
    s = arr.std()
    if s > 1e-12:
        return (arr - arr.mean()) / s
    return arr - arr.mean()


def _compute_error_metrics(
    ext_norm: np.ndarray, residuals: np.ndarray
) -> Tuple[float, float, float]:
    """Compute (ε, σ, C) from regression residuals.

    Args:
        ext_norm: Z-scored external values.
        residuals: OLS regression residuals (single or multi-feature).

    Returns:
        (systematic_error, random_error, calibration) tuple.
    """
    # σ (random error): std of regression residuals
    random_error = float(np.std(residuals))

    # ε (systematic error / heteroscedasticity): does |residual| depend on position?
    # Guard: when residuals are negligibly small, Spearman on noise is meaningless.
    abs_residuals = np.abs(residuals)
    if random_error > 0.01 and np.std(abs_residuals) > 1e-10:
        sp_corr, _ = spearmanr(ext_norm, abs_residuals)
        systematic_error = abs(float(sp_corr))
    else:
        systematic_error = 0.0

    # C (calibration): residual consistency across range
    calibration = _compute_binned_calibration(ext_norm, residuals)

    return systematic_error, random_error, calibration


def _compute_binned_calibration(
    ext_norm: np.ndarray, residuals: np.ndarray
) -> float:
    """Compute calibration as residual consistency across the external range.

    Splits data into equal-frequency bins by external value, computes
    mean |residual| per bin, and returns 1 - CV (coefficient of variation).
    High calibration = errors are uniform across the range.

    Args:
        ext_norm: Z-scored external values.
        residuals: OLS regression residuals.

    Returns:
        Calibration score in [0, 1].
    """
    n = len(ext_norm)

    # Near-perfect fit: residuals are negligible noise, not meaningful error
    if np.std(residuals) < 0.01:
        return 1.0

    # Determine number of bins (at least 3 data points per bin)
    n_bins = max(3, min(10, n // 5))
    if n < n_bins * 3:
        n_bins = max(3, n // 3)

    # Sort and split into equal-frequency bins
    sort_idx = np.argsort(ext_norm)
    abs_res_sorted = np.abs(residuals[sort_idx])
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)

    bin_errors = []
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if end > start:
            bin_errors.append(float(np.mean(abs_res_sorted[start:end])))

    if len(bin_errors) < 2:
        return 1.0

    bin_errors = np.array(bin_errors)
    mean_err = bin_errors.mean()

    if mean_err < 1e-10:
        return 1.0

    # CV = coefficient of variation = std / mean
    cv = float(bin_errors.std() / mean_err)
    return float(np.clip(1.0 - cv, 0.0, 1.0))
