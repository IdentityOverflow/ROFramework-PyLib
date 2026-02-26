"""
Knowledge assessment: K(d_ext) = (ρ, ε, σ, C).

Implements the framework's graded knowledge definition (Section 4.4):
- ρ (correlation): How strongly an internal DoF tracks an external DoF
- ε (systematic_error): Consistent bias between internal and external
- σ (random_error): Noise / inconsistency in the mapping
- C (calibration): Whether stated uncertainty matches actual error
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr

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
) -> Optional[KnowledgeAssessment]:
    """Compute K(d_ext) from observation history.

    For each internal DoF, computes Pearson correlation with the external DoF.
    Selects the best-correlated internal DoF and computes error metrics.

    Algorithm:
        1. Extract paired (external_value, internal_value) sequences from log
        2. For each internal DoF, compute |Pearson correlation| with external DoF
        3. Select the internal DoF with highest |correlation|
        4. Normalize both series to z-scores for scale-independent error
        5. Compute ε = mean(residuals), σ = std(residuals)
        6. Compute C = calibration from residual consistency

    Args:
        observation_log: Paired observation history.
        external_dof: The external DoF to assess knowledge of.
        internal_dofs: List of internal DoFs to check correlation against.
        min_samples: Minimum observation pairs required.

    Returns:
        KnowledgeAssessment, or None if insufficient data.
    """
    if len(observation_log) < min_samples:
        return None

    best_corr = 0.0
    best_int_dof: Optional[DoF] = None
    best_ext_vals: Optional[List[float]] = None
    best_int_vals: Optional[List[float]] = None

    for int_dof in internal_dofs:
        ext_vals, int_vals = observation_log.get_paired_values(external_dof, int_dof)

        if len(ext_vals) < min_samples:
            continue

        # Need variance in both series for meaningful correlation
        if np.std(ext_vals) < 1e-12 or np.std(int_vals) < 1e-12:
            continue

        corr, _ = pearsonr(ext_vals, int_vals)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_int_dof = int_dof
            best_ext_vals = ext_vals
            best_int_vals = int_vals

    if best_int_dof is None or best_ext_vals is None or best_int_vals is None:
        return KnowledgeAssessment(
            external_dof=external_dof,
            best_internal_dof=None,
            correlation=0.0,
            systematic_error=0.0,
            random_error=1.0,
            calibration=0.0,
            n_samples=len(observation_log),
        )

    ext_arr = np.array(best_ext_vals)
    int_arr = np.array(best_int_vals)

    # Z-score normalize for scale-independent error computation
    ext_std = ext_arr.std()
    int_std = int_arr.std()
    ext_norm = (ext_arr - ext_arr.mean()) / ext_std if ext_std > 1e-12 else ext_arr - ext_arr.mean()
    int_norm = (int_arr - int_arr.mean()) / int_std if int_std > 1e-12 else int_arr - int_arr.mean()

    # Align sign: if the best match is negatively correlated (inverse mapping),
    # flip internal values so residuals measure genuine bias, not sign mismatch.
    if best_corr < 0:
        int_norm = -int_norm

    residuals = int_norm - ext_norm
    systematic_error = float(np.mean(residuals))
    random_error = float(np.std(residuals))

    # Calibration: how consistent is the residual pattern?
    # Perfect calibration → residuals are zero-mean with low spread.
    # C = 1 - (|ε| + σ/2), clipped to [0, 1].
    calibration = float(np.clip(1.0 - abs(systematic_error) - random_error * 0.5, 0.0, 1.0))

    return KnowledgeAssessment(
        external_dof=external_dof,
        best_internal_dof=best_int_dof,
        correlation=abs(float(best_corr)),
        systematic_error=systematic_error,
        random_error=random_error,
        calibration=calibration,
        n_samples=len(best_ext_vals),
    )
