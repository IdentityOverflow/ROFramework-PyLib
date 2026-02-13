"""
Observer implementation.

An observer is a configuration within the Block Universe characterized by:
- Boundary (internal/external DoF partition)
- Mapping functions (external -> internal)
- Resolution (per-DoF finite granularity)
- Memory (correlation structure across temporal DoF via ObservationLog)

O = (B, M, R, Mem)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ro_framework.core.dof import DoF
from ro_framework.core.state import State
from ro_framework.observer.mapping import MappingFunction


@dataclass(frozen=True)
class ObservationPair:
    """A paired (external, internal) observation record.

    Represents a single observation event where the observer
    mapped an external state to an internal state.
    """

    external_state: State
    internal_state: State
    timestamp: float


class ObservationLog:
    """Paired observation history with capacity limits.

    Stores (external, internal) pairs from each observe() call.
    Provides access methods for both knowledge assessment and
    temporal memory analysis.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._pairs: List[ObservationPair] = []

    def append(self, pair: ObservationPair) -> None:
        """Append an observation pair, evicting oldest if at capacity."""
        self._pairs.append(pair)
        if len(self._pairs) > self.capacity:
            self._pairs.pop(0)

    def get_internal_states(self) -> List[State]:
        """Extract just the internal states (for temporal memory analysis)."""
        return [p.internal_state for p in self._pairs]

    def get_paired_values(
        self, external_dof: DoF, internal_dof: DoF
    ) -> tuple:
        """Extract paired (external, internal) value sequences for a DoF pair.

        Returns:
            (ext_values, int_values) — two aligned lists of floats.
            Pairs where either value is None are skipped.
        """
        ext_vals: List[float] = []
        int_vals: List[float] = []
        for pair in self._pairs:
            e = pair.external_state.get_value(external_dof)
            i = pair.internal_state.get_value(internal_dof)
            if e is not None and i is not None:
                ext_vals.append(float(e))
                int_vals.append(float(i))
        return ext_vals, int_vals

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs)

    def clear(self) -> None:
        self._pairs.clear()


@dataclass
class Observer:
    """Observer: O = (B, M, R, Mem).

    A configuration within the Block that maps external DoFs to internal DoFs
    with finite resolution, maintaining an observation history.

    Attributes:
        name: Identifier for this observer.
        internal_dofs: DoFs internal to the observer (boundary: inside).
        external_dofs: DoFs external to the observer (boundary: outside).
        world_model: Mapping from external to internal DoFs.
        self_model: Optional mapping from internal to internal (consciousness).
        resolution: Per-DoF resolution limits.
        temporal_dof: Optional temporal DoF for memory analysis.
        observation_log: Paired observation history.
        internal_state: Current internal state.
    """

    name: str
    internal_dofs: List[DoF]
    external_dofs: List[DoF]
    world_model: MappingFunction
    self_model: Optional[MappingFunction] = None
    resolution: Dict[DoF, float] = field(default_factory=dict)
    temporal_dof: Optional[DoF] = None
    log_capacity: int = 1000
    internal_state: Optional[State] = None

    # Non-init field, created in __post_init__
    observation_log: ObservationLog = field(init=False)

    def __post_init__(self) -> None:
        """Initialize resolution defaults and observation log."""
        if not self.resolution:
            self.resolution = {dof: 1e-6 for dof in self.internal_dofs}
        self.observation_log = ObservationLog(capacity=self.log_capacity)

    # ------------------------------------------------------------------
    # Core observation
    # ------------------------------------------------------------------

    def observe(self, external_state: State) -> State:
        """Map external DoFs to internal DoFs via world model.

        Records the (external, internal) pair in the observation log.

        Args:
            external_state: State with values on external DoFs.

        Returns:
            Internal state with values on internal DoFs.
        """
        internal_state = self.world_model(external_state)
        self.internal_state = internal_state

        self.observation_log.append(ObservationPair(
            external_state=external_state,
            internal_state=internal_state,
            timestamp=float(len(self.observation_log)),
        ))

        return internal_state

    def self_observe(self) -> Optional[State]:
        """Recursive self-modeling: map internal DoFs to internal DoFs.

        This is the structural definition of consciousness —
        internal->internal correlation with the same architectural type
        as external->internal correlation.

        Returns:
            Self-representation state, or None if no self-model.
        """
        if self.self_model is None or self.internal_state is None:
            return None
        return self.self_model(self.internal_state)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def get_resolution(self, dof: DoF) -> float:
        """Get resolution limit for a specific DoF."""
        return self.resolution.get(dof, 1e-6)

    # ------------------------------------------------------------------
    # Knowledge assessment — K(d_ext) = (ρ, ε, σ, C)
    # ------------------------------------------------------------------

    def assess_knowledge(self, external_dof: DoF, min_samples: int = 10):
        """Compute graded knowledge of an external DoF.

        Returns KnowledgeAssessment with correlation, bias, noise,
        and calibration, or None if insufficient observation history.

        Args:
            external_dof: The external DoF to assess knowledge of.
            min_samples: Minimum observations required.

        Returns:
            KnowledgeAssessment or None.
        """
        from ro_framework.knowledge.assessment import compute_knowledge

        return compute_knowledge(
            self.observation_log, external_dof, self.internal_dofs, min_samples
        )

    def know(
        self,
        external_dof: DoF,
        threshold: float = 0.7,
        min_samples: int = 10,
    ) -> bool:
        """Check if observer has knowledge of an external DoF.

        Knowledge requires high correlation and reasonable calibration.

        Args:
            external_dof: External DoF to check.
            threshold: Minimum correlation for knowledge.
            min_samples: Minimum observations required.

        Returns:
            True if knowledge criteria are met.
        """
        assessment = self.assess_knowledge(external_dof, min_samples)
        if assessment is None:
            return False
        return assessment.correlation >= threshold and assessment.calibration >= 0.4

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def estimate_uncertainty(self, dof: DoF) -> float:
        """Estimate uncertainty for a DoF using quadrature addition.

        Combines three independent uncertainty sources:
        - Resolution limits (structural floor)
        - Model uncertainty (from mapping, if available)
        - Empirical uncertainty (variance from recent observations)

        Args:
            dof: DoF to estimate uncertainty for.

        Returns:
            Combined uncertainty estimate.
        """
        if self.internal_state is None:
            return 1.0

        resolution_unc = self.get_resolution(dof)

        # Model uncertainty
        model_unc = 0.0
        if hasattr(self.world_model, "compute_uncertainty"):
            model_unc_dict = self.world_model.compute_uncertainty(self.internal_state)
            model_unc = model_unc_dict.get(dof, 0.0)

        # Empirical uncertainty from recent observations
        empirical_unc = 0.0
        internal_states = self.observation_log.get_internal_states()
        if len(internal_states) >= 5:
            recent = internal_states[-50:]
            values = [s.get_value(dof) for s in recent if s.get_value(dof) is not None]
            if len(values) >= 2:
                empirical_unc = float(np.std(values))

        # Quadrature addition for independent sources
        return float(np.sqrt(resolution_unc**2 + model_unc**2 + empirical_unc**2))

    # ------------------------------------------------------------------
    # Memory analysis (uses observation_log internally)
    # ------------------------------------------------------------------

    def has_memory(self, threshold: float = 0.5, max_lag: int = 5) -> bool:
        """Check if observer has memory structure via temporal correlation.

        Args:
            threshold: Minimum correlation to consider significant.
            max_lag: Maximum temporal lag to check.

        Returns:
            True if significant temporal correlation detected.
        """
        from ro_framework.correlation.measures import temporal_correlation

        internal_states = self.observation_log.get_internal_states()
        if self.temporal_dof is None or len(internal_states) < 3:
            return False

        for dof in self.internal_dofs:
            try:
                for lag in range(1, min(max_lag + 1, len(internal_states) // 2)):
                    corr = temporal_correlation(
                        states=internal_states,
                        dof=dof,
                        temporal_dof=self.temporal_dof,
                        lag=lag,
                    )
                    if abs(corr) > threshold:
                        return True
            except (ValueError, ZeroDivisionError):
                continue
        return False

    def analyze_memory_structure(self, max_lag: int = 10) -> Dict[DoF, List[float]]:
        """Analyze memory structure via temporal correlation profiles.

        Returns:
            Dict mapping each internal DoF to its lag-correlation profile.
        """
        from ro_framework.correlation.measures import temporal_correlation

        internal_states = self.observation_log.get_internal_states()
        if self.temporal_dof is None or len(internal_states) < 3:
            return {}

        analysis: Dict[DoF, List[float]] = {}
        for dof in self.internal_dofs:
            try:
                correlations = []
                for lag in range(1, min(max_lag + 1, len(internal_states) // 2)):
                    corr = temporal_correlation(
                        states=internal_states,
                        dof=dof,
                        temporal_dof=self.temporal_dof,
                        lag=lag,
                    )
                    correlations.append(corr)
                analysis[dof] = correlations
            except (ValueError, ZeroDivisionError):
                analysis[dof] = []
        return analysis

    def get_memory_correlations(self, dof1: DoF, dof2: DoF) -> float:
        """Compute Pearson correlation between two DoFs across observation history."""
        from ro_framework.correlation.measures import pearson_correlation

        internal_states = self.observation_log.get_internal_states()
        if len(internal_states) < 2:
            return 0.0
        try:
            return pearson_correlation(internal_states, dof1, dof2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def clear_memory(self) -> None:
        """Clear the observation log."""
        self.observation_log.clear()

    def get_memory_size(self) -> int:
        """Get number of recorded observations."""
        return len(self.observation_log)

    # ------------------------------------------------------------------
    # Consciousness
    # ------------------------------------------------------------------

    def recursive_depth(self) -> int:
        """Compute depth of recursive self-modeling by following the structural chain.

        - Depth 0: No self-model
        - Depth 1: Self-model exists (internal -> internal)
        - Depth 2+: Self-model itself has a self-model (nested recursion)

        Returns:
            Recursive depth.
        """
        if self.self_model is None:
            return 0

        depth = 1
        current = self.self_model
        while hasattr(current, "self_model") and current.self_model is not None:
            depth += 1
            current = current.self_model
            if depth > 10:
                break
        return depth

    def is_conscious(self, threshold: float = 0.5, test_states: Optional[List[State]] = None) -> bool:
        """Check if observer is structurally conscious.

        Args:
            threshold: Minimum consciousness score.
            test_states: Optional test states for evaluation.

        Returns:
            True if consciousness score exceeds threshold.
        """
        from ro_framework.consciousness.evaluation import ConsciousnessEvaluator

        evaluator = ConsciousnessEvaluator(self)
        metrics = evaluator.evaluate(test_states)
        return metrics.consciousness_score() >= threshold

    def get_consciousness_metrics(self, test_states: Optional[List[State]] = None):
        """Get full consciousness evaluation metrics.

        Returns:
            ConsciousnessMetrics with all measurements and overall score.
        """
        from ro_framework.consciousness.evaluation import ConsciousnessEvaluator

        evaluator = ConsciousnessEvaluator(self)
        return evaluator.evaluate(test_states)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Observer(name='{self.name}', "
            f"internal_dofs={len(self.internal_dofs)}, "
            f"external_dofs={len(self.external_dofs)}, "
            f"has_self_model={self.self_model is not None}, "
            f"observations={len(self.observation_log)})"
        )
