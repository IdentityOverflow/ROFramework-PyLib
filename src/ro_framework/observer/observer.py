"""
Observer implementation.

An observer is a configuration within the Block Universe characterized by:
- Boundary (internal/external DoF partition)
- Mapping functions (external -> internal)
- Resolution (per-DoF finite granularity)
- Memory (correlation structure across temporal DoF via ObservationLog)

O = (B, M, R, Mem)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "external_state": self.external_state.to_dict(),
            "internal_state": self.internal_state.to_dict(),
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObservationPair":
        return ObservationPair(
            external_state=State.from_dict(d["external_state"]),
            internal_state=State.from_dict(d["internal_state"]),
            timestamp=d["timestamp"],
        )


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "pairs": [p.to_dict() for p in self._pairs],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObservationLog":
        log = ObservationLog(capacity=d["capacity"])
        for p in d["pairs"]:
            log.append(ObservationPair.from_dict(p))
        return log


@dataclass(frozen=True)
class ClosureAssessment:
    """Closed(O) recognition result (ro_framework.md §5.3).

    Closure asks where the self-model's outputs GO: a probe's
    self-representations are exported (reports, logs, external readers);
    a loop's are consumed by the observer's own subsequent processing.

    Criteria:
        structural: a consumption path exists — consumption_gain > 0 with a
            self-model whose output DoFs (d_meta) are declared.
        corr_internal: max |lagged correlation| between d_meta values at t
            and internal DoF values at t+lag.
        corr_external: same against external DoF values at t+lag
            (the "external consumer" side of §5.3's comparison).
        closed: structural AND corr_internal > corr_external AND enough
            samples. A recognition criterion, like B, M, R, Mem — computed
            from static correlation structure in the observation log, not
            from any runtime activity.
    """

    structural: bool
    corr_internal: float
    corr_external: float
    n_samples: int
    closed: bool


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
        consumption_gain: g — strength with which the self-model's outputs
            are routed back into the world model's input on each observe()
            (0.0 = pure probe, exact pre-v2 behavior; >0 = loop). Sweepable
            by design: the §5.5 closure-sweep experiment turns this knob.
        self_encoder: Optional BehavioralEncoder (§5.4, the twist): when
            attached, the self-model's input is augmented with a behavioral
            encoding of the self-model itself plus R(d_meta) — the
            self-model receives a description of its own representing.
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
    consumption_gain: float = 0.0
    self_encoder: Optional[Any] = None
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

        Raises:
            ValueError: If external_state is missing declared external DoFs,
                or if the world model output is missing declared internal DoFs.
        """
        missing_ext = [d for d in self.external_dofs if external_state.get_value(d) is None]
        if missing_ext:
            raise ValueError(
                f"External state missing declared DoFs: {[d.name for d in missing_ext]}"
            )

        # Consumption loop (§5.3, closure): route the self-model's reading of
        # the PREVIOUS internal state back into the world model's input,
        # scaled by g. At g=0 this block is inert and observe() is exactly
        # the pre-v2 probe. Correlation across the temporal DoF (last cycle's
        # d_meta -> this cycle's processing) is what closure_assessment()
        # later recognizes.
        input_state = external_state
        if (self.consumption_gain > 0.0
                and self.self_model is not None
                and self.internal_state is not None):
            meta_state = self.self_model(
                self._augment_with_self_encoding(self.internal_state))
            for dof in self.d_meta:
                v = meta_state.get_value(dof)
                if isinstance(v, (int, float)):
                    input_state = input_state.set_value(
                        dof, self.consumption_gain * float(v)
                    )

        internal_state = self.world_model(input_state)

        missing_int = [d for d in self.internal_dofs if internal_state.get_value(d) is None]
        if missing_int:
            raise ValueError(
                f"World model output missing declared internal DoFs: "
                f"{[d.name for d in missing_int]}"
            )

        self.internal_state = internal_state

        self.observation_log.append(ObservationPair(
            external_state=external_state,
            internal_state=internal_state,
            timestamp=float(len(self.observation_log)),
        ))

        return internal_state

    def observe_batch(self, external_states: List[State]) -> List[State]:
        """Observe a batch of external states.

        If the world model supports ``batch_call`` (e.g. ``_CallableMapping``),
        a single vectorized call is made. Otherwise falls back to sequential
        ``observe()`` calls.

        Args:
            external_states: List of external States to observe.

        Returns:
            List of internal States (each also logged in observation_log).
        """
        if not external_states:
            return []

        # Consumption is inherently sequential (each cycle consumes the
        # previous cycle's self-model output) — a vectorized batch would
        # silently open the loop. Fall back to sequential when g > 0.
        if self.consumption_gain > 0.0 and self.self_model is not None:
            return [self.observe(s) for s in external_states]

        if not hasattr(self.world_model, "batch_call"):
            return [self.observe(s) for s in external_states]

        # Validate external DoFs on all inputs
        for s in external_states:
            missing_ext = [d for d in self.external_dofs if s.get_value(d) is None]
            if missing_ext:
                raise ValueError(
                    f"External state missing declared DoFs: {[d.name for d in missing_ext]}"
                )

        internal_states = self.world_model.batch_call(external_states)

        # Validate and log each pair
        results: List[State] = []
        for ext_state, int_state in zip(external_states, internal_states):
            missing_int = [d for d in self.internal_dofs if int_state.get_value(d) is None]
            if missing_int:
                raise ValueError(
                    f"World model output missing declared internal DoFs: "
                    f"{[d.name for d in missing_int]}"
                )
            self.internal_state = int_state
            self.observation_log.append(ObservationPair(
                external_state=ext_state,
                internal_state=int_state,
                timestamp=float(len(self.observation_log)),
            ))
            results.append(int_state)
        return results

    def self_observe(self) -> Optional[State]:
        """Recursive self-modeling: map internal DoFs to internal DoFs.

        This is the SUBSTRATE condition of consciousness (§5.1) —
        internal->internal correlation with the same architectural type as
        external->internal correlation. It is necessary, not sufficient:
        the v2 criterion additionally requires closure (the self-model's
        outputs consumed inside the boundary — see closure_assessment())
        and the twist (the self-model representing its own representing).

        Returns:
            Self-representation state, or None if no self-model.
        """
        if self.self_model is None or self.internal_state is None:
            return None
        return self.self_model(
            self._augment_with_self_encoding(self.internal_state))

    def _augment_with_self_encoding(self, state: State) -> State:
        """Add the self-model's behavioral self-encoding to an input state.

        No-op without a self_encoder. Extra values on the state are
        harmless to mappings that do not declare the encoding DoFs;
        twist_assessment() checks declaration separately.
        """
        if self.self_encoder is None or self.self_model is None:
            return state
        resolution = {d: self.get_resolution(d) for d in self.d_meta}
        enc = self.self_encoder.encode(self.self_model, resolution)
        for dof in self.self_encoder.all_dofs:
            v = enc.get_value(dof)
            if v is not None:
                state = state.set_value(dof, float(v))
        return state

    @property
    def d_meta(self) -> List[DoF]:
        """The self-model's output DoFs — the carriers of self-representation.

        Closure (§5.3) is a question about where values on these DoFs go:
        consumed by the observer's own processing (loop) or exported
        (probe). Empty if there is no self-model or it declares no
        output_dofs.
        """
        if self.self_model is None:
            return []
        return list(getattr(self.self_model, "output_dofs", None) or [])

    def closure_assessment(
        self, lag: int = 1, min_samples: int = 10
    ) -> ClosureAssessment:
        """Recognize Closed(O): are the self-model's outputs consumed inside B?

        Per §5.3, closure holds when (i) a consumption path exists
        (d_meta reaches the world model's domain — realized here by
        consumption_gain > 0) and (ii) d_meta values at time t correlate
        more strongly with the observer's own internal configurations at
        t+lag than with external configurations at t+lag.

        The d_meta series is recomputed by applying the self-model to the
        logged internal states, so the assessment is a pure recognition
        over static correlation structure — it works retroactively on any
        observation history and needs no runtime bookkeeping. (Limitation:
        for a stochastic self-model the recomputed series is a resample,
        not a replay.)

        Args:
            lag: Temporal offset for the consumption correlation (>= 1).
            min_samples: Minimum aligned samples required.

        Returns:
            ClosureAssessment.
        """
        structural = (
            self.consumption_gain > 0.0
            and self.self_model is not None
            and len(self.d_meta) > 0
        )

        pairs = list(self.observation_log)
        if self.self_model is None or len(pairs) < min_samples + lag:
            return ClosureAssessment(
                structural=structural, corr_internal=0.0,
                corr_external=0.0, n_samples=len(pairs), closed=False,
            )

        meta_states = [self.self_model(p.internal_state) for p in pairs]

        def _series(states, dof):
            vals = []
            for s in states:
                v = s.get_value(dof) if s is not None else None
                vals.append(float(v) if isinstance(v, (int, float)) else None)
            return vals

        def _lagged_corr(m_vals, t_vals):
            a, b = [], []
            for i in range(len(m_vals) - lag):
                mv, tv = m_vals[i], t_vals[i + lag]
                if mv is not None and tv is not None:
                    a.append(mv)
                    b.append(tv)
            if len(a) < min_samples:
                return None
            a, b = np.asarray(a), np.asarray(b)
            if a.std() < 1e-12 or b.std() < 1e-12:
                return None
            return float(abs(np.corrcoef(a, b)[0, 1]))

        internal_states = [p.internal_state for p in pairs]
        external_states = [p.external_state for p in pairs]

        corr_int, corr_ext, n_used = 0.0, 0.0, 0
        for m_dof in self.d_meta:
            m_vals = _series(meta_states, m_dof)
            for t_dof in self.internal_dofs:
                c = _lagged_corr(m_vals, _series(internal_states, t_dof))
                if c is not None:
                    corr_int = max(corr_int, c)
                    n_used = max(n_used, len(pairs) - lag)
            for t_dof in self.external_dofs:
                c = _lagged_corr(m_vals, _series(external_states, t_dof))
                if c is not None:
                    corr_ext = max(corr_ext, c)

        closed = bool(structural and n_used >= min_samples
                      and corr_int > corr_ext)
        return ClosureAssessment(
            structural=structural, corr_internal=corr_int,
            corr_external=corr_ext, n_samples=n_used, closed=closed,
        )

    def is_closed(self, lag: int = 1, min_samples: int = 10) -> bool:
        """Convenience: Closed(O) as a bool. See closure_assessment()."""
        return self.closure_assessment(lag=lag, min_samples=min_samples).closed

    def twist_assessment(self, n_perturb: int = 8,
                         perturb_scale: float = 0.1, n_foils: int = 4,
                         foil_scale: float = 0.3, seed: int = 0):
        """Recognize twisted(O): does the self-model's output carry
        information about its own mapping beyond the state —
        I(d_meta ; M_self | S) > 0? (§5.4; white-box consumption checks +
        the state-matched battery-foil intervention test; see
        observer.self_encoding.TwistAssessment.)"""
        from ro_framework.observer.self_encoding import assess_twist

        return assess_twist(self, n_perturb=n_perturb,
                            perturb_scale=perturb_scale, n_foils=n_foils,
                            foil_scale=foil_scale, seed=seed)

    def is_twisted(self, n_perturb: int = 8, perturb_scale: float = 0.1,
                   n_foils: int = 4, foil_scale: float = 0.3,
                   seed: int = 0) -> bool:
        """Convenience: twisted(O) as a bool. See twist_assessment()."""
        return self.twist_assessment(
            n_perturb=n_perturb, perturb_scale=perturb_scale,
            n_foils=n_foils, foil_scale=foil_scale, seed=seed
        ).twisted

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def get_resolution(self, dof: DoF) -> float:
        """Get resolution limit for a specific DoF."""
        return self.resolution.get(dof, 1e-6)

    # ------------------------------------------------------------------
    # Knowledge assessment — K(d_ext) = (ρ, ε, σ, C)
    # ------------------------------------------------------------------

    def assess_knowledge(
        self, external_dof: DoF, min_samples: int = 10, max_features: int = 1,
    ):
        """Compute graded knowledge of an external DoF.

        Returns KnowledgeAssessment with correlation, bias, noise,
        and calibration, or None if insufficient observation history.

        Args:
            external_dof: The external DoF to assess knowledge of.
            min_samples: Minimum observations required.
            max_features: Maximum internal DoFs to use jointly.
                1 = single best feature (default). >1 = multiple regression.

        Returns:
            KnowledgeAssessment or None.
        """
        from ro_framework.knowledge.assessment import compute_knowledge

        return compute_knowledge(
            self.observation_log, external_dof, self.internal_dofs,
            min_samples, max_features,
        )

    def know(
        self,
        external_dof: DoF,
        threshold: float = 0.7,
        min_calibration: float = 0.4,
        min_samples: int = 10,
        max_features: int = 1,
    ) -> bool:
        """Check if observer has knowledge of an external DoF.

        Knowledge requires high correlation and reasonable calibration.

        Args:
            external_dof: External DoF to check.
            threshold: Minimum correlation for knowledge.
            min_calibration: Minimum calibration for knowledge.
            min_samples: Minimum observations required.
            max_features: Maximum internal DoFs to use jointly.

        Returns:
            True if knowledge criteria are met.
        """
        assessment = self.assess_knowledge(external_dof, min_samples, max_features)
        if assessment is None:
            return False
        return assessment.correlation >= threshold and assessment.calibration >= min_calibration

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

    def is_conscious(self, lag: int = 1, min_samples: int = 10,
                     n_perturb: int = 8) -> bool:
        """The v2 criterion, binary in kind (§5.5):

            conscious iff Closed(O) AND twisted(O)

        — the self-model's outputs are consumed inside the boundary AND
        the self-model represents its own representing. Richness (depth,
        bandwidth, integration, calibration) is graded and reported
        separately by richness().

        Note: pre-v2 this method thresholded a graded score; that score
        survives as richness().consciousness_score().
        """
        return (self.is_closed(lag=lag, min_samples=min_samples)
                and self.is_twisted(n_perturb=n_perturb))

    def richness(self, test_states: Optional[List[State]] = None):
        """Graded richness metrics (§5.5): depth, self-accuracy,
        architectural similarity, calibration, metacognition, limitation
        awareness — the tower's quality, orthogonal to the binary kind.

        Returns:
            ConsciousnessMetrics.
        """
        from ro_framework.consciousness.evaluation import ConsciousnessEvaluator

        return ConsciousnessEvaluator(self).evaluate(test_states)

    def get_consciousness_metrics(self, test_states: Optional[List[State]] = None):
        """Get full consciousness evaluation metrics.

        Returns:
            ConsciousnessMetrics with all measurements and overall score.
        """
        from ro_framework.consciousness.evaluation import ConsciousnessEvaluator

        evaluator = ConsciousnessEvaluator(self)
        return evaluator.evaluate(test_states)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observer metadata, DoFs, resolution, and observation log.

        The world_model and self_model are NOT serialized (callables/models
        cannot be reliably serialized). They must be re-supplied on load.
        """
        return {
            "name": self.name,
            "internal_dofs": [d.to_dict() for d in self.internal_dofs],
            "external_dofs": [d.to_dict() for d in self.external_dofs],
            "resolution": {d.name: v for d, v in self.resolution.items()},
            "temporal_dof": self.temporal_dof.to_dict() if self.temporal_dof else None,
            "log_capacity": self.log_capacity,
            "consumption_gain": self.consumption_gain,
            "observation_log": self.observation_log.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        world_model: MappingFunction,
        self_model: Optional[MappingFunction] = None,
    ) -> "Observer":
        """Reconstruct an Observer from a serialized dictionary.

        Args:
            d: Dictionary from ``to_dict()``.
            world_model: The world model mapping (must be re-supplied).
            self_model: Optional self-model mapping.

        Returns:
            Reconstructed Observer with restored observation history.
        """
        internal_dofs = [DoF.from_dict(dd) for dd in d["internal_dofs"]]
        external_dofs = [DoF.from_dict(dd) for dd in d["external_dofs"]]

        # Rebuild resolution dict keyed by the reconstructed DoF objects
        dof_by_name = {dof.name: dof for dof in internal_dofs}
        resolution = {dof_by_name[name]: val for name, val in d["resolution"].items()
                      if name in dof_by_name}

        temporal_dof = DoF.from_dict(d["temporal_dof"]) if d.get("temporal_dof") else None

        obs = cls(
            name=d["name"],
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=world_model,
            self_model=self_model,
            resolution=resolution,
            temporal_dof=temporal_dof,
            log_capacity=d.get("log_capacity", 1000),
            consumption_gain=d.get("consumption_gain", 0.0),
        )

        # Restore observation log
        obs.observation_log = ObservationLog.from_dict(d["observation_log"])

        return obs

    def save(self, path: Union[str, Path]) -> None:
        """Save observer to a JSON file.

        The world_model and self_model are NOT saved. They must be
        re-supplied when loading via ``Observer.load()``.

        Args:
            path: File path to write to.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        world_model: MappingFunction,
        self_model: Optional[MappingFunction] = None,
    ) -> "Observer":
        """Load observer from a JSON file.

        Args:
            path: File path to read from.
            world_model: The world model mapping (must be re-supplied).
            self_model: Optional self-model mapping.

        Returns:
            Reconstructed Observer with restored observation history.
        """
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d, world_model=world_model, self_model=self_model)

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
