"""
Knowledge trajectory tracking over training epochs.

Records K(d_ext) = (ρ, ε, σ, C) at each step, enabling temporal analysis
of how knowledge forms — not just what it is at the end.  Detects phase
transitions: grokking (weak → strong), resonance (rising ρ, high σ),
and forgetting (ρ drops).

Note: "training" here means training the *model itself* (e.g. a neural
network), not training an SAE or other interpretability tool.  The tracker
monitors how the model's knowledge of specific external DoFs evolves as
its weights change during standard training.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ro_framework.core.dof import DoF
from ro_framework.knowledge.assessment import KnowledgeAssessment


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single snapshot of knowledge at a given epoch.

    Attributes:
        epoch: Training step / epoch number.
        assessment: The KnowledgeAssessment computed at this epoch.
    """

    epoch: int
    assessment: KnowledgeAssessment

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        a = self.assessment
        return {
            "epoch": self.epoch,
            "correlation": a.correlation,
            "systematic_error": a.systematic_error,
            "random_error": a.random_error,
            "calibration": a.calibration,
            "n_samples": a.n_samples,
            "external_dof": a.external_dof.to_dict(),
            "best_internal_dof": a.best_internal_dof.to_dict() if a.best_internal_dof else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryPoint":
        """Reconstruct from serialized dictionary."""
        ext_dof = DoF.from_dict(d["external_dof"])
        best_int = DoF.from_dict(d["best_internal_dof"]) if d["best_internal_dof"] else None
        assessment = KnowledgeAssessment(
            external_dof=ext_dof,
            best_internal_dof=best_int,
            correlation=d["correlation"],
            systematic_error=d["systematic_error"],
            random_error=d["random_error"],
            calibration=d["calibration"],
            n_samples=d["n_samples"],
        )
        return cls(epoch=d["epoch"], assessment=assessment)


class KnowledgeTracker:
    """Tracks K(d_ext) over training epochs for specified external DoFs.

    Wraps an Observer and records knowledge assessments at configurable
    intervals, enabling temporal analysis of knowledge formation and
    phase transition detection.

    Example::

        tracker = KnowledgeTracker(observer, external_dofs=[temp_dof])

        for epoch in range(1000):
            train_one_epoch(model)
            feed_observations(observer)
            tracker.step(epoch)

        # When did the model learn temperature?
        grok_epoch = tracker.detect_grokking(temp_dof)

        # Full trajectory
        for pt in tracker.trajectory(temp_dof):
            print(f"Epoch {pt.epoch}: ρ={pt.assessment.correlation:.3f}")

    Args:
        observer: The Observer whose knowledge to track.
        external_dofs: Which external DoFs to monitor.  Defaults to
            ``observer.external_dofs``.
        assess_interval: Assess every N steps (1 = every step).
        min_samples: Minimum observation pairs for assessment
            (passed to ``observer.assess_knowledge``).
    """

    def __init__(
        self,
        observer: Any,  # Observer, but avoid circular import at runtime
        external_dofs: Optional[List[DoF]] = None,
        assess_interval: int = 1,
        min_samples: int = 10,
        max_features: int = 1,
    ) -> None:
        self.observer = observer
        self.external_dofs = list(external_dofs) if external_dofs else list(observer.external_dofs)
        self.assess_interval = max(1, assess_interval)
        self.min_samples = min_samples
        self.max_features = max_features
        self._trajectories: Dict[str, List[TrajectoryPoint]] = {
            dof.name: [] for dof in self.external_dofs
        }
        self._step_count = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(self, epoch: int) -> Dict[DoF, Optional[KnowledgeAssessment]]:
        """Record knowledge assessments at the current epoch.

        If ``epoch`` is not aligned with ``assess_interval``, assessments
        are skipped and the method returns an empty dict.

        Args:
            epoch: Current training epoch / step number.

        Returns:
            Dict mapping each tracked DoF to its KnowledgeAssessment
            (or None if insufficient data).
        """
        self._step_count += 1
        if self._step_count % self.assess_interval != 0 and self._step_count != 1:
            return {}

        results: Dict[DoF, Optional[KnowledgeAssessment]] = {}
        for dof in self.external_dofs:
            assessment = self.observer.assess_knowledge(
                dof, min_samples=self.min_samples, max_features=self.max_features,
            )
            results[dof] = assessment
            if assessment is not None:
                self._trajectories[dof.name].append(
                    TrajectoryPoint(epoch=epoch, assessment=assessment)
                )
        return results

    def trajectory(self, dof: DoF) -> List[TrajectoryPoint]:
        """Get the full trajectory for a specific DoF.

        Args:
            dof: The external DoF to get trajectory for.

        Returns:
            List of TrajectoryPoints in chronological order.
        """
        return list(self._trajectories.get(dof.name, []))

    def latest(self, dof: DoF) -> Optional[KnowledgeAssessment]:
        """Get the most recent assessment for a DoF.

        Args:
            dof: The external DoF.

        Returns:
            Most recent KnowledgeAssessment, or None if no assessments yet.
        """
        points = self._trajectories.get(dof.name, [])
        return points[-1].assessment if points else None

    # ------------------------------------------------------------------
    # Phase transition detection
    # ------------------------------------------------------------------

    def detect_grokking(self, dof: DoF) -> Optional[int]:
        """Find the epoch where knowledge transitions to "strong".

        Grokking is the moment when knowledge_type changes from
        weak/false/uncertain to "strong" for the first time.

        Args:
            dof: The external DoF to check.

        Returns:
            Epoch of the transition, or None if never grokked.
        """
        points = self._trajectories.get(dof.name, [])
        prev_type: Optional[str] = None
        for pt in points:
            curr_type = pt.assessment.knowledge_type
            if curr_type == "strong" and prev_type in (None, "weak", "false", "uncertain"):
                return pt.epoch
            prev_type = curr_type
        return None

    def detect_resonance(
        self,
        dof: DoF,
        rho_threshold: float = 0.3,
        sigma_threshold: float = 0.5,
    ) -> List[int]:
        """Find epochs where a feature is locking in (pre-grokking).

        Resonance: correlation is rising (above threshold) but random
        error is still high — the feature is being amplified but hasn't
        stabilized yet.

        Args:
            dof: The external DoF to check.
            rho_threshold: Minimum correlation to count as "rising".
            sigma_threshold: Minimum random error to count as "not yet stable".

        Returns:
            List of epochs exhibiting resonance.
        """
        points = self._trajectories.get(dof.name, [])
        resonance_epochs: List[int] = []

        for i, pt in enumerate(points):
            a = pt.assessment
            rho_rising = a.correlation >= rho_threshold
            sigma_high = a.random_error >= sigma_threshold

            # Also check that ρ increased from previous point
            if i > 0:
                prev_rho = points[i - 1].assessment.correlation
                rho_increasing = a.correlation > prev_rho
            else:
                rho_increasing = rho_rising  # first point: just check threshold

            if rho_rising and sigma_high and rho_increasing:
                resonance_epochs.append(pt.epoch)

        return resonance_epochs

    def detect_forgetting(self, dof: DoF, rho_drop: float = 0.2) -> List[int]:
        """Find epochs where correlation drops significantly.

        Forgetting is detected when ρ drops by more than ``rho_drop``
        from the previous peak correlation.

        Args:
            dof: The external DoF to check.
            rho_drop: Minimum drop from peak to count as forgetting.

        Returns:
            List of epochs where forgetting was detected.
        """
        points = self._trajectories.get(dof.name, [])
        forgetting_epochs: List[int] = []
        peak_rho = 0.0

        for pt in points:
            rho = pt.assessment.correlation
            if rho > peak_rho:
                peak_rho = rho
            elif peak_rho - rho >= rho_drop:
                forgetting_epochs.append(pt.epoch)

        return forgetting_epochs

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state to a JSON-compatible dictionary.

        The Observer is NOT serialized (same pattern as world_model in
        Observer — user re-supplies on load).
        """
        return {
            "assess_interval": self.assess_interval,
            "min_samples": self.min_samples,
            "external_dofs": [dof.to_dict() for dof in self.external_dofs],
            "step_count": self._step_count,
            "trajectories": {
                name: [pt.to_dict() for pt in pts]
                for name, pts in self._trajectories.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        observer: Any,
    ) -> "KnowledgeTracker":
        """Reconstruct a KnowledgeTracker from its serialized dictionary.

        Args:
            d: Dictionary from ``to_dict()``.
            observer: The Observer to attach to (must be re-supplied).

        Returns:
            Reconstructed KnowledgeTracker with trajectory history.
        """
        external_dofs = [DoF.from_dict(dd) for dd in d["external_dofs"]]
        tracker = cls(
            observer=observer,
            external_dofs=external_dofs,
            assess_interval=d["assess_interval"],
            min_samples=d["min_samples"],
        )
        tracker._step_count = d["step_count"]
        tracker._trajectories = {
            name: [TrajectoryPoint.from_dict(pt) for pt in pts]
            for name, pts in d["trajectories"].items()
        }
        return tracker

    def save(self, path: Union[str, Path]) -> None:
        """Save tracker state to a JSON file.

        Args:
            path: File path to write to.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        observer: Any,
    ) -> "KnowledgeTracker":
        """Load a KnowledgeTracker from a JSON file.

        Args:
            path: File path to read from.
            observer: The Observer to attach to (must be re-supplied).

        Returns:
            Reconstructed KnowledgeTracker.
        """
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d, observer)

    def __repr__(self) -> str:
        n_points = sum(len(pts) for pts in self._trajectories.values())
        return (
            f"KnowledgeTracker(dofs={len(self.external_dofs)}, "
            f"points={n_points}, steps={self._step_count})"
        )
