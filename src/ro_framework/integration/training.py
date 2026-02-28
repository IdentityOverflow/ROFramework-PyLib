"""Knowledge-guided training utilities.

Provides ``KnowledgeRegularizer`` which reads the knowledge assessment tuple
K(d_ext) = (ρ, ε, σ, C) from a ``KnowledgeTracker`` and adjusts training
dynamics (weight decay, loss penalties) to accelerate feature generalization.

The core hypothesis: if grokking is driven by competition between loss
minimization and weight-decay regularization, then selectively increasing
regularization pressure when features are memorized (high ρ, low C) should
push the model toward resonant (generalized) solutions faster.

Usage::

    regularizer = KnowledgeRegularizer(tracker, base_weight_decay=1.0)

    for epoch in range(num_epochs):
        # ... standard training step ...
        if epoch % eval_interval == 0:
            tracker.step(epoch)
            regularizer.update(epoch)
            for pg in optimizer.param_groups:
                pg['weight_decay'] = regularizer.get_weight_decay()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ro_framework.knowledge.tracker import KnowledgeTracker


@dataclass
class FeatureRegularization:
    """Per-feature regularization state snapshot.

    Attributes:
        feature_name: Name of the external DoF being tracked.
        knowledge_type: Classification from KnowledgeAssessment
            (``"strong"``, ``"weak"``, ``"false"``, ``"uncertain"``).
        correlation: ρ — how strongly an internal DoF tracks this feature.
        systematic_error: ε — signed bias in the mapping.
        calibration: C — whether stated uncertainty matches actual error.
        weight_decay_multiplier: Computed multiplier for this feature.
        epoch_updated: Epoch at which this state was last computed.
    """

    feature_name: str
    knowledge_type: str
    correlation: float
    systematic_error: float
    calibration: float
    weight_decay_multiplier: float
    epoch_updated: int


class KnowledgeRegularizer:
    """Adjusts training regularization based on knowledge assessment dynamics.

    Reads K(d_ext) from a :class:`KnowledgeTracker` at eval intervals and
    computes:

    1. **Weight decay multiplier** — increased when features are memorized
       (high ρ, low C), decreased when generalized (high ρ, high C).
    2. **Bias penalty** — additive loss term for features with systematic
       error (``knowledge_type == "false"``).

    The multipliers are aggregated into a single global weight decay scalar
    applied to the optimizer. This matches the grokking theory where global
    regularization pressure (not per-parameter) drives frequency selection.

    Args:
        tracker: KnowledgeTracker to read assessments from.
        base_weight_decay: The optimizer's base weight decay value.
        memorized_multiplier: WD multiplier when features are memorized
            (high ρ, low C). Values > 1 increase regularization.
        generalized_multiplier: WD multiplier when all features are
            generalized (high ρ, high C). Values < 1 decrease it.
        uncertain_multiplier: WD multiplier when features are uncertain
            (low ρ). Typically 1.0 (no change).
        false_multiplier: WD multiplier when features have false knowledge
            (high ρ, high |ε|).
        bias_penalty_weight: Coefficient for the additive bias penalty term.
        memorized_min_correlation: ρ threshold above which a feature is
            considered "tracked" for memorization detection.
        memorized_max_calibration: C threshold below which a tracked
            feature is classified as memorized.
        generalized_min_correlation: ρ threshold for generalization.
        generalized_min_calibration: C threshold for generalization.
    """

    def __init__(
        self,
        tracker: KnowledgeTracker,
        base_weight_decay: float = 1.0,
        memorized_multiplier: float = 3.0,
        generalized_multiplier: float = 0.5,
        uncertain_multiplier: float = 1.0,
        false_multiplier: float = 2.0,
        bias_penalty_weight: float = 0.1,
        memorized_min_correlation: float = 0.5,
        memorized_max_calibration: float = 0.3,
        generalized_min_correlation: float = 0.7,
        generalized_min_calibration: float = 0.5,
    ) -> None:
        self._tracker = tracker
        self._base_wd = base_weight_decay
        self._memorized_mult = memorized_multiplier
        self._generalized_mult = generalized_multiplier
        self._uncertain_mult = uncertain_multiplier
        self._false_mult = false_multiplier
        self._bias_penalty_weight = bias_penalty_weight
        self._mem_min_rho = memorized_min_correlation
        self._mem_max_cal = memorized_max_calibration
        self._gen_min_rho = generalized_min_correlation
        self._gen_min_cal = generalized_min_calibration

        self._feature_states: Dict[str, FeatureRegularization] = {}
        self._current_multiplier: float = 1.0

    def update(self, epoch: int) -> Dict[str, FeatureRegularization]:
        """Read latest K from tracker and compute per-feature multipliers.

        Should be called at eval intervals, after ``tracker.step(epoch)``.

        Args:
            epoch: Current training epoch.

        Returns:
            Dictionary mapping feature names to their regularization state.
        """
        self._feature_states.clear()

        for dof in self._tracker.external_dofs:
            latest = self._tracker.latest(dof)
            if latest is None:
                continue

            multiplier = self._classify_multiplier(
                latest.correlation,
                latest.calibration,
                latest.knowledge_type,
            )

            self._feature_states[dof.name] = FeatureRegularization(
                feature_name=dof.name,
                knowledge_type=latest.knowledge_type,
                correlation=latest.correlation,
                systematic_error=latest.systematic_error,
                calibration=latest.calibration,
                weight_decay_multiplier=multiplier,
                epoch_updated=epoch,
            )

        self._current_multiplier = self._aggregate_multiplier()
        return dict(self._feature_states)

    def get_weight_decay(self) -> float:
        """Return the current effective weight decay.

        Multiplies the base weight decay by the aggregated multiplier
        from the latest ``update()`` call.
        """
        return self._base_wd * self._current_multiplier

    def get_loss_penalty(self) -> float:
        """Return an additive loss penalty for features with systematic bias.

        For features with ``knowledge_type == "false"`` (high correlation
        but high systematic error), returns a penalty proportional to |ε|.
        This can be added to the loss before ``backward()`` to encourage
        the model to reduce bias.
        """
        penalty = 0.0
        for state in self._feature_states.values():
            if state.knowledge_type == "false":
                penalty += abs(state.systematic_error)
        return penalty * self._bias_penalty_weight

    def feature_states(self) -> Dict[str, FeatureRegularization]:
        """Return current per-feature regularization states."""
        return dict(self._feature_states)

    @property
    def current_multiplier(self) -> float:
        """The current aggregate weight decay multiplier."""
        return self._current_multiplier

    def _classify_multiplier(
        self, correlation: float, calibration: float, knowledge_type: str
    ) -> float:
        """Determine the weight decay multiplier for a single feature."""
        if knowledge_type == "false":
            return self._false_mult

        if (
            correlation >= self._mem_min_rho
            and calibration < self._mem_max_cal
        ):
            # Memorized: high correlation but poor calibration
            return self._memorized_mult

        if (
            correlation >= self._gen_min_rho
            and calibration >= self._gen_min_cal
        ):
            # Generalized: high correlation and good calibration
            return self._generalized_mult

        # Uncertain or weak: don't interfere
        return self._uncertain_mult

    def _aggregate_multiplier(self) -> float:
        """Aggregate per-feature multipliers into a single scalar.

        Strategy: conservative. If ANY feature is memorized (multiplier > 1),
        use the maximum multiplier (push harder). Only reduce weight decay
        when ALL features are generalized.
        """
        if not self._feature_states:
            return 1.0

        multipliers = [
            s.weight_decay_multiplier for s in self._feature_states.values()
        ]

        max_mult = max(multipliers)

        # If any feature needs increased regularization, use the max
        if max_mult > 1.0:
            return max_mult

        # Only reduce if all features are at or below 1.0
        # Use the minimum (most aggressive reduction) — all features agree
        return min(multipliers)
