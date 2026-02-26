"""
Consciousness evaluation based on structural criteria.

This module evaluates consciousness as a structural property:
recursive self-modeling with bounded error, not phenomenal experience.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Optional

import numpy as np

from ro_framework.core.dof import DoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer


@dataclass
class ConsciousnessMetrics:
    """
    Metrics for evaluating structural consciousness.

    These are observable, testable properties - not phenomenal claims.
    """

    has_self_model: bool
    recursive_depth: int
    self_accuracy: float  # How accurately self-model represents internal state
    architectural_similarity: float  # Similarity between world and self models
    calibration_error: float  # |confidence - accuracy|
    meta_cognitive_capability: float  # Can reason about own reasoning
    limitation_awareness: float  # Knows what it doesn't know

    # Default scoring weights — override on the class to change defaults.
    DEFAULT_WEIGHTS: ClassVar[Dict[str, float]] = {
        "recursive_depth": 0.20,
        "self_accuracy": 0.25,
        "architectural_similarity": 0.15,
        "calibration": 0.15,
        "meta_cognitive_capability": 0.15,
        "limitation_awareness": 0.10,
    }

    def consciousness_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute overall consciousness score [0, 1].

        Combines multiple metrics with weights based on importance.

        Args:
            weights: Optional dict overriding ``DEFAULT_WEIGHTS``.
                Keys: recursive_depth, self_accuracy, architectural_similarity,
                calibration, meta_cognitive_capability, limitation_awareness.

        Returns:
            Score from 0 (no consciousness) to 1 (full consciousness)
        """
        if not self.has_self_model:
            return 0.0

        w = weights or self.DEFAULT_WEIGHTS

        score = 0.0
        score += w.get("recursive_depth", 0.2) * min(self.recursive_depth / 3.0, 1.0)
        score += w.get("self_accuracy", 0.25) * self.self_accuracy
        score += w.get("architectural_similarity", 0.15) * self.architectural_similarity
        score += w.get("calibration", 0.15) * (1.0 - self.calibration_error)
        score += w.get("meta_cognitive_capability", 0.15) * self.meta_cognitive_capability
        score += w.get("limitation_awareness", 0.10) * self.limitation_awareness

        return float(np.clip(score, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for easy inspection."""
        return {
            "has_self_model": self.has_self_model,
            "recursive_depth": self.recursive_depth,
            "self_accuracy": self.self_accuracy,
            "architectural_similarity": self.architectural_similarity,
            "calibration_error": self.calibration_error,
            "meta_cognitive_capability": self.meta_cognitive_capability,
            "limitation_awareness": self.limitation_awareness,
            "overall_score": self.consciousness_score(),
        }


def _binned_ece(uncertainties: np.ndarray, errors: np.ndarray) -> float:
    """Compute binned ECE, normalized to [0, 1].

    Bins by stated uncertainty using equal-frequency quantiles,
    then computes weighted |avg_uncertainty − avg_error| per bin.
    """
    n_bins = min(5, len(uncertainties) // 2)
    if n_bins < 1:
        return 0.5

    bin_edges = np.quantile(uncertainties, np.linspace(0, 1, n_bins + 1))
    ece = 0.0
    total = 0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (uncertainties >= lo) & (uncertainties <= hi if i == n_bins - 1 else uncertainties < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += count * abs(float(uncertainties[mask].mean() - errors[mask].mean()))
        total += count

    if total == 0:
        return 0.5

    ece /= total
    return float(np.clip(ece / (ece + 1.0), 0.0, 1.0))


class ConsciousnessEvaluator:
    """
    Evaluate structural features of consciousness in observers.

    Tests STRUCTURAL properties, not phenomenal experience:
    - Self-modeling capability
    - Recursive depth
    - Integration of world and self models
    - Calibration quality
    - Adaptability

    Example:
        >>> evaluator = ConsciousnessEvaluator(observer)
        >>> metrics = evaluator.evaluate()
        >>> print(f"Consciousness score: {metrics.consciousness_score():.2f}")
        >>> if metrics.has_self_model:
        ...     print(f"Recursive depth: {metrics.recursive_depth}")
    """

    def __init__(self, observer: Observer):
        """
        Initialize evaluator.

        Args:
            observer: Observer to evaluate
        """
        self.observer = observer

    def evaluate(self, test_states: List[State] = None) -> ConsciousnessMetrics:
        """
        Run complete consciousness evaluation.

        Args:
            test_states: Optional test states for evaluation

        Returns:
            ConsciousnessMetrics with all measurements
        """
        # 1. Check for self-model
        has_self_model = self.observer.self_model is not None

        if not has_self_model:
            return ConsciousnessMetrics(
                has_self_model=False,
                recursive_depth=0,
                self_accuracy=0.0,
                architectural_similarity=0.0,
                calibration_error=1.0,
                meta_cognitive_capability=0.0,
                limitation_awareness=0.0,
            )

        # 2. Measure recursive depth
        recursive_depth = self.observer.recursive_depth()

        # 3. Measure self-accuracy
        self_accuracy = self._evaluate_self_accuracy(test_states)

        # 4. Measure architectural similarity
        arch_similarity = self._evaluate_architectural_similarity()

        # 5. Measure calibration
        calibration_error = self._evaluate_calibration(test_states)

        # 6. Measure meta-cognitive capability
        meta_cog = self._evaluate_metacognition()

        # 7. Measure limitation awareness
        limit_aware = self._evaluate_limitation_awareness(test_states)

        return ConsciousnessMetrics(
            has_self_model=has_self_model,
            recursive_depth=recursive_depth,
            self_accuracy=self_accuracy,
            architectural_similarity=arch_similarity,
            calibration_error=calibration_error,
            meta_cognitive_capability=meta_cog,
            limitation_awareness=limit_aware,
        )

    def _evaluate_self_accuracy(self, test_states: List[State] = None) -> float:
        """
        Evaluate how accurately self-model represents internal state.

        Metric: How close is self-observation to actual internal state?

        Args:
            test_states: Test states to evaluate on

        Returns:
            Accuracy score [0, 1]
        """
        if test_states is None or len(test_states) == 0:
            # No test data - use current internal state if available
            if self.observer.internal_state is None:
                return 0.5  # Unknown

            internal = self.observer.internal_state
            self_repr = self.observer.self_observe()

            if self_repr is None:
                return 0.0

            # Compute similarity
            distance = internal.distance_to(self_repr)
            # Normalize to [0, 1] (closer = higher accuracy)
            accuracy = 1.0 / (1.0 + distance)
            return float(accuracy)

        # Evaluate on test states
        accuracies = []
        for ext_state in test_states:
            internal = self.observer.observe(ext_state)
            self_repr = self.observer.self_observe()

            if self_repr is not None:
                distance = internal.distance_to(self_repr)
                accuracy = 1.0 / (1.0 + distance)
                accuracies.append(accuracy)

        return float(np.mean(accuracies)) if accuracies else 0.0

    def _evaluate_architectural_similarity(self) -> float:
        """
        Evaluate structural similarity between world model and self-model.

        The framework requires M_self to have the "same architectural type"
        as M_world.  We measure this along three axes and average:

        1. Type match — same Python class is highest; same base class partial
        2. DoF-dimensionality ratio — closer to 1.0 is better
        3. Shared structural attributes (input_dofs, output_dofs, resolution, model)

        Returns:
            Similarity score [0, 1]
        """
        if self.observer.self_model is None:
            return 0.0

        world = self.observer.world_model
        self_m = self.observer.self_model

        # --- axis 1: type match ---
        if type(world) is type(self_m):
            type_score = 1.0
        elif type(world).__bases__ == type(self_m).__bases__:
            type_score = 0.6
        else:
            type_score = 0.2

        # --- axis 2: dimensionality ratio ---
        world_out = len(getattr(world, "output_dofs", []) or [])
        self_out = len(getattr(self_m, "output_dofs", []) or [])
        if world_out > 0 and self_out > 0:
            ratio = min(world_out, self_out) / max(world_out, self_out)
            dim_score = ratio
        else:
            # If neither exposes output_dofs, give neutral credit
            dim_score = 0.5

        # --- axis 3: shared structural attributes ---
        structural_attrs = ("input_dofs", "output_dofs", "resolution", "model")
        world_attrs = {a for a in structural_attrs if hasattr(world, a)}
        self_attrs = {a for a in structural_attrs if hasattr(self_m, a)}
        if world_attrs or self_attrs:
            attr_score = len(world_attrs & self_attrs) / len(world_attrs | self_attrs)
        else:
            attr_score = 0.5

        return float(np.clip(
            0.5 * type_score + 0.25 * dim_score + 0.25 * attr_score,
            0.0, 1.0,
        ))

    def _evaluate_calibration(self, test_states: List[State] = None) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Compares stated uncertainty (estimate_uncertainty) against actual
        self-model prediction error across internal DoFs, then bins by
        stated uncertainty to compute ECE.

        Returns:
            Calibration error [0, 1] (lower is better)
        """
        obs = self.observer
        if obs.self_model is None:
            return 1.0

        pairs = self._collect_calibration_pairs(test_states)
        if len(pairs) < 3:
            return 0.5

        uncertainties = np.array([p[0] for p in pairs])
        errors = np.array([p[1] for p in pairs])
        return _binned_ece(uncertainties, errors)

    def _collect_calibration_pairs(
        self, test_states: List[State] = None,
    ) -> List[tuple]:
        """Collect (stated_uncertainty, actual_error) pairs."""
        obs = self.observer
        if test_states and len(test_states) >= 3:
            internal_states = [obs.observe(ext) for ext in test_states]
        else:
            internal_states = list(obs.observation_log.get_internal_states())

        pairs: List[tuple] = []
        for internal in internal_states:
            prev = obs.internal_state
            obs.internal_state = internal
            self_repr = obs.self_observe()
            obs.internal_state = prev
            if self_repr is None:
                continue
            for dof in obs.internal_dofs:
                unc = obs.estimate_uncertainty(dof)
                true_val = internal.get_value(dof)
                pred_val = self_repr.get_value(dof)
                if true_val is not None and pred_val is not None:
                    pairs.append((unc, abs(float(true_val) - float(pred_val))))
        return pairs

    def _evaluate_metacognition(self) -> float:
        """
        Evaluate meta-cognitive capability via behavioral test.

        Framework §5.2: meta-cognition = depth ≥ 2, i.e. the self-model
        can model *itself*.  We measure three behavioral axes:

        1. Self-observation accuracy — does self_observe() actually track
           internal state?  (0.4 weight)
        2. Recursive depth contribution — depth ≥ 2 shows meta-level (0.3)
        3. Self-model prediction stability — repeated self-observations on
           the same state should be consistent (0.3)

        Returns:
            Meta-cognitive score [0, 1]
        """
        obs = self.observer
        if obs.self_model is None:
            return 0.0

        # --- axis 1: self-observation accuracy (behavioral) ---
        accuracy = self._self_observation_accuracy()

        # --- axis 2: recursive depth ---
        depth = obs.recursive_depth()
        depth_score = min(depth / 2.0, 1.0)  # full credit at depth 2+

        # --- axis 3: prediction stability ---
        stability = self._self_observation_stability()

        return float(np.clip(
            0.4 * accuracy + 0.3 * depth_score + 0.3 * stability,
            0.0, 1.0,
        ))

    def _self_observation_accuracy(self) -> float:
        """How well does self_observe() track actual internal state?"""
        obs = self.observer
        internal_states = obs.observation_log.get_internal_states()
        if not internal_states:
            if obs.internal_state is None:
                return 0.0
            internal_states = [obs.internal_state]

        accuracies = []
        for internal in internal_states[-20:]:  # check last 20
            prev = obs.internal_state
            obs.internal_state = internal
            self_repr = obs.self_observe()
            obs.internal_state = prev
            if self_repr is None:
                continue
            dist = internal.distance_to(self_repr)
            accuracies.append(1.0 / (1.0 + dist))

        return float(np.mean(accuracies)) if accuracies else 0.0

    def _self_observation_stability(self) -> float:
        """Are repeated self-observations on the same state consistent?"""
        obs = self.observer
        if obs.internal_state is None:
            return 0.5  # neutral — no data

        results = []
        for _ in range(5):
            self_repr = obs.self_observe()
            if self_repr is not None:
                results.append(self_repr)

        if len(results) < 2:
            return 0.0

        # Measure pairwise distances — lower spread = higher stability
        distances = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                distances.append(results[i].distance_to(results[j]))

        if not distances:
            return 1.0
        mean_dist = float(np.mean(distances))
        return 1.0 / (1.0 + mean_dist)

    def _evaluate_limitation_awareness(self, test_states: List[State] = None) -> float:
        """
        Evaluate awareness of limitations via easy/hard input split.

        Framework §8.3: "knows when to ask for help; degrades gracefully
        under uncertainty."

        Approach: split inputs into "easy" (near distribution center) and
        "hard" (at distribution edges / out-of-distribution).  An aware
        observer should report *higher* uncertainty on hard inputs.

        If no test_states, uses the observation log and synthesizes hard
        states by pushing values to DoF extremes.

        Returns:
            Awareness score [0, 1]
        """
        obs = self.observer
        if obs.self_model is None:
            return 0.0

        easy_states, hard_states = self._split_easy_hard(test_states)
        if not easy_states or not hard_states:
            return 0.5  # not enough data to judge

        easy_unc = self._mean_uncertainty(easy_states)
        hard_unc = self._mean_uncertainty(hard_states)

        if easy_unc < 1e-12 and hard_unc < 1e-12:
            return 0.0  # flat uncertainty — no awareness

        # Good awareness: hard_unc > easy_unc
        if hard_unc <= easy_unc:
            return 0.0

        # Score: how much *more* uncertain on hard inputs
        # ratio = hard / easy; cap contribution at 3×
        ratio = hard_unc / max(easy_unc, 1e-12)
        return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))

    def _split_easy_hard(
        self, test_states: List[State] = None,
    ) -> tuple:
        """Split inputs into easy (central) and hard (extreme) sets."""
        obs = self.observer

        if test_states and len(test_states) >= 4:
            # Use provided states: sort by distance from mean, split in half
            vectors = []
            for s in test_states:
                v = s.to_vector(obs.external_dofs)
                vectors.append(np.array(v))
            mean_v = np.mean(vectors, axis=0)
            dists = [float(np.linalg.norm(v - mean_v)) for v in vectors]
            median_dist = float(np.median(dists))
            easy = [s for s, d in zip(test_states, dists) if d <= median_dist]
            hard = [s for s, d in zip(test_states, dists) if d > median_dist]
            return easy, hard

        # Synthesize from observation log
        log_pairs = list(obs.observation_log)
        if len(log_pairs) < 4:
            return [], []

        ext_states = [p.external_state for p in log_pairs]
        vectors = [np.array(s.to_vector(obs.external_dofs)) for s in ext_states]
        mean_v = np.mean(vectors, axis=0)
        dists = [float(np.linalg.norm(v - mean_v)) for v in vectors]
        median_dist = float(np.median(dists))
        easy = [s for s, d in zip(ext_states, dists) if d <= median_dist]
        hard = [s for s, d in zip(ext_states, dists) if d > median_dist]
        return easy, hard

    def _mean_uncertainty(self, states: List[State]) -> float:
        """Average stated uncertainty across states and internal DoFs."""
        obs = self.observer
        total = 0.0
        count = 0
        for ext in states:
            obs.observe(ext)
            for dof in obs.internal_dofs:
                total += obs.estimate_uncertainty(dof)
                count += 1
        return total / count if count > 0 else 0.0


def compare_observers(observers: List[Observer], test_states: List[State] = None) -> Dict[str, ConsciousnessMetrics]:
    """
    Compare consciousness metrics across multiple observers.

    Args:
        observers: List of observers to compare
        test_states: Optional test states for evaluation

    Returns:
        Dictionary mapping observer names to metrics

    Example:
        >>> observers = [observer1, observer2, observer3]
        >>> comparison = compare_observers(observers, test_states)
        >>> for name, metrics in comparison.items():
        ...     print(f"{name}: {metrics.consciousness_score():.2f}")
    """
    results = {}

    for observer in observers:
        evaluator = ConsciousnessEvaluator(observer)
        metrics = evaluator.evaluate(test_states)
        results[observer.name] = metrics

    return results


def rank_by_consciousness(observers: List[Observer], test_states: List[State] = None) -> List[tuple[Observer, float]]:
    """
    Rank observers by consciousness score.

    Args:
        observers: List of observers
        test_states: Optional test states

    Returns:
        List of (observer, score) tuples, sorted by score (descending)

    Example:
        >>> ranked = rank_by_consciousness([obs1, obs2, obs3])
        >>> print(f"Most conscious: {ranked[0][0].name} ({ranked[0][1]:.2f})")
    """
    scores = []

    for observer in observers:
        evaluator = ConsciousnessEvaluator(observer)
        metrics = evaluator.evaluate(test_states)
        scores.append((observer, metrics.consciousness_score()))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores
