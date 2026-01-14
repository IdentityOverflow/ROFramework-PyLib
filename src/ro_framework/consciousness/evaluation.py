"""
Consciousness evaluation based on structural criteria.

This module evaluates consciousness as a structural property:
recursive self-modeling with bounded error, not phenomenal experience.
"""

from dataclasses import dataclass
from typing import Dict, List, Any

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

    def consciousness_score(self) -> float:
        """
        Compute overall consciousness score [0, 1].

        Combines multiple metrics with weights based on importance.

        Returns:
            Score from 0 (no consciousness) to 1 (full consciousness)
        """
        if not self.has_self_model:
            return 0.0

        # Weighted combination
        score = 0.0
        score += 0.2 * min(self.recursive_depth / 3.0, 1.0)  # Depth (cap at 3)
        score += 0.25 * self.self_accuracy  # Accuracy
        score += 0.15 * self.architectural_similarity  # Similarity
        score += 0.15 * (1.0 - self.calibration_error)  # Calibration (inverted)
        score += 0.15 * self.meta_cognitive_capability  # Meta-cognition
        score += 0.10 * self.limitation_awareness  # Awareness of limits

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
        Evaluate similarity between world model and self-model architectures.

        For consciousness, both should have similar structure.

        Returns:
            Similarity score [0, 1]
        """
        if self.observer.self_model is None:
            return 0.0

        # Check if both are same type
        world_type = type(self.observer.world_model).__name__
        self_type = type(self.observer.self_model).__name__

        if world_type == self_type:
            return 1.0

        # Partial credit for similar types
        if "Neural" in world_type and "Neural" in self_type:
            return 0.7

        return 0.3

    def _evaluate_calibration(self, test_states: List[State] = None) -> float:
        """
        Evaluate calibration: does confidence match accuracy?

        Good calibration means when the model says 90% confident,
        it's correct 90% of the time.

        Args:
            test_states: Test states

        Returns:
            Calibration error [0, 1] (lower is better)
        """
        # Simplified implementation - full version would:
        # 1. Get uncertainty estimates from model
        # 2. Measure actual accuracy
        # 3. Compute calibration error (e.g., ECE)

        # Placeholder: Check if observer can estimate uncertainty
        if hasattr(self.observer, "estimate_uncertainty"):
            return 0.2  # Assume reasonable calibration if method exists

        return 0.5  # Unknown

    def _evaluate_metacognition(self) -> float:
        """
        Evaluate meta-cognitive capability.

        Can the observer reason about its own reasoning?
        Can it identify sources of errors?

        Returns:
            Meta-cognitive score [0, 1]
        """
        # Check for meta-cognitive capabilities
        score = 0.0

        # Has uncertainty estimation?
        if hasattr(self.observer, "estimate_uncertainty"):
            score += 0.3

        # Has memory (needed for learning from mistakes)?
        if self.observer.has_memory():
            score += 0.3

        # Has self-model (can reflect on internal states)?
        if self.observer.self_model is not None:
            score += 0.4

        return float(score)

    def _evaluate_limitation_awareness(self, test_states: List[State] = None) -> float:
        """
        Evaluate awareness of own limitations.

        Does the observer know what it doesn't know?
        Does uncertainty increase on ambiguous inputs?

        Args:
            test_states: Test states (should include ambiguous cases)

        Returns:
            Awareness score [0, 1]
        """
        # Simplified: Check if observer can report uncertainty
        if not hasattr(self.observer, "estimate_uncertainty"):
            return 0.0

        # If we have test states, check if uncertainty varies appropriately
        if test_states and len(test_states) > 1:
            uncertainties = []
            for state in test_states:
                # Get uncertainty for first internal DoF (simplified)
                if self.observer.internal_dofs:
                    unc = self.observer.estimate_uncertainty(self.observer.internal_dofs[0])
                    uncertainties.append(unc)

            if uncertainties:
                # Good awareness means uncertainty varies (not always high or low)
                var = np.var(uncertainties)
                score = min(var / 0.01, 1.0)  # Normalize
                return float(score)

        # Default: Has mechanism for uncertainty
        return 0.5


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
