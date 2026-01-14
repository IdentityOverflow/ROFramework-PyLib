"""
Observer implementation.

An observer is a configuration within the Block Universe characterized by:
- Boundary (internal/external DoF partition)
- Mapping functions (external → internal)
- Resolution (per-DoF finite granularity)
- Memory (correlation structure across temporal DoF)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ro_framework.core.dof import DoF
from ro_framework.core.state import State
from ro_framework.observer.mapping import MappingFunction


@dataclass
class Observer:
    """
    Observer: A configuration within the Block characterized by:
    - Boundary (B): Partition of DoFs into internal and external
    - Mapping (M): External → Internal function
    - Resolution (R): Per-DoF finite granularity
    - Memory (Mem): Correlation structure across temporal DoF

    Mathematical notation: O = (B, M, R, Mem)

    The observer maps external configurations to internal configurations,
    maintaining finite resolution and potentially memory structure.

    Attributes:
        name: Identifier for this observer
        internal_dofs: DoFs internal to the observer
        external_dofs: DoFs external to the observer
        world_model: Mapping from external to internal DoFs
        self_model: Optional mapping from internal to internal (for consciousness)
        resolution: Per-DoF resolution limits
        temporal_dof: Optional temporal DoF for memory tracking
        memory_buffer: Finite-length history of internal states
        internal_state: Current internal state

    Example:
        >>> # Define DoFs
        >>> external_dof = PolarDoF(name="sensor", pole_negative=-1, pole_positive=1)
        >>> internal_dof = PolarDoF(name="latent", pole_negative=-10, pole_positive=10)
        >>>
        >>> # Create mapping
        >>> world_model = IdentityMapping(
        ...     input_dofs=[external_dof],
        ...     output_dofs=[internal_dof]
        ... )
        >>>
        >>> # Create observer
        >>> observer = Observer(
        ...     name="simple_observer",
        ...     internal_dofs=[internal_dof],
        ...     external_dofs=[external_dof],
        ...     world_model=world_model
        ... )
        >>>
        >>> # Observe
        >>> external_state = State(values={external_dof: 0.5})
        >>> internal_state = observer.observe(external_state)
    """

    name: str
    internal_dofs: List[DoF]
    external_dofs: List[DoF]
    world_model: MappingFunction
    self_model: Optional[MappingFunction] = None
    resolution: Dict[DoF, float] = field(default_factory=dict)
    temporal_dof: Optional[DoF] = None
    memory_buffer: List[State] = field(default_factory=list)
    memory_capacity: int = 1000
    internal_state: Optional[State] = None

    def __post_init__(self) -> None:
        """Initialize resolution dict if empty."""
        if not self.resolution:
            self.resolution = {dof: 1e-6 for dof in self.internal_dofs}

    def observe(self, external_state: State) -> State:
        """
        Perform observation: map external DoFs to internal DoFs.

        This is the core mechanism of observation in the framework.
        The observer applies its world model to translate external
        sensory data into internal representations.

        Args:
            external_state: State with values on external DoFs

        Returns:
            Internal state with values on internal DoFs

        Example:
            >>> external = State(values={vision_dof: image_data})
            >>> internal = observer.observe(external)
        """
        # Apply world model mapping
        internal_state = self.world_model(external_state)

        # Update internal state
        self.internal_state = internal_state

        # Store in memory (correlation across temporal DoF)
        if self.temporal_dof is not None:
            self.memory_buffer.append(internal_state)

            # Maintain memory capacity
            if len(self.memory_buffer) > self.memory_capacity:
                self.memory_buffer.pop(0)

        return internal_state

    def self_observe(self) -> Optional[State]:
        """
        Perform self-observation: map internal DoFs to internal DoFs.

        This is the recursive self-modeling that defines consciousness
        in the structural sense. The observer applies its self-model to
        its own internal state.

        Returns:
            Self-representation state, or None if no self-model

        Example:
            >>> self_repr = observer.self_observe()
            >>> if self_repr is not None:
            ...     print(f"Observer is self-aware: {self_repr}")
        """
        if self.self_model is None:
            return None

        if self.internal_state is None:
            return None

        # Apply self-model mapping (recursion!)
        self_representation = self.self_model(self.internal_state)

        return self_representation

    def get_resolution(self, dof: DoF) -> float:
        """
        Get resolution limit for a specific DoF.

        Args:
            dof: DoF to get resolution for

        Returns:
            Resolution limit (minimum distinguishable difference)
        """
        return self.resolution.get(dof, 1e-6)

    def has_memory(self, threshold: float = 0.5, max_lag: int = 5) -> bool:
        """
        Check if observer has memory structure using temporal correlation.

        Memory exists if significant temporal correlation is detected across
        the temporal DoF, beyond what would be expected from random noise.
        Uses the correlation module's temporal_correlation() function.

        Args:
            threshold: Minimum correlation to consider "significant"
            max_lag: Maximum temporal lag to check

        Returns:
            True if memory structure detected

        Example:
            >>> if observer.has_memory():
            ...     print("Observer has memory!")
        """
        from ro_framework.correlation.measures import temporal_correlation

        if self.temporal_dof is None or len(self.memory_buffer) < 3:
            return False

        # Check for significant temporal correlation in any internal DoF
        for dof in self.internal_dofs:
            try:
                # Check multiple lags
                for lag in range(1, min(max_lag + 1, len(self.memory_buffer) // 2)):
                    corr = temporal_correlation(
                        states=self.memory_buffer,
                        dof=dof,
                        temporal_dof=self.temporal_dof,
                        lag=lag
                    )

                    if abs(corr) > threshold:
                        return True

            except (ValueError, ZeroDivisionError):
                # Not enough variance or data for this DoF
                continue

        return False

    def is_conscious(self, threshold: float = 0.5, test_states: List[State] = None) -> bool:
        """
        Check if observer is structurally conscious.

        Uses ConsciousnessEvaluator to assess multiple structural properties:
        - Has self-model
        - Recursive depth
        - Self-accuracy
        - Architectural similarity between world and self models
        - Calibration quality
        - Meta-cognitive capability
        - Limitation awareness

        Args:
            threshold: Minimum consciousness score to be considered conscious (default: 0.5)
            test_states: Optional test states for evaluation

        Returns:
            True if consciousness score exceeds threshold

        Example:
            >>> if observer.is_conscious():
            ...     print("Observer is structurally conscious!")
            >>> # With custom threshold
            >>> if observer.is_conscious(threshold=0.7):
            ...     print("High consciousness score!")
        """
        from ro_framework.consciousness.evaluation import ConsciousnessEvaluator

        evaluator = ConsciousnessEvaluator(self)
        metrics = evaluator.evaluate(test_states)
        score = metrics.consciousness_score()

        return score >= threshold

    def recursive_depth(self) -> int:
        """
        Compute depth of recursive self-modeling.

        - Depth 0: No self-model
        - Depth 1: Self-model exists (internal → internal)
        - Depth 2+: Meta-models exist (model can represent its own modeling process)

        Checks if the observer can represent its own modeling process by examining
        whether internal DoFs can encode information about the world_model and self_model.

        Returns:
            Recursive depth

        Example:
            >>> depth = observer.recursive_depth()
            >>> print(f"Recursive depth: {depth}")
        """
        if self.self_model is None:
            return 0

        # Depth 1: Has self-model
        depth = 1

        # Check for depth 2: Can the internal state represent the modeling process itself?
        # This requires checking if internal_dofs have enough capacity to encode
        # information about the models
        #
        # Heuristic: If internal state dimension >= 2 * external state dimension,
        # it potentially has capacity to represent both world state AND the modeling process

        if len(self.internal_dofs) >= 2 * len(self.external_dofs):
            # Has capacity for meta-representation
            depth = 2

        # Note: Depth 3+ would require explicit meta-meta-models or demonstrated
        # capability to reason about reasoning about reasoning, which requires
        # more sophisticated checks (e.g., training/testing on meta-cognitive tasks)

        return depth

    def get_consciousness_metrics(self, test_states: List[State] = None):
        """
        Get full consciousness evaluation metrics.

        Uses ConsciousnessEvaluator to compute all structural consciousness metrics:
        - has_self_model: Whether self-model exists
        - recursive_depth: Depth of recursive self-modeling
        - self_accuracy: How accurately self-model represents internal state
        - architectural_similarity: Similarity between world and self models
        - calibration_error: |confidence - accuracy|
        - meta_cognitive_capability: Can reason about own reasoning
        - limitation_awareness: Knows what it doesn't know

        Args:
            test_states: Optional test states for evaluation

        Returns:
            ConsciousnessMetrics with all measurements and overall score

        Example:
            >>> metrics = observer.get_consciousness_metrics()
            >>> print(f"Consciousness score: {metrics.consciousness_score():.2f}")
            >>> print(f"Recursive depth: {metrics.recursive_depth}")
            >>> print(f"Self-accuracy: {metrics.self_accuracy:.2f}")
        """
        from ro_framework.consciousness.evaluation import ConsciousnessEvaluator

        evaluator = ConsciousnessEvaluator(self)
        return evaluator.evaluate(test_states)

    def know(
        self,
        external_dof: DoF,
        threshold: float = 0.7,
        min_samples: int = 10,
    ) -> bool:
        """
        Check if observer has knowledge of an external DoF.

        Knowledge requires:
        1. High correlation between external and internal DoFs
        2. Stability across contexts
        3. Bounded error (accuracy)
        4. Calibration (confidence matches accuracy)

        Args:
            external_dof: External DoF to check knowledge of
            threshold: Minimum correlation for "knowledge"
            min_samples: Minimum number of observations required

        Returns:
            True if knowledge criteria are met

        Example:
            >>> if observer.know(vision_dof):
            ...     print("Observer knows about vision!")
        """
        if len(self.memory_buffer) < min_samples:
            return False

        # This is a placeholder implementation
        # Full implementation would:
        # 1. Find which internal DoF(s) correlate with external_dof
        # 2. Compute correlation strength
        # 3. Check calibration
        # 4. Verify stability

        # For now, return False (unknown)
        return False

    def estimate_uncertainty(self, dof: DoF) -> float:
        """
        Estimate uncertainty in current knowledge of a DoF.

        Uncertainty comes from:
        - Resolution limits (structural)
        - Measurement noise (physical)
        - Model uncertainty (epistemic)

        Args:
            dof: DoF to estimate uncertainty for

        Returns:
            Uncertainty estimate

        Example:
            >>> uncertainty = observer.estimate_uncertainty(latent_dof)
            >>> print(f"Uncertainty: {uncertainty:.4f}")
        """
        if self.internal_state is None:
            return 1.0  # Maximum uncertainty

        # Get resolution-based uncertainty
        resolution_uncertainty = self.get_resolution(dof)

        # Add model uncertainty if available
        if hasattr(self.world_model, "compute_uncertainty"):
            model_uncertainty_dict = self.world_model.compute_uncertainty(self.internal_state)
            model_uncertainty = model_uncertainty_dict.get(dof, 0.0)
        else:
            model_uncertainty = 0.0

        # Combine uncertainties (simplified - should use proper uncertainty propagation)
        total_uncertainty = resolution_uncertainty + model_uncertainty

        return total_uncertainty

    def analyze_memory_structure(self, max_lag: int = 10) -> Dict[DoF, List[float]]:
        """
        Analyze memory structure using temporal correlation analysis.

        Returns temporal correlations for each internal DoF, showing
        how strongly the DoF's current value predicts future values.

        Args:
            max_lag: Maximum temporal lag to analyze

        Returns:
            Dictionary mapping each DoF to its temporal correlation profile

        Example:
            >>> memory_analysis = observer.analyze_memory_structure()
            >>> for dof, correlations in memory_analysis.items():
            ...     print(f"{dof.name}: {correlations}")
        """
        from ro_framework.correlation.measures import temporal_correlation

        if self.temporal_dof is None or len(self.memory_buffer) < 3:
            return {}

        analysis = {}

        for dof in self.internal_dofs:
            try:
                correlations = []
                for lag in range(1, min(max_lag + 1, len(self.memory_buffer) // 2)):
                    corr = temporal_correlation(
                        states=self.memory_buffer,
                        dof=dof,
                        temporal_dof=self.temporal_dof,
                        lag=lag
                    )
                    correlations.append(corr)
                analysis[dof] = correlations
            except (ValueError, ZeroDivisionError):
                # Not enough variance or data
                analysis[dof] = []

        return analysis

    def get_memory_correlations(self, dof1: DoF, dof2: DoF) -> float:
        """
        Compute correlation between two DoFs across memory.

        Uses Pearson correlation from the correlation module to measure
        how two internal DoFs co-vary over time.

        Args:
            dof1: First DoF
            dof2: Second DoF

        Returns:
            Correlation coefficient between -1 and 1

        Example:
            >>> corr = observer.get_memory_correlations(latent1, latent2)
            >>> print(f"Correlation: {corr:.3f}")
        """
        from ro_framework.correlation.measures import pearson_correlation

        if len(self.memory_buffer) < 2:
            return 0.0

        try:
            return pearson_correlation(self.memory_buffer, dof1, dof2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def clear_memory(self) -> None:
        """
        Clear the memory buffer.

        Useful for resetting the observer or managing memory usage.

        Example:
            >>> observer.clear_memory()
        """
        self.memory_buffer.clear()

    def get_memory_size(self) -> int:
        """
        Get current size of memory buffer.

        Returns:
            Number of states in memory

        Example:
            >>> size = observer.get_memory_size()
            >>> print(f"Memory size: {size}")
        """
        return len(self.memory_buffer)

    def __repr__(self) -> str:
        """String representation of observer."""
        return (
            f"Observer(name='{self.name}', "
            f"internal_dofs={len(self.internal_dofs)}, "
            f"external_dofs={len(self.external_dofs)}, "
            f"conscious={self.is_conscious()}, "
            f"memory_size={self.get_memory_size()})"
        )
