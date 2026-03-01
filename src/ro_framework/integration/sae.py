"""
SAE integration for knowledge assessment on real models.

Wraps a TransformerLens model + SAELens sparse autoencoder as an Observer,
enabling graded knowledge assessment K(d_ext) = (ρ, ε, σ, C) on real
learned features.

External DoFs are user-defined labels (sentiment, topic, etc.).
Internal DoFs are SAE feature activations at a specific layer.

Requires: pip install ro-framework[sae]
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ro_framework.core.dof import DoF, ScalarDoF
from ro_framework.core.state import State
from ro_framework.knowledge.assessment import KnowledgeAssessment, compute_knowledge
from ro_framework.observer.observer import ObservationLog, ObservationPair, Observer

try:
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    _SAE_AVAILABLE = True
except ImportError:
    _SAE_AVAILABLE = False


def _require_sae() -> None:
    """Raise ImportError if SAE dependencies are not installed."""
    if not _SAE_AVAILABLE:
        raise ImportError(
            "SAE integration requires transformer-lens and sae-lens. "
            "Install with: pip install ro-framework[sae]"
        )


def create_feature_dofs(
    n_features: int,
    prefix: str = "sae_feat",
) -> List[ScalarDoF]:
    """Create ScalarDoFs for SAE feature activations.

    SAE features are non-negative (ReLU/TopK activation), so ScalarDoF
    (magnitude-only) is the appropriate DoF type.

    Args:
        n_features: Number of SAE features (dictionary size).
        prefix: Name prefix for each DoF.

    Returns:
        List of n_features ScalarDoFs.
    """
    return [
        ScalarDoF(name=f"{prefix}_{i}", description=f"SAE feature {i}")
        for i in range(n_features)
    ]


@dataclass
class FeatureKnowledge:
    """Knowledge assessment for a single SAE feature relative to a label."""

    feature_dof: DoF
    assessment: KnowledgeAssessment


class SAEObserver:
    """Observer wrapping a TransformerLens model + SAELens SAE.

    Bridges the RO Framework to real models by:
    1. Running text through a language model (e.g., GPT-2)
    2. Extracting activations at a specific layer
    3. Encoding activations through a pre-trained SAE
    4. Recording (label, SAE features) observation pairs
    5. Assessing knowledge: which SAE features track which labels?

    The Observer's external DoFs are user-provided labels (abstract properties
    like sentiment, topic, is_code). Internal DoFs are SAE feature activations.
    Knowledge assessment K(d_ext) tells you which features track each label
    with what correlation, bias, noise, and calibration.

    Args:
        model: TransformerLens HookedTransformer model.
        sae: SAELens SAE (pre-trained sparse autoencoder).
        hook_point: Model hook point matching the SAE (e.g., "blocks.8.hook_resid_pre").
        label_dofs: User-defined external DoFs for labels.
        name: Observer name.
        feature_dofs: Optional pre-created feature DoFs. Auto-created if None.
        aggregation: How to aggregate per-token activations to per-text.
            "mean" (default), "last", or "max".
        top_k_features: If set, only track the top-K most frequently active
            features (reduces memory for large SAEs). None = track all.
        log_capacity: Maximum observation pairs to retain.
        device: PyTorch device for inference.
    """

    def __init__(
        self,
        model: Any,  # HookedTransformer, typed as Any for import flexibility
        sae: Any,  # SAE
        hook_point: str,
        label_dofs: List[DoF],
        name: str = "sae_observer",
        feature_dofs: Optional[List[ScalarDoF]] = None,
        aggregation: str = "mean",
        top_k_features: Optional[int] = None,
        log_capacity: int = 1000,
        device: str = "cpu",
    ):
        _require_sae()

        if aggregation not in ("mean", "last", "max"):
            raise ValueError(f"aggregation must be 'mean', 'last', or 'max', got '{aggregation}'")

        self.model = model
        self.sae = sae
        self.hook_point = hook_point
        self.label_dofs = list(label_dofs)
        self.name = name
        self.aggregation = aggregation
        self.top_k_features = top_k_features
        self.device = device

        # Determine which layer to stop at for efficiency
        self._stop_at_layer = self._parse_layer(hook_point) + 1

        # Create feature DoFs
        n_features = sae.cfg.d_sae
        if top_k_features is not None and top_k_features < n_features:
            n_features = top_k_features
        if feature_dofs is not None:
            self.feature_dofs = list(feature_dofs)
        else:
            self.feature_dofs = create_feature_dofs(n_features)

        self._n_sae_features = sae.cfg.d_sae
        self._n_tracked_features = len(self.feature_dofs)

        # Build the internal Observer
        # We bypass world_model by appending pairs directly to the observation log
        self._observer = Observer(
            name=name,
            internal_dofs=self.feature_dofs,
            external_dofs=self.label_dofs,
            world_model=_IdentityMapping(self.label_dofs, self.feature_dofs),
            log_capacity=log_capacity,
        )

        # Track which SAE feature indices to keep (for top_k filtering)
        self._feature_indices: Optional[np.ndarray] = None
        if top_k_features is not None:
            # Will be set after first batch based on activation frequency
            self._feature_indices = None
            self._activation_counts = np.zeros(sae.cfg.d_sae)
            self._warmup_samples = 0
            self._warmup_complete = False
        else:
            self._warmup_complete = True

    @staticmethod
    def _parse_layer(hook_point: str) -> int:
        """Extract layer number from hook point string like 'blocks.8.hook_resid_pre'."""
        parts = hook_point.split(".")
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
        raise ValueError(f"Could not parse layer number from hook_point: {hook_point}")

    def _encode_text(self, text: str) -> np.ndarray:
        """Run text through model + SAE, return aggregated feature activations.

        Returns:
            1D numpy array of shape (n_sae_features,) with feature activations.
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                text, stop_at_layer=self._stop_at_layer
            )
            activations = cache[self.hook_point]  # [batch, seq, d_model]

            # Aggregate across sequence dimension
            if activations.dim() == 3:
                activations = activations[0]  # Remove batch dim → [seq, d_model]

            if self.aggregation == "mean":
                aggregated = activations.mean(dim=0)  # [d_model]
            elif self.aggregation == "last":
                aggregated = activations[-1]  # [d_model]
            elif self.aggregation == "max":
                aggregated = activations.max(dim=0).values  # [d_model]

            # Encode through SAE
            feature_acts = self.sae.encode(aggregated.unsqueeze(0))  # [1, d_sae]
            feature_acts = feature_acts.squeeze(0)  # [d_sae]

        return feature_acts.cpu().numpy().astype(np.float64)

    def _select_features(self, all_features: np.ndarray) -> np.ndarray:
        """Select tracked features (top-K or all).

        Args:
            all_features: Full SAE feature vector of shape (d_sae,).

        Returns:
            Feature vector of shape (n_tracked_features,).
        """
        if self._feature_indices is not None:
            return all_features[self._feature_indices]
        if self.top_k_features is not None and not self._warmup_complete:
            # During warmup, return all features (we'll filter later)
            return all_features[:self._n_tracked_features]
        return all_features

    def _update_warmup(self, all_features: np.ndarray) -> None:
        """Track activation frequency during warmup to determine top-K features."""
        if self._warmup_complete:
            return
        self._activation_counts += (all_features > 0).astype(np.float64)
        self._warmup_samples += 1

        # After 50 samples, select top-K most frequently active features
        if self._warmup_samples >= 50:
            top_indices = np.argsort(self._activation_counts)[-self.top_k_features:]
            self._feature_indices = np.sort(top_indices)
            self._warmup_complete = True

    def observe_text(
        self,
        text: str,
        labels: Dict[DoF, float],
    ) -> State:
        """Feed text through model + SAE, record observation pair with labels.

        Args:
            text: Input text to process.
            labels: Dict mapping label DoFs to their values for this text.
                All label_dofs must be present.

        Returns:
            Internal state (SAE feature activations).

        Raises:
            ValueError: If labels are missing declared label DoFs.
        """
        missing = [d for d in self.label_dofs if d not in labels]
        if missing:
            raise ValueError(f"Missing labels for DoFs: {[d.name for d in missing]}")

        # Run model + SAE
        all_features = self._encode_text(text)

        # Update warmup if needed
        if not self._warmup_complete:
            self._update_warmup(all_features)

        # Select tracked features
        features = self._select_features(all_features)

        # Create states
        external_state = State(values={dof: labels[dof] for dof in self.label_dofs})
        internal_state = State(
            values={dof: float(features[i]) for i, dof in enumerate(self.feature_dofs)}
        )

        # Record observation pair directly (bypass world_model)
        self._observer.observation_log.append(ObservationPair(
            external_state=external_state,
            internal_state=internal_state,
            timestamp=float(len(self._observer.observation_log)),
        ))

        self._observer.internal_state = internal_state
        return internal_state

    def observe_texts(
        self,
        texts: List[str],
        labels_list: List[Dict[DoF, float]],
    ) -> List[State]:
        """Batch version of observe_text.

        Processes texts sequentially (model inference is the bottleneck,
        not the observation recording).

        Args:
            texts: List of input texts.
            labels_list: List of label dicts, one per text.

        Returns:
            List of internal states.
        """
        if len(texts) != len(labels_list):
            raise ValueError(
                f"texts and labels_list must have same length, "
                f"got {len(texts)} and {len(labels_list)}"
            )
        return [self.observe_text(t, l) for t, l in zip(texts, labels_list)]

    def assess_knowledge(
        self, label_dof: DoF, min_samples: int = 10, max_features: int = 10,
    ) -> Optional[KnowledgeAssessment]:
        """Assess how well SAE features track a specific label.

        Uses multiple regression with top-k features jointly, so ρ reflects
        the observer's *combined* knowledge — not just one feature's tracking.
        This is important for distributed representations where no single SAE
        feature captures the full label (e.g., sentiment, question syntax).

        Args:
            label_dof: The external DoF (label) to assess knowledge of.
            min_samples: Minimum observations required.
            max_features: Maximum features for joint regression (default 10).
                Use 1 for single-feature assessment.

        Returns:
            KnowledgeAssessment or None if insufficient data.
        """
        return compute_knowledge(
            self._observer.observation_log,
            label_dof,
            self.feature_dofs,
            min_samples,
            max_features,
        )

    def top_features_for(
        self,
        label_dof: DoF,
        n: int = 10,
        min_samples: int = 10,
    ) -> List[FeatureKnowledge]:
        """Find the top-N SAE features most correlated with a label.

        Computes correlation between the label and each SAE feature,
        returns the N features with highest |correlation|.

        Args:
            label_dof: The external DoF (label) to find features for.
            n: Number of top features to return.
            min_samples: Minimum observations required.

        Returns:
            List of FeatureKnowledge, sorted by |correlation| descending.
        """
        log = self._observer.observation_log
        if len(log) < min_samples:
            return []

        results = []
        for feat_dof in self.feature_dofs:
            k = compute_knowledge(log, label_dof, [feat_dof], min_samples)
            if k is not None:
                results.append(FeatureKnowledge(feature_dof=feat_dof, assessment=k))

        # Sort by absolute correlation, descending
        results.sort(key=lambda fk: abs(fk.assessment.correlation), reverse=True)
        return results[:n]

    @property
    def observer(self) -> Observer:
        """Access the underlying Observer for advanced usage."""
        return self._observer

    @property
    def n_observations(self) -> int:
        """Number of observations recorded so far."""
        return len(self._observer.observation_log)


class _IdentityMapping:
    """Placeholder mapping for SAEObserver.

    The actual mapping (model + SAE) is handled externally in observe_text().
    This exists only to satisfy Observer's world_model requirement.
    """

    def __init__(self, input_dofs: List[DoF], output_dofs: List[DoF]):
        self.input_dofs = input_dofs
        self.output_dofs = output_dofs

    def __call__(self, state: State) -> State:
        # Should never be called directly — SAEObserver bypasses this
        return State(values={dof: 0.0 for dof in self.output_dofs})


def create_multilayer_sae_observers(
    model_name: str,
    release: str,
    layers: List[int],
    label_dofs: List[DoF],
    hook_pattern: str = "blocks.{layer}.hook_resid_pre",
    aggregation: str = "mean",
    top_k_features: Optional[int] = None,
    log_capacity: int = 1000,
    device: str = "cpu",
) -> Dict[int, SAEObserver]:
    """Create SAEObservers at multiple layers sharing one model instance.

    This is the recommended way to do multi-layer analysis: load the model
    once, create SAE observers at each layer, and compare knowledge profiles
    across depth.

    Args:
        model_name: TransformerLens model name (e.g., "gpt2-small").
        release: SAELens release name (e.g., "gpt2-small-res-jb").
        layers: List of layer indices to create observers for.
        label_dofs: User-defined external DoFs for labels.
        hook_pattern: Hook point pattern with {layer} placeholder.
        aggregation: Aggregation method for per-token activations.
        top_k_features: If set, only track top-K most active features per layer.
        log_capacity: Max observations per observer.
        device: PyTorch device.

    Returns:
        Dict mapping layer index to SAEObserver.
    """
    _require_sae()

    model = HookedTransformer.from_pretrained(model_name, device=device)

    observers = {}
    for layer in layers:
        hook_point = hook_pattern.format(layer=layer)
        sae = SAE.from_pretrained(
            release=release,
            sae_id=hook_point,
            device=device,
        )
        # SAE.from_pretrained returns (sae, cfg, sparsity) tuple in some versions
        if isinstance(sae, tuple):
            sae = sae[0]

        observers[layer] = SAEObserver(
            model=model,
            sae=sae,
            hook_point=hook_point,
            label_dofs=label_dofs,
            name=f"sae_observer_L{layer}",
            aggregation=aggregation,
            top_k_features=top_k_features,
            log_capacity=log_capacity,
            device=device,
        )

    return observers
