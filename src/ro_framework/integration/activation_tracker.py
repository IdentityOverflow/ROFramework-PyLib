"""
Online feature discovery via activation tracking and PCA.

Discovers emerging features during training by tracking which principal
directions in activation space are being amplified and stabilized.  A
direction with growing, stable variance is a candidate monosemantic
feature — a resonance that's locking in.

Requires PyTorch.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from ro_framework.core.dof import PolarDoF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedDirection:
    """A principal direction tracked at a single epoch.

    Attributes:
        direction: Unit vector in activation space, shape (hidden_dim,).
        eigenvalue: Variance explained by this direction.
        epoch: Epoch when this snapshot was taken.
        stability: |cos_sim| with the matched direction at the previous epoch.
            None for the first epoch.
        readout_alignment: Cosine similarity with the readout layer's row
            space, in [0, 1].  None if no readout layer was provided.
    """

    direction: np.ndarray
    eigenvalue: float
    epoch: int
    stability: Optional[float] = None
    readout_alignment: Optional[float] = None


@dataclass(frozen=True)
class DirectionSnapshot:
    """All tracked directions at a single epoch.

    Attributes:
        epoch: Training epoch.
        directions: TrackedDirection list, sorted by eigenvalue (largest first).
        explained_variance_ratio: Fraction of total variance each direction
            explains, shape (top_k,).
        total_variance: Trace of the covariance matrix.
        cov_matrix: Optional. The raw covariance matrix at this epoch, used
            for Slow Feature Analysis (SFA).
    """

    epoch: int
    directions: List[TrackedDirection]
    explained_variance_ratio: np.ndarray
    total_variance: float
    cov_matrix: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DiscoveredDoF:
    """A DoF discovered from activation analysis.

    Attributes:
        dof: PolarDoF created for this direction.
        projection: Unit vector to project activations onto this DoF.
        eigenvalue: Variance along this direction (or SFA settle score).
        stability_epochs: Consecutive epochs this direction has been stable.
        source_layer: Name of the layer this was discovered from.
        discovery_method: Method used to find this ("pca" or "sfa").
    """

    dof: PolarDoF
    projection: np.ndarray
    eigenvalue: float
    stability_epochs: int
    source_layer: str
    discovery_method: str = "pca"


# ---------------------------------------------------------------------------
# Direction matching
# ---------------------------------------------------------------------------


def _match_directions(
    prev: np.ndarray,
    curr: np.ndarray,
) -> List[Tuple[int, int, float]]:
    """Match directions between epochs by maximum |cosine similarity|.

    Uses greedy matching: highest similarity pairs first.  Handles the
    sign ambiguity of eigenvectors (a direction and its negation are the
    same direction).

    Args:
        prev: (K, D) matrix of previous epoch directions.
        curr: (K, D) matrix of current epoch directions.

    Returns:
        List of (prev_idx, curr_idx, |cosine_similarity|) tuples.
    """
    sims = np.abs(prev @ curr.T)  # (K_prev, K_curr)
    matches: List[Tuple[int, int, float]] = []
    used_prev: set = set()
    used_curr: set = set()

    flat_order = np.argsort(sims.ravel())[::-1]
    k = min(len(prev), len(curr))
    for flat_idx in flat_order:
        i, j = divmod(int(flat_idx), sims.shape[1])
        if i not in used_prev and j not in used_curr:
            matches.append((i, j, float(sims[i, j])))
            used_prev.add(i)
            used_curr.add(j)
            if len(matches) == k:
                break

    return matches


# ---------------------------------------------------------------------------
# ActivationTracker
# ---------------------------------------------------------------------------


class ActivationTracker:
    """Tracks activations at a PyTorch layer and discovers features via PCA.

    Registers a forward hook on the target layer, collects running
    statistics (Welford's algorithm) during forward passes, then computes
    PCA at the end of each collection window.  No individual activations
    are stored — only the running mean and covariance accumulator.

    Typical usage::

        tracker = ActivationTracker(model, "relu", top_k=10)
        tracker.attach()

        for epoch in range(num_epochs):
            train_one_epoch(model)
            tracker.begin_collection()
            run_eval_forward_passes(model)  # hook fires automatically
            snapshot = tracker.end_collection(epoch)

        discovered = tracker.discover_dofs()
        tracker.detach()

    Args:
        model: PyTorch nn.Module.
        layer_name: Dot-separated path to the target module (e.g. "relu",
            "encoder.layer3").
        top_k: Number of principal directions to track.
        device: Device string for input tensors.
        readout_layer_name: Optional layer whose weight matrix defines
            the task-relevant subspace (e.g. "fc2").
        store_covariance: If True, stores the raw covariance matrix in each
            snapshot. Required for Slow Feature Analysis (SFA).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        top_k: int = 10,
        device: str = "cpu",
        readout_layer_name: Optional[str] = None,
        store_covariance: bool = False,
    ) -> None:
        self._model = model
        self._layer_name = layer_name
        self._top_k = top_k
        self._device = device
        self._readout_layer_name = readout_layer_name
        self._store_covariance = store_covariance

        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._snapshots: List[DirectionSnapshot] = []

        # Direction identity tracking: maps original direction index to its
        # current position in the latest snapshot.  Built up via greedy
        # matching across epochs.
        self._direction_histories: List[List[Tuple[int, float, Optional[float]]]] = []
        # Each inner list is [(epoch, eigenvalue, stability), ...] for one
        # tracked direction identity.

        # Welford running statistics (reset each collection window)
        self._collecting = False
        self._n = 0
        self._mean: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _get_module(self, name: str) -> nn.Module:
        """Resolve a dot-separated module name."""
        mod = self._model
        for part in name.split("."):
            mod = getattr(mod, part)
        return mod

    def attach(self) -> None:
        """Register forward hook on the target layer.  Idempotent."""
        if self._hook_handle is not None:
            return
        target = self._get_module(self._layer_name)
        self._hook_handle = target.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        """Remove forward hook.  Idempotent."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------------------------------------------
    # Collection cycle
    # ------------------------------------------------------------------

    def begin_collection(self) -> None:
        """Start collecting activations for a new epoch.

        Resets running statistics (mean, covariance accumulator, count).
        """
        self._collecting = True
        self._n = 0
        self._mean = None
        self._M2 = None

    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """Forward hook callback.  Updates Welford running statistics."""
        if not self._collecting:
            return

        # Flatten to (batch, features)
        act = output.detach().cpu().numpy()
        if act.ndim == 1:
            act = act.reshape(1, -1)
        elif act.ndim > 2:
            act = act.reshape(act.shape[0], -1)

        d = act.shape[1]
        if self._mean is None:
            self._mean = np.zeros(d, dtype=np.float64)
            self._M2 = np.zeros((d, d), dtype=np.float64)

        for x in act:
            self._n += 1
            delta = x - self._mean
            self._mean += delta / self._n
            delta2 = x - self._mean
            self._M2 += np.outer(delta, delta2)

    def end_collection(self, epoch: int) -> DirectionSnapshot:
        """Finalize collection, compute PCA, return snapshot.

        Raises:
            ValueError: If fewer than 2 samples were collected.
        """
        self._collecting = False

        if self._n < 2:
            raise ValueError(
                f"Need at least 2 samples for PCA, got {self._n}. "
                "Did you call begin_collection() and run forward passes?"
            )

        cov = self._M2 / (self._n - 1)
        total_var = float(np.trace(cov))

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending; reverse for largest-first
        idx = np.argsort(eigenvalues)[::-1]
        k = min(self._top_k, len(eigenvalues))
        top_eigenvalues = eigenvalues[idx[:k]]
        top_eigenvectors = eigenvectors[:, idx[:k]].T  # (k, D), rows are directions

        evr = top_eigenvalues / total_var if total_var > 0 else np.zeros(k)

        # Compute readout alignment if configured
        readout_alignments = self._compute_readout_alignments(top_eigenvectors)

        # Match with previous epoch
        stabilities = self._compute_stabilities(top_eigenvectors)

        directions = []
        for i in range(k):
            directions.append(TrackedDirection(
                direction=top_eigenvectors[i].copy(),
                eigenvalue=float(top_eigenvalues[i]),
                epoch=epoch,
                stability=stabilities[i] if stabilities else None,
                readout_alignment=readout_alignments[i] if readout_alignments else None,
            ))

        snapshot = DirectionSnapshot(
            epoch=epoch,
            directions=directions,
            explained_variance_ratio=evr.copy(),
            total_variance=total_var,
            cov_matrix=cov.copy() if self._store_covariance else None,
        )
        self._snapshots.append(snapshot)

        # Update direction identity histories
        self._update_histories(snapshot, stabilities)

        return snapshot

    def _compute_readout_alignments(
        self, directions: np.ndarray,
    ) -> Optional[List[float]]:
        """Compute alignment of directions with readout layer row space."""
        if self._readout_layer_name is None:
            return None

        readout = self._get_module(self._readout_layer_name)
        if not hasattr(readout, "weight"):
            return None

        W = readout.weight.detach().cpu().numpy()  # (out, hidden)
        max_sv = np.linalg.svd(W, compute_uv=False)[0]
        if max_sv < 1e-12:
            return [0.0] * len(directions)

        alignments = []
        for v in directions:
            alignments.append(float(np.linalg.norm(W @ v) / max_sv))
        return alignments

    def _compute_stabilities(
        self, curr_directions: np.ndarray,
    ) -> Optional[List[float]]:
        """Compute stability of each current direction vs previous epoch."""
        if not self._snapshots:
            return None

        prev = self._snapshots[-1]
        prev_dirs = np.array([d.direction for d in prev.directions])
        matches = _match_directions(prev_dirs, curr_directions)

        # Build stability array indexed by current direction
        stab = [0.0] * len(curr_directions)
        for _, curr_idx, sim in matches:
            stab[curr_idx] = sim
        return stab

    def _update_histories(
        self,
        snapshot: DirectionSnapshot,
        stabilities: Optional[List[float]],
    ) -> None:
        """Update direction identity histories with new snapshot."""
        if len(self._snapshots) <= 1:
            # First snapshot: create one history per direction
            self._direction_histories = []
            for i, d in enumerate(snapshot.directions):
                self._direction_histories.append([
                    (d.epoch, d.eigenvalue, d.stability)
                ])
            return

        prev = self._snapshots[-2]
        prev_dirs = np.array([d.direction for d in prev.directions])
        curr_dirs = np.array([d.direction for d in snapshot.directions])
        matches = _match_directions(prev_dirs, curr_dirs)

        # Map: prev_rank -> history_index
        # We need to know which history each prev direction belongs to.
        # Build from the last snapshot's ordering.
        prev_to_history: Dict[int, int] = {}
        for hist_idx, history in enumerate(self._direction_histories):
            if history:
                last_epoch = history[-1][0]
                if last_epoch == prev.epoch:
                    # Find which prev direction this history corresponds to
                    # by matching the eigenvalue (should be unique enough)
                    last_ev = history[-1][1]
                    for pi, pd in enumerate(prev.directions):
                        if abs(pd.eigenvalue - last_ev) < 1e-12 and pi not in prev_to_history:
                            prev_to_history[pi] = hist_idx
                            break

        matched_curr: set = set()
        for prev_idx, curr_idx, sim in matches:
            if prev_idx in prev_to_history:
                hist_idx = prev_to_history[prev_idx]
                d = snapshot.directions[curr_idx]
                self._direction_histories[hist_idx].append(
                    (d.epoch, d.eigenvalue, d.stability)
                )
                matched_curr.add(curr_idx)

        # New directions that didn't match any previous
        for i, d in enumerate(snapshot.directions):
            if i not in matched_curr:
                self._direction_histories.append([
                    (d.epoch, d.eigenvalue, d.stability)
                ])

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def snapshots(self) -> List[DirectionSnapshot]:
        """Get all snapshots in chronological order."""
        return list(self._snapshots)

    def eigenvalue_trajectory(
        self, direction_idx: int = 0,
    ) -> List[Tuple[int, float]]:
        """Get (epoch, eigenvalue) pairs for a direction identity.

        Args:
            direction_idx: Index into the initial direction ordering
                (0 = originally largest).
        """
        if direction_idx >= len(self._direction_histories):
            return []
        return [(e, ev) for e, ev, _ in self._direction_histories[direction_idx]]

    def stability_trajectory(
        self, direction_idx: int = 0,
    ) -> List[Tuple[int, float]]:
        """Get (epoch, stability) pairs for a direction identity.

        Skips the first epoch (stability is None).
        """
        if direction_idx >= len(self._direction_histories):
            return []
        return [
            (e, s) for e, _, s in self._direction_histories[direction_idx]
            if s is not None
        ]

    def detect_eigenvalue_spike(
        self,
        direction_idx: int = 0,
        relative_threshold: float = 2.0,
    ) -> Optional[int]:
        """Find epoch where eigenvalue grows by > threshold relative to previous.

        Returns the epoch of the first spike, or None.
        """
        traj = self.eigenvalue_trajectory(direction_idx)
        if len(traj) < 2:
            return None
        for i in range(1, len(traj)):
            prev_ev = traj[i - 1][1]
            curr_ev = traj[i][1]
            if prev_ev > 1e-12 and curr_ev / prev_ev > relative_threshold:
                return traj[i][0]
        return None

    # ------------------------------------------------------------------
    # Feature discovery
    # ------------------------------------------------------------------

    def discover_dofs(
        self,
        min_stability: float = 0.9,
        min_stable_epochs: int = 3,
        min_variance_fraction: float = 0.01,
    ) -> List[DiscoveredDoF]:
        """Extract candidate DoFs from tracked directions.

        A direction becomes a DoF when:
        1. It has been stable (|cos_sim| > min_stability) for at least
           min_stable_epochs consecutive epochs.
        2. It explains at least min_variance_fraction of total variance.

        Returns:
            List of DiscoveredDoF, sorted by eigenvalue (largest first).
        """
        if not self._snapshots:
            return []

        latest = self._snapshots[-1]
        discovered: List[DiscoveredDoF] = []

        for hist_idx, history in enumerate(self._direction_histories):
            if not history:
                continue

            # Count consecutive stable epochs from the end
            stable_count = 0
            for i in range(len(history) - 1, -1, -1):
                _, _, stab = history[i]
                if stab is not None and stab >= min_stability:
                    stable_count += 1
                else:
                    break

            if stable_count < min_stable_epochs:
                continue

            # Check variance fraction
            last_epoch, last_ev, _ = history[-1]
            if last_epoch != latest.epoch:
                continue
            if latest.total_variance > 0 and last_ev / latest.total_variance < min_variance_fraction:
                continue

            # Find the direction in the latest snapshot
            direction = None
            for d in latest.directions:
                if abs(d.eigenvalue - last_ev) < 1e-12:
                    direction = d.direction
                    break
            if direction is None:
                continue

            discovered.append(DiscoveredDoF(
                dof=PolarDoF(name=f"pc_{hist_idx}"),
                projection=direction.copy(),
                eigenvalue=last_ev,
                stability_epochs=stable_count,
                source_layer=self._layer_name,
            ))

        discovered.sort(key=lambda d: d.eigenvalue, reverse=True)
        return discovered

    def create_projection_mapping(
        self,
        discovered_dofs: Optional[List[DiscoveredDoF]] = None,
    ) -> "_ProjectionMapping":
        """Create a mapping that projects activations onto discovered DoFs.

        If discovered_dofs is None, calls discover_dofs() with defaults.
        """
        if discovered_dofs is None:
            discovered_dofs = self.discover_dofs()

        projections = {dd.dof: dd.projection for dd in discovered_dofs}
        return _ProjectionMapping(
            model=self._model,
            layer_name=self._layer_name,
            projections=projections,
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshots and config.  Model is NOT serialized."""
        return {
            "layer_name": self._layer_name,
            "top_k": self._top_k,
            "device": self._device,
            "readout_layer_name": self._readout_layer_name,
            "snapshots": [
                {
                    "epoch": s.epoch,
                    "directions": [
                        {
                            "direction": d.direction.tolist(),
                            "eigenvalue": d.eigenvalue,
                            "epoch": d.epoch,
                            "stability": d.stability,
                            "readout_alignment": d.readout_alignment,
                        }
                        for d in s.directions
                    ],
                    "explained_variance_ratio": s.explained_variance_ratio.tolist(),
                    "total_variance": s.total_variance,
                }
                for s in self._snapshots
            ],
            "direction_histories": [
                [(e, ev, s) for e, ev, s in hist]
                for hist in self._direction_histories
            ],
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], model: nn.Module,
    ) -> "ActivationTracker":
        """Reconstruct from dict.  Model must be re-supplied."""
        tracker = cls(
            model=model,
            layer_name=d["layer_name"],
            top_k=d["top_k"],
            device=d.get("device", "cpu"),
            readout_layer_name=d.get("readout_layer_name"),
        )

        for snap_d in d["snapshots"]:
            directions = []
            for dd in snap_d["directions"]:
                directions.append(TrackedDirection(
                    direction=np.array(dd["direction"]),
                    eigenvalue=dd["eigenvalue"],
                    epoch=dd["epoch"],
                    stability=dd["stability"],
                    readout_alignment=dd.get("readout_alignment"),
                ))
            tracker._snapshots.append(DirectionSnapshot(
                epoch=snap_d["epoch"],
                directions=directions,
                explained_variance_ratio=np.array(snap_d["explained_variance_ratio"]),
                total_variance=snap_d["total_variance"],
            ))

        tracker._direction_histories = [
            [(e, ev, s) for e, ev, s in hist]
            for hist in d["direction_histories"]
        ]

        return tracker

    def __repr__(self) -> str:
        n_snap = len(self._snapshots)
        n_hist = len(self._direction_histories)
        return (
            f"ActivationTracker(layer={self._layer_name!r}, top_k={self._top_k}, "
            f"snapshots={n_snap}, tracked_directions={n_hist})"
        )


# ---------------------------------------------------------------------------
# _ProjectionMapping
# ---------------------------------------------------------------------------


class _ProjectionMapping:
    """Projects model activations onto discovered DoF directions.

    Attaches a temporary hook, runs one forward pass, captures activations,
    and returns the dot product with each projection vector as a State.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        projections: Dict[PolarDoF, np.ndarray],
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._layer_name = layer_name
        self._projections = projections
        self._device = device
        self._dofs = list(projections.keys())

    @property
    def dofs(self) -> List[PolarDoF]:
        return list(self._dofs)

    def _get_module(self, name: str) -> nn.Module:
        mod = self._model
        for part in name.split("."):
            mod = getattr(mod, part)
        return mod

    def project(self, activations: np.ndarray) -> Dict[PolarDoF, float]:
        """Project a single activation vector onto all discovered directions.

        Args:
            activations: Shape (hidden_dim,).

        Returns:
            Dict mapping each DoF to its scalar projection value.
        """
        result = {}
        for dof, vec in self._projections.items():
            result[dof] = float(np.dot(activations, vec))
        return result

    def project_batch(self, activations: np.ndarray) -> List[Dict[PolarDoF, float]]:
        """Project a batch of activations.

        Args:
            activations: Shape (N, hidden_dim).

        Returns:
            List of N dicts, each mapping DoF to scalar value.
        """
        # Stack projections into matrix for efficient batch projection
        proj_matrix = np.array([self._projections[d] for d in self._dofs])  # (K, D)
        projected = activations @ proj_matrix.T  # (N, K)

        results = []
        for i in range(len(activations)):
            results.append({
                self._dofs[j]: float(projected[i, j])
                for j in range(len(self._dofs))
            })
        return results


# ---------------------------------------------------------------------------
# Offline / Batch Feature Discovery
# ---------------------------------------------------------------------------


def extract_sfa_dofs(
    h_final: np.ndarray,
    h_earlier: np.ndarray,
    layer_name: str = "unknown_layer",
    n_components: int = 10,
    min_settle_score: float = 10.0,
    regularization: float = 1e-6,
) -> List[DiscoveredDoF]:
    """Discover settled features using Slow Feature Analysis / Cointegration.
    
    Finds directions in activation space that have high variance in the final 
    state but low variance in their change from an earlier state. This isolates 
    features that have "settled" (formed and stabilized) while embedding noise 
    continues to drift.
    
    Args:
        h_final: Activations at current epoch, shape (N, D).
        h_earlier: Activations at previous epoch for the exact same inputs, shape (N, D).
        layer_name: String to identify the source layer in the resulting DoFs.
        n_components: Maximum number of directions to return.
        min_settle_score: Minimum generalized eigenvalue (ratio of final variance
            to change variance) to consider a direction settled.
        regularization: Small value added to the diagonal of the change covariance
            to avoid singularity.
            
    Returns:
        List of DiscoveredDoF, sorted by settle score (highest first).
    """
    import scipy.linalg
    
    if h_final.shape != h_earlier.shape:
        raise ValueError("h_final and h_earlier must have identical shapes.")
        
    # Covariance of final state
    hf_centered = h_final - h_final.mean(axis=0)
    cov_final = (hf_centered.T @ hf_centered) / (len(h_final) - 1)
    
    # Covariance of temporal change
    diff = h_final - h_earlier
    diff_centered = diff - diff.mean(axis=0)
    cov_diff = (diff_centered.T @ diff_centered) / (len(diff) - 1)
    
    # Regularize to avoid singularity
    cov_diff += np.eye(cov_diff.shape[0]) * regularization
    
    # Solve Generalized Eigenvalue Problem
    eigenvalues, eigenvectors = scipy.linalg.eigh(cov_final, cov_diff)
    
    # Filter and sort
    discovered = []
    idx = np.argsort(eigenvalues)[::-1]
    
    for i in idx:
        score = float(eigenvalues[i])
        if score < min_settle_score:
            continue
            
        direction = eigenvectors[:, i].copy()
        direction = direction / np.clip(np.linalg.norm(direction), 1e-12, None)
        
        discovered.append(DiscoveredDoF(
            dof=PolarDoF(name=f"sfa_{len(discovered)}"),
            projection=direction,
            eigenvalue=score,          # We store the settle score here
            stability_epochs=-1,       # N/A for offline SFA
            source_layer=layer_name,
            discovery_method="sfa",
        ))
        
        if len(discovered) >= n_components:
            break
            
    return discovered
