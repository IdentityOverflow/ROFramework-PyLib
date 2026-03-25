"""
Criticality monitoring tools for the Seed architecture.

Provides verification that the network operates at criticality (σ ≈ 1)
and that cascade size distributions follow power laws. These are
VERIFICATION tools — the optimization target is branching ratio σ = 1
(Rule 2a), not the power law itself.

See docs/seed_architecture.md Sections 4.1, 4.2, and 8.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from ro_framework.seed.node import OscillatoryNode


# ---------------------------------------------------------------------------
# Cascade extraction
# ---------------------------------------------------------------------------

def extract_cascades(
    activation_series: np.ndarray,
    threshold: float = 0.3,
) -> List[int]:
    """Extract cascade sizes from an activation time series.

    A cascade is a maximal contiguous run of timesteps where
    |activation| > threshold.

    Args:
        activation_series: 1-D array of activation values.
        threshold: Activation magnitude threshold for "active".

    Returns:
        List of cascade sizes (run lengths). Empty if no cascades found.
    """
    if len(activation_series) == 0:
        return []

    active = np.abs(np.asarray(activation_series)) > threshold
    if not np.any(active):
        return []

    # Find runs: diff of active gives +1 at run start, -1 at run end
    padded = np.concatenate([[False], active, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return (ends - starts).tolist()


# ---------------------------------------------------------------------------
# Power-law verification (Clauset et al. 2009)
# ---------------------------------------------------------------------------

def verify_power_law(
    cascade_sizes: Sequence[int],
    min_samples: int = 50,
    x_min: int = 1,
) -> Tuple[bool, float, float]:
    """Verify that cascade size distribution follows a power law.

    Uses MLE for the discrete power-law exponent and KS distance for
    goodness of fit (Clauset, Shalizi & Newman, 2009).

    This is a VERIFICATION tool (Section 4.2), not the optimization
    target (which is branching ratio σ = 1, Section 3.2).

    Args:
        cascade_sizes: Observed cascade sizes (positive integers).
        min_samples: Minimum number of cascades for reliable fitting.
        x_min: Minimum cascade size for the power-law fit.

    Returns:
        (is_power_law, alpha, ks_distance)
        - is_power_law: True if ks < 0.15 and 1.2 < α < 3.0
        - alpha: MLE power-law exponent
        - ks_distance: KS distance (lower = better fit)
    """
    sizes = np.array([s for s in cascade_sizes if s >= x_min], dtype=np.float64)

    if len(sizes) < min_samples:
        return (False, 0.0, 1.0)

    n = len(sizes)
    sorted_sizes = np.sort(sizes).astype(int)
    max_val = int(sorted_sizes[-1])

    # Truncation limit for normalization sums
    trunc = max(max_val * 2, 10000)
    all_k = np.arange(x_min, trunc + 1, dtype=np.float64)
    sum_log_sizes = np.sum(np.log(sizes))

    # --- Discrete MLE via numerical optimization ---
    # Log-likelihood: L(α) = -α * Σ ln(x_i) - n * ln(Σ_{k=x_min}^{∞} k^{-α})
    from scipy.optimize import minimize_scalar

    def neg_log_likelihood(alpha: float) -> float:
        if alpha <= 1.0:
            return 1e30
        log_norm = np.log(np.sum(all_k ** (-alpha)))
        return alpha * sum_log_sizes + n * log_norm

    result = minimize_scalar(neg_log_likelihood, bounds=(1.01, 10.0), method="bounded")
    alpha = float(result.x)

    if alpha <= 1.0 or not np.isfinite(alpha):
        return (False, alpha, 1.0)

    # --- KS distance: empirical vs discrete power-law CDF ---
    # For discrete data, compare CDFs at each unique value (step-function CDF)
    pmf_unnorm = all_k ** (-alpha)
    norm_const = pmf_unnorm.sum()
    cdf_table = np.cumsum(pmf_unnorm) / norm_const

    unique_vals, counts = np.unique(sorted_sizes, return_counts=True)
    empirical_cdf = np.cumsum(counts) / n

    theoretical_cdf = np.empty(len(unique_vals))
    for i, val in enumerate(unique_vals):
        idx = int(val) - x_min
        if 0 <= idx < len(cdf_table):
            theoretical_cdf[i] = cdf_table[idx]
        else:
            theoretical_cdf[i] = 1.0

    ks_distance = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    # --- KS for exponential alternative ---
    lam = 1.0 / np.mean(sizes)
    exp_cdf = 1.0 - np.exp(-lam * unique_vals.astype(np.float64))
    ks_exp = float(np.max(np.abs(empirical_cdf - exp_cdf)))

    # Power law is accepted if:
    # 1. KS distance is reasonable (< 0.15)
    # 2. Exponent is in healthy range
    # 3. Power law fits at least as well as exponential
    is_power_law = (
        ks_distance < 0.15
        and 1.2 < alpha < 3.0
        and ks_distance <= ks_exp
    )

    return (is_power_law, float(alpha), ks_distance)


# ---------------------------------------------------------------------------
# Branching ratio measurement
# ---------------------------------------------------------------------------

def measure_branching_ratio(
    node_history: np.ndarray,
    neighbor_histories: Dict[str, np.ndarray],
    threshold: float = 0.3,
) -> float:
    """Compute empirical branching ratio σ for a single node.

    σ = mean(number of neighbors activating at t+1 | node activated at t)

    This is the PRIMARY criticality metric (Section 3.1).

    Args:
        node_history: 1-D array of the node's activation history.
        neighbor_histories: {neighbor_id: 1-D activation array}.
            All arrays must have the same length.
        threshold: Activation magnitude threshold.

    Returns:
        Empirical branching ratio. 0.0 if no activation events found.
    """
    node_history = np.asarray(node_history)
    n = len(node_history)
    if n < 2 or not neighbor_histories:
        return 0.0

    # Timesteps where the node is active (excluding last, since we need t+1)
    node_active = np.abs(node_history[:-1]) > threshold
    active_times = np.where(node_active)[0]

    if len(active_times) == 0:
        return 0.0

    # Count active neighbors at t+1 for each active timestep
    total_propagated = 0
    for nid, hist in neighbor_histories.items():
        hist = np.asarray(hist)
        if len(hist) != n:
            continue
        neighbor_active_next = np.abs(hist[1:]) > threshold
        total_propagated += np.sum(neighbor_active_next[active_times])

    return float(total_propagated / len(active_times))


# ---------------------------------------------------------------------------
# Fast mutual information (for Rules 4/5)
# ---------------------------------------------------------------------------

def fast_mi(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 8,
) -> float:
    """Fast binned mutual information on raw numpy arrays.

    Uses np.histogram2d for speed — no State/DoF construction overhead.
    Suitable for per-step MI estimation in Rules 4 and 5.

    Args:
        x: 1-D array of values.
        y: 1-D array of values (same length as x).
        bins: Number of bins for discretization.

    Returns:
        Mutual information in nats (natural log). Non-negative.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y) or len(x) < 2:
        return 0.0

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()

    # Marginals
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI = Σ p(x,y) * log(p(x,y) / (p(x)*p(y)))
    # Only sum over non-zero entries
    mask = pxy > 0
    outer = np.outer(px, py)
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / outer[mask]))

    return max(0.0, float(mi))


# ---------------------------------------------------------------------------
# Scale distribution (experimental — Q5)
# ---------------------------------------------------------------------------

def measure_scale_distribution(
    nodes: Dict[str, "OscillatoryNode"],
    threshold: float = 0.1,
    mi_bins: int = 8,
) -> Tuple[bool, float, float]:
    """Measure spatial scale invariance across the node network.

    Computes pairwise MI between node activation histories, builds an
    adjacency graph, finds connected components, and tests whether
    component sizes follow a power law.

    This is experimental (Q5 in the spec). O(N²) in node count.

    Args:
        nodes: {node_id: OscillatoryNode} with populated activation_history.
        threshold: MI threshold for considering two nodes correlated.
        mi_bins: Bins for MI estimation.

    Returns:
        (is_power_law, beta, ks_distance) for the component size distribution.
    """
    node_ids = sorted(nodes.keys())
    n = len(node_ids)
    if n < 3:
        return (False, 0.0, 1.0)

    # Extract activation histories as arrays
    histories = {}
    min_len = float("inf")
    for nid in node_ids:
        h = np.array(list(nodes[nid].activation_history), dtype=np.float64)
        histories[nid] = h
        min_len = min(min_len, len(h))

    min_len = int(min_len)
    if min_len < 10:
        return (False, 0.0, 1.0)

    # Truncate to common length
    for nid in node_ids:
        histories[nid] = histories[nid][-min_len:]

    # Build adjacency matrix via pairwise MI
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            mi = fast_mi(histories[node_ids[i]], histories[node_ids[j]], mi_bins)
            if mi > threshold:
                adj[i, j] = True
                adj[j, i] = True

    # Find connected components
    n_components, labels = sparse.csgraph.connected_components(
        sparse.csr_matrix(adj), directed=False
    )

    if n_components < 3:
        return (False, 0.0, 1.0)

    # Component sizes
    _, counts = np.unique(labels, return_counts=True)
    component_sizes = counts[counts >= 1].tolist()

    return verify_power_law(component_sizes, min_samples=3, x_min=1)
