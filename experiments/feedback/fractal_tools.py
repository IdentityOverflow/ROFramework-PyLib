"""Shared measurement tools for the feedback/fractal experiments.

Three dimension estimators and one stability estimator, all plain numpy:

  box_counting_dimension   D_box of a binary 2-D image (spatial fractal).
  correlation_dimension    D2 of a point cloud (Grassberger & Procaccia 1983):
                           C(r) ~ r^D2 in the scaling region.
  spectral_slope           slope of the radially averaged spatial power
                           spectrum, log P vs log k (self-similar images give
                           a straight line; its slope relates to roughness).
  effective_depth          for a gain-g iterated copy, how many generations
                           stay above a visibility floor: log(eps)/log(g).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Spatial: box counting
# ---------------------------------------------------------------------------

def box_counting_dimension(
    mask: np.ndarray,
    min_box: int = 1,
    max_box: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Box-counting dimension of a binary image.

    Counts boxes of side eps (powers of two from min_box up) that contain at
    least one True pixel, then fits log N(eps) against log(1/eps).

    Returns (dimension, eps_array, counts_array). Dimension is 0.0 for an
    empty mask and the fit uses only box sizes with a nonzero count.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0, np.array([]), np.array([])
    h, w = mask.shape
    side = min(h, w)
    if max_box is None:
        max_box = side // 4
    sizes = []
    e = min_box
    while e <= max_box:
        sizes.append(e)
        e *= 2
    counts = []
    for e in sizes:
        hh = (h // e) * e
        ww = (w // e) * e
        sub = mask[:hh, :ww].reshape(hh // e, e, ww // e, e)
        occupied = sub.any(axis=(1, 3))
        counts.append(int(occupied.sum()))
    sizes_a = np.array(sizes, dtype=float)
    counts_a = np.array(counts, dtype=float)
    keep = counts_a > 0
    if keep.sum() < 2:
        return 0.0, sizes_a, counts_a
    slope, _ = np.polyfit(np.log(1.0 / sizes_a[keep]), np.log(counts_a[keep]), 1)
    return float(slope), sizes_a, counts_a


# ---------------------------------------------------------------------------
# Spatial: power-spectrum slope
# ---------------------------------------------------------------------------

def spectral_slope(frame: np.ndarray) -> float:
    """Slope of log radial power vs log wavenumber for a 2-D image.

    A flat (white) image has power at k=0 only; the slope is then
    undefined and we return nan. Fractal/self-similar textures give a
    roughly straight line over a decade of k.
    """
    f = np.asarray(frame, dtype=float)
    f = f - f.mean()
    if np.allclose(f, 0.0):
        return float("nan")
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(P.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = min(cy, cx)
    radial = np.bincount(r.ravel(), weights=P.ravel())[: rmax]
    nr = np.bincount(r.ravel())[: rmax]
    radial = radial / np.maximum(nr, 1)
    k = np.arange(len(radial))
    sel = (k >= 2) & (radial > 0)
    if sel.sum() < 4:
        return float("nan")
    slope, _ = np.polyfit(np.log(k[sel]), np.log(radial[sel]), 1)
    return float(slope)


def effective_depth(gain: float, floor: float = 0.05) -> float:
    """Generations of an iterated copy that stay above a visibility floor.

    For gain < 1 the k-th copy has brightness gain**k; it drops below
    `floor` after log(floor)/log(gain) generations. For gain >= 1 the depth
    is infinite (every copy stays visible, saturation permitting).
    """
    if gain >= 1.0:
        return float("inf")
    if gain <= 0.0:
        return 0.0
    return float(np.log(floor) / np.log(gain))


# ---------------------------------------------------------------------------
# Dynamical: correlation dimension
# ---------------------------------------------------------------------------

def correlation_dimension(
    points: np.ndarray,
    n_pairs: int = 200_000,
    rng: Optional[np.random.Generator] = None,
    q_lo: float = 0.02,
    q_hi: float = 0.30,
) -> Tuple[float, float]:
    """Grassberger-Procaccia correlation dimension of a point cloud.

    Samples random pairs, computes their distances d, and fits the slope of
    log C(r) against log r, where C(r) is the fraction of pairs with d < r,
    over the quantile window [q_lo, q_hi] of the distance distribution.

    Returns (D2, spread) where spread is the std of the local slopes across
    the fit window (large spread = no clean scaling region). A point cloud
    that has collapsed to one point returns (0.0, 0.0).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 50:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    d = np.linalg.norm(pts[i[keep]] - pts[j[keep]], axis=1)
    d = d[d > 0]
    if len(d) < 100 or np.ptp(d) < 1e-12:
        return 0.0, 0.0
    lo, hi = np.quantile(d, [q_lo, q_hi])
    if hi <= lo:
        return 0.0, 0.0
    rs = np.exp(np.linspace(np.log(lo), np.log(hi), 12))
    d_sorted = np.sort(d)
    C = np.searchsorted(d_sorted, rs) / len(d_sorted)
    sel = C > 0
    if sel.sum() < 4:
        return float("nan"), float("nan")
    lr, lC = np.log(rs[sel]), np.log(C[sel])
    slope, _ = np.polyfit(lr, lC, 1)
    local = np.diff(lC) / np.diff(lr)
    return float(slope), float(np.std(local))


def pca_project(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Project a (T, N) trajectory onto its top-k principal axes."""
    X = np.asarray(points, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] <= k:
        return X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:k].T


def participation_ratio(points: np.ndarray) -> float:
    """Effective number of active dimensions: (sum λ)^2 / sum λ^2 of the
    covariance eigenvalues. Linear sibling of the correlation dimension."""
    X = np.asarray(points, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[0] < 2:
        return 0.0
    cov = X.T @ X / (X.shape[0] - 1)
    lam = np.linalg.eigvalsh(cov)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())
