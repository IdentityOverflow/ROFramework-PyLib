"""
Seed Architecture — Frequency Band Formation & Cross-Scale Coupling

Tests the prediction from seed_architecture.md Section 4.3-4.4:
  - When exposed to multi-timescale input, nodes should cluster into
    distinct frequency bands through entrainment
  - Slow nodes should model fast-node aggregate behavior (cross-scale coupling)

Protocol:
  1. Initialize 64 nodes with log-uniform frequencies [0.01, 1.0]
  2. Drive with a composite signal: slow modulation (0.05 Hz) * fast carrier (0.3 Hz)
     This has two distinct timescales — the spec predicts two frequency bands
  3. Run for a long time (20k+ steps) to allow entrainment
  4. Measure:
     a. Frequency distribution — did bands form? (histogram, gap detection)
     b. Cross-scale coupling — do slow nodes correlate with fast-node aggregates?
     c. Phase-amplitude coupling — does slow phase modulate fast amplitude?

Usage:
    python experiments/seed/04_frequency_bands.py [--nodes N] [--steps N]
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
from scipy import stats

from ro_framework.seed.network import SeedNetwork
from ro_framework.seed.node import SeedConfig


# ---------------------------------------------------------------------------
# Sensor — multi-timescale signal
# ---------------------------------------------------------------------------

class MultiTimescaleSensor:
    """Drives nodes with a composite signal: slow modulation * fast carrier.

    Each node receives a drive proportional to its frequency proximity
    to the signal components.
    """

    def __init__(
        self,
        slow_freq: float = 0.05,
        fast_freq: float = 0.3,
        bandwidth: float = 0.8,
    ):
        self.slow_freq = slow_freq
        self.fast_freq = fast_freq
        self.bandwidth = bandwidth

    def __call__(
        self, external_input: np.ndarray, node_frequencies: Dict[str, float]
    ) -> Dict[str, float]:
        t = float(external_input[0])

        # Composite signal: slow envelope modulates fast carrier
        slow = 0.5 * (1.0 + np.sin(2 * np.pi * self.slow_freq * t))  # [0, 1]
        fast = np.sin(2 * np.pi * self.fast_freq * t)
        composite = slow * fast  # amplitude-modulated

        # Each node gets drive weighted by frequency proximity to BOTH components
        log_slow = np.log(max(self.slow_freq, 1e-6))
        log_fast = np.log(max(self.fast_freq, 1e-6))

        drives = {}
        for nid, nf in node_frequencies.items():
            log_nf = np.log(max(nf, 1e-6))
            # Proximity to slow component
            w_slow = np.exp(-0.5 * ((log_nf - log_slow) / self.bandwidth) ** 2)
            # Proximity to fast component
            w_fast = np.exp(-0.5 * ((log_nf - log_fast) / self.bandwidth) ** 2)
            # Both contribute
            drives[nid] = float(w_slow * slow + w_fast * composite)
        return drives


class MeanActuator:
    def __call__(self, node_activations: Dict[str, float]) -> np.ndarray:
        if not node_activations:
            return np.array([0.0])
        return np.array([np.mean(list(node_activations.values()))])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def detect_frequency_bands(
    frequencies: np.ndarray,
    min_gap_ratio: float = 1.5,
) -> List[List[int]]:
    """Detect clusters in log-frequency space using gap detection.

    Args:
        frequencies: Array of node frequencies.
        min_gap_ratio: Minimum ratio between adjacent sorted frequencies
            to be considered a band boundary (in log space).

    Returns:
        List of bands, each band is a list of node indices.
    """
    sorted_idx = np.argsort(frequencies)
    sorted_freqs = frequencies[sorted_idx]
    log_freqs = np.log(sorted_freqs)

    # Find gaps
    gaps = np.diff(log_freqs)
    median_gap = np.median(gaps) if len(gaps) > 0 else 1.0
    threshold = median_gap * min_gap_ratio

    bands = []
    current_band = [sorted_idx[0]]
    for i in range(1, len(sorted_idx)):
        if gaps[i - 1] > threshold:
            bands.append(current_band)
            current_band = []
        current_band.append(sorted_idx[i])
    if current_band:
        bands.append(current_band)

    return bands


def measure_phase_amplitude_coupling(
    slow_activations: np.ndarray,
    fast_activations: np.ndarray,
    n_phase_bins: int = 8,
) -> float:
    """Measure modulation index (Tort et al. 2010).

    Bins the slow signal's phase and measures whether fast amplitude
    varies across phase bins. MI = 0 means no coupling, MI = 1 means
    perfect coupling.

    Args:
        slow_activations: Time series of slow-band mean activation.
        fast_activations: Time series of fast-band mean activation.
        n_phase_bins: Number of phase bins.

    Returns:
        Modulation index (0 = no coupling, higher = stronger coupling).
    """
    from scipy.signal import hilbert

    n = len(slow_activations)
    if n < 50:
        return 0.0

    # Analytic signal of slow oscillation → phase
    analytic_slow = hilbert(slow_activations - np.mean(slow_activations))
    slow_phase = np.angle(analytic_slow)

    # Amplitude envelope of fast oscillation
    analytic_fast = hilbert(fast_activations - np.mean(fast_activations))
    fast_amp = np.abs(analytic_fast)

    # Bin fast amplitude by slow phase
    bin_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    bin_means = np.zeros(n_phase_bins)
    for i in range(n_phase_bins):
        mask = (slow_phase >= bin_edges[i]) & (slow_phase < bin_edges[i + 1])
        if np.any(mask):
            bin_means[i] = np.mean(fast_amp[mask])

    # Normalize to probability distribution
    total = np.sum(bin_means)
    if total < 1e-10:
        return 0.0
    p = bin_means / total

    # KL divergence from uniform (= modulation index)
    uniform = np.ones(n_phase_bins) / n_phase_bins
    # Avoid log(0)
    p_safe = np.clip(p, 1e-10, None)
    kl = np.sum(p_safe * np.log(p_safe / uniform))
    mi = kl / np.log(n_phase_bins)  # normalize to [0, 1]

    return float(mi)


def measure_cross_scale_correlation(
    slow_activations: np.ndarray,
    fast_activations: np.ndarray,
    window: int = 50,
) -> float:
    """Measure whether slow nodes track fast-node aggregate statistics.

    Computes correlation between slow activation and the rolling
    variance of fast activations (slow nodes should model fast
    aggregate behavior per Section 4.4).

    Args:
        slow_activations: Time series of slow-band mean activation.
        fast_activations: Time series of fast-band mean activation.
        window: Rolling window for fast variance computation.

    Returns:
        Pearson correlation between slow activation and fast rolling variance.
    """
    n = len(slow_activations)
    if n < window + 10:
        return 0.0

    # Rolling variance of fast signal
    fast_var = np.array([
        np.var(fast_activations[max(0, i - window):i])
        for i in range(window, n)
    ])

    # Align slow signal
    slow_aligned = slow_activations[window:]

    if len(slow_aligned) != len(fast_var):
        min_len = min(len(slow_aligned), len(fast_var))
        slow_aligned = slow_aligned[:min_len]
        fast_var = fast_var[:min_len]

    if np.std(slow_aligned) < 1e-10 or np.std(fast_var) < 1e-10:
        return 0.0

    r, _ = stats.pearsonr(slow_aligned, fast_var)
    return float(r)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_init: int = 64,
    total_steps: int = 20000,
    slow_freq: float = 0.05,
    fast_freq: float = 0.3,
):
    print("Seed Frequency Band Formation & Cross-Scale Coupling")
    print(f"  Nodes: {n_init}")
    print(f"  Steps: {total_steps}")
    print(f"  Signal: {slow_freq} Hz (slow) * {fast_freq} Hz (fast)")
    print("=" * 70)

    config = SeedConfig(
        n_init=n_init,
        k_neighbors=6,
        n_seed_nodes=8,
        max_nodes=256,
        noise_floor=0.05,
        drive_amplitude=0.2,
        activation_threshold=0.5,
        learning_rate=0.01,
        w_max=2.0,
        prune_weight_threshold=0.005,
        prune_weight_window=200,
        freq_learning_rate=0.001,
        freq_range=(0.01, 1.0),
    )

    sensor = MultiTimescaleSensor(slow_freq=slow_freq, fast_freq=fast_freq)
    actuator = MeanActuator()
    net = SeedNetwork(config, sensor, actuator, seed=42)

    # Record initial frequency distribution
    init_freqs = np.array([n.frequency for n in net.nodes.values()])
    node_ids = sorted(net.nodes.keys())

    # Storage for time series (last portion for analysis)
    record_start = max(0, total_steps - 5000)
    per_node_history: Dict[str, List[float]] = {nid: [] for nid in node_ids}

    log_interval = 2000
    t0 = time.time()

    print(f"\nInitial freq range: [{init_freqs.min():.4f}, {init_freqs.max():.4f}]")
    print(f"Initial freq std:   {init_freqs.std():.4f}")
    print()

    for step in range(total_steps):
        t = step * config.dt
        net.step(np.array([t]))

        # Record activations for analysis window
        if step >= record_start:
            for nid in node_ids:
                if nid in net.nodes:
                    per_node_history[nid].append(net.nodes[nid].activation)

        if (step + 1) % log_interval == 0:
            freqs = np.array([net.nodes[nid].frequency for nid in node_ids if nid in net.nodes])
            sigmas = np.array([net.nodes[nid].branching_ratio for nid in node_ids if nid in net.nodes])
            n_active = sum(
                1 for nid in node_ids
                if nid in net.nodes and abs(net.nodes[nid].activation) > config.activation_threshold
            )
            elapsed = time.time() - t0
            print(f"  Step {step + 1:6d} | nodes={len(net.nodes):3d} | "
                  f"σ={sigmas.mean():.3f}±{sigmas.std():.3f} | "
                  f"active={n_active:3d} | "
                  f"freq=[{freqs.min():.4f}, {freqs.max():.4f}] | "
                  f"t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    # 1. Frequency distribution
    final_freqs = np.array([net.nodes[nid].frequency for nid in node_ids if nid in net.nodes])
    print(f"\n--- Frequency Distribution ---")
    print(f"  Final range: [{final_freqs.min():.4f}, {final_freqs.max():.4f}]")
    print(f"  Final std:   {final_freqs.std():.4f}")
    print(f"  Initial std: {init_freqs.std():.4f}")

    # Histogram in log space
    log_freqs = np.log10(final_freqs)
    n_bins = 15
    counts, edges = np.histogram(log_freqs, bins=n_bins)
    print(f"\n  Log-frequency histogram:")
    for i in range(n_bins):
        freq_lo = 10 ** edges[i]
        freq_hi = 10 ** edges[i + 1]
        bar = "#" * counts[i]
        print(f"    [{freq_lo:6.3f}-{freq_hi:6.3f}] {counts[i]:3d} {bar}")

    # 2. Band detection
    bands = detect_frequency_bands(final_freqs, min_gap_ratio=2.0)
    print(f"\n--- Band Detection ---")
    print(f"  Number of bands detected: {len(bands)}")
    for i, band in enumerate(bands):
        band_freqs = final_freqs[band]
        print(f"  Band {i + 1}: {len(band)} nodes, "
              f"freq=[{band_freqs.min():.4f}, {band_freqs.max():.4f}], "
              f"mean={band_freqs.mean():.4f}")

    # 3. Cross-scale coupling (if we have at least 2 bands)
    if len(bands) >= 2:
        print(f"\n--- Cross-Scale Coupling ---")

        # Sort bands by mean frequency
        band_mean_freqs = [final_freqs[b].mean() for b in bands]
        band_order = np.argsort(band_mean_freqs)
        slow_band = bands[band_order[0]]
        fast_band = bands[band_order[-1]]

        active_node_ids = [nid for nid in node_ids if nid in net.nodes]

        # Build time series for each band
        slow_ts = np.mean([
            np.array(per_node_history[active_node_ids[idx]])
            for idx in slow_band
            if idx < len(active_node_ids)
        ], axis=0)
        fast_ts = np.mean([
            np.array(per_node_history[active_node_ids[idx]])
            for idx in fast_band
            if idx < len(active_node_ids)
        ], axis=0)

        min_len = min(len(slow_ts), len(fast_ts))
        slow_ts = slow_ts[:min_len]
        fast_ts = fast_ts[:min_len]

        # Phase-amplitude coupling
        pac = measure_phase_amplitude_coupling(slow_ts, fast_ts)
        print(f"  Phase-amplitude coupling (MI): {pac:.4f}")
        print(f"    (0 = no coupling, >0.01 = weak, >0.05 = moderate, >0.1 = strong)")

        # Cross-scale correlation (slow tracks fast variance)
        csc = measure_cross_scale_correlation(slow_ts, fast_ts, window=50)
        print(f"  Slow-fast variance correlation: {csc:.4f}")
        print(f"    (slow activation ↔ fast rolling variance)")

        # Direct correlation between bands
        direct_r, direct_p = stats.pearsonr(slow_ts, fast_ts)
        print(f"  Direct correlation: r={direct_r:.4f}, p={direct_p:.2e}")

    else:
        print(f"\n--- Cross-Scale Coupling ---")
        print(f"  Cannot measure — fewer than 2 bands detected")
        print(f"  This means frequency entrainment collapsed all nodes")
        print(f"  toward a single frequency, rather than forming distinct bands.")

    # 4. Compare to signal frequencies
    print(f"\n--- Signal Alignment ---")
    print(f"  Input slow freq: {slow_freq:.3f} Hz")
    print(f"  Input fast freq: {fast_freq:.3f} Hz")
    if len(bands) >= 2:
        band_means = sorted([final_freqs[b].mean() for b in bands])
        print(f"  Band centers:    {', '.join(f'{m:.3f}' for m in band_means)} Hz")
        # Distance from signal frequencies
        for bm in band_means:
            closest_signal = min([slow_freq, fast_freq], key=lambda sf: abs(np.log(bm / sf)))
            ratio = bm / closest_signal
            print(f"    {bm:.3f} Hz → closest signal {closest_signal:.3f} Hz (ratio {ratio:.2f})")

    # 5. Did entrainment narrow or cluster?
    print(f"\n--- Entrainment Summary ---")
    narrowed = final_freqs.std() < init_freqs.std()
    print(f"  Frequency spread narrowed: {'YES' if narrowed else 'NO'} "
          f"(std {init_freqs.std():.4f} → {final_freqs.std():.4f})")
    if len(bands) > 1:
        print(f"  Bands formed: YES ({len(bands)} bands)")
        print(f"  Interpretation: entrainment produced clustering, not collapse")
    else:
        print(f"  Bands formed: NO (all nodes converged toward single frequency)")
        print(f"  Interpretation: entrainment is too strong or signal bandwidth too wide")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--slow-freq", type=float, default=0.05)
    parser.add_argument("--fast-freq", type=float, default=0.3)
    args = parser.parse_args()
    run_experiment(
        n_init=args.nodes,
        total_steps=args.steps,
        slow_freq=args.slow_freq,
        fast_freq=args.fast_freq,
    )
