"""
Seed Architecture — Criticality Validation (Minimal)

Tests whether Rule 2a alone (Hebbian weight adjustment governed by
branching ratio σ) produces self-organized criticality in a sparse
activation regime.

Key finding: Rule 2a works when node drives are uncorrelated (sparse
random pulses). With a common structured signal, neighbor correlation
is dominated by the signal, not coupling, and σ can't converge.

This experiment validates the core mechanism with independent drives,
then compares against correlated (signal) drives to demonstrate the
difference.

Usage:
    python experiments/seed/01_criticality_validation.py [--steps N] [--nodes N] [--plot]
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from ro_framework.seed.criticality import extract_cascades, verify_power_law
from ro_framework.seed.node import OscillatoryNode, SeedConfig


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_ring_lattice(
    config: SeedConfig, seed: int = 42
) -> Dict[str, OscillatoryNode]:
    """Create N nodes in a ring-lattice with k nearest frequency neighbors."""
    rng = np.random.default_rng(seed)
    freq_lo, freq_hi = config.freq_range

    log_freqs = np.linspace(np.log(freq_lo), np.log(freq_hi), config.n_init)
    log_freqs += rng.standard_normal(config.n_init) * 0.05
    frequencies = np.clip(np.exp(log_freqs), freq_lo, freq_hi)
    sorted_idx = np.argsort(frequencies)

    nodes: Dict[str, OscillatoryNode] = {}
    node_ids: List[str] = []

    for i in sorted_idx:
        nid = f"n{i}"
        node_ids.append(nid)
        nodes[nid] = OscillatoryNode(
            node_id=nid,
            frequency=float(frequencies[i]),
            phase=float(rng.uniform(0, 2 * np.pi)),
            _config=config,
        )

    # Ring-lattice connectivity
    n = len(node_ids)
    k = config.k_neighbors
    for i, nid in enumerate(node_ids):
        for offset in range(1, k // 2 + 1):
            for j in [i - offset, i + offset]:
                j = j % n
                other = node_ids[j]
                if other != nid:
                    w = float(rng.uniform(0.1, 0.5))
                    nodes[nid].form_connection(other, initial_weight=w)

    return nodes


def minimal_step(
    nodes: Dict[str, OscillatoryNode],
    drives: Dict[str, float],
    rng: np.random.Generator,
) -> None:
    """One timestep: activate nodes, then adjust weights (Rule 2a only)."""
    neighborhoods = {}
    for nid, node in nodes.items():
        neighborhoods[nid] = {
            oid: nodes[oid].activation
            for oid in node.coupling_weights
            if oid in nodes
        }

    for nid, node in nodes.items():
        node.step(neighborhoods[nid], drives.get(nid, 0.0), rng)

    for node in nodes.values():
        node.adjust_couplings()


# ---------------------------------------------------------------------------
# Drive patterns
# ---------------------------------------------------------------------------

def independent_random_drives(
    node_ids: list, rng: np.random.Generator,
    pulse_prob: float = 0.05, pulse_range: tuple = (0.5, 1.5),
) -> Dict[str, float]:
    """Each node gets independent sparse random pulses."""
    drives = {}
    for nid in node_ids:
        if rng.random() < pulse_prob:
            drives[nid] = float(rng.uniform(*pulse_range))
        else:
            drives[nid] = 0.0
    return drives


SIGNAL_FREQS = [0.02, 0.08, 0.25, 0.7]
SIGNAL_AMPS = [1.0, 0.8, 0.6, 0.4]
_LOG_SF = np.log(np.array(SIGNAL_FREQS))


def correlated_signal_drives(
    t: float, node_frequencies: Dict[str, float],
    bandwidth: float = 1.0,
) -> Dict[str, float]:
    """All nodes near a signal frequency get the same drive simultaneously."""
    signal = np.array([
        amp * np.sin(2 * np.pi * freq * t)
        for freq, amp in zip(SIGNAL_FREQS, SIGNAL_AMPS)
    ])
    drives = {}
    for nid, nf in node_frequencies.items():
        log_nf = np.log(max(nf, 1e-6))
        weights = np.exp(-0.5 * ((log_nf - _LOG_SF) / bandwidth) ** 2)
        drives[nid] = float(np.dot(weights, signal))
    return drives


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(
    label: str,
    config: SeedConfig,
    n_steps: int,
    drive_fn,
    seed: int = 42,
) -> dict:
    """Run the minimal experiment for one drive condition."""
    print(f"\n--- {label} ---")
    rng = np.random.default_rng(seed)
    nodes = build_ring_lattice(config, seed=seed)

    all_activations = {nid: [] for nid in nodes}
    active_counts = []
    sigma_history = []
    log_interval = max(1, n_steps // 5)

    t0 = time.time()
    for step in range(n_steps):
        drives = drive_fn(step, nodes, rng)
        minimal_step(nodes, drives, rng)

        n_active = sum(1 for n in nodes.values()
                       if abs(n.activation) > config.activation_threshold)
        active_counts.append(n_active)
        for nid, node in nodes.items():
            all_activations[nid].append(node.activation)

        if (step + 1) % log_interval == 0:
            sigmas = [n.branching_ratio for n in nodes.values()]
            n_conns = sum(len(n.coupling_weights) for n in nodes.values()) // 2
            near_crit = sum(1 for s in sigmas if 0.5 < s < 1.5)
            sigma_history.append((step + 1, np.mean(sigmas), np.std(sigmas)))
            elapsed = time.time() - t0
            print(f"  Step {step+1:5d} | σ={np.mean(sigmas):.3f} ± {np.std(sigmas):.3f} | "
                  f"near_crit={near_crit}/{len(nodes)} | "
                  f"active={n_active/len(nodes):.1%} | conns={n_conns} | t={elapsed:.1f}s")

    # Analysis
    final_sigmas = [n.branching_ratio for n in nodes.values()]
    mean_sigma = np.mean(final_sigmas)
    mean_active = np.mean(active_counts) / len(nodes)

    all_cascades = []
    for nid, acts in all_activations.items():
        if len(acts) > 100:
            all_cascades.extend(
                extract_cascades(np.array(acts), threshold=config.activation_threshold)
            )

    cascade_pl = (False, 0.0, 1.0)
    if len(all_cascades) >= 50:
        cascade_pl = verify_power_law(all_cascades)

    active_arr = np.array(active_counts, dtype=float)
    baseline = np.median(active_arr)
    avalanches = extract_cascades(active_arr, threshold=baseline + 0.5)
    avalanche_pl = (False, 0.0, 1.0)
    if len(avalanches) >= 50:
        avalanche_pl = verify_power_law(avalanches)

    print(f"\n  Results:")
    print(f"    σ = {mean_sigma:.3f} ± {np.std(final_sigmas):.3f} "
          f"({'PASS' if abs(mean_sigma - 1.0) < 0.5 else 'FAIL'})")
    print(f"    Active fraction: {mean_active:.1%} "
          f"({'SPARSE' if mean_active < 0.3 else 'DENSE'})")
    print(f"    Temporal cascades: {len(all_cascades)}, "
          f"α={cascade_pl[1]:.2f}, KS={cascade_pl[2]:.3f} "
          f"({'PASS' if cascade_pl[0] else 'INCONCLUSIVE'})")
    print(f"    Network avalanches: {len(avalanches)}, "
          f"α={avalanche_pl[1]:.2f}, KS={avalanche_pl[2]:.3f} "
          f"({'PASS' if avalanche_pl[0] else 'INCONCLUSIVE'})")

    return {
        "label": label,
        "sigma_mean": mean_sigma,
        "sigma_std": np.std(final_sigmas),
        "sigma_history": sigma_history,
        "frac_active": mean_active,
        "cascade_count": len(all_cascades),
        "cascade_pl": cascade_pl,
        "avalanche_count": len(avalanches),
        "avalanche_pl": avalanche_pl,
        "final_sigmas": final_sigmas,
        "active_counts": active_counts,
        "all_cascades": all_cascades,
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(n_steps: int = 10000, n_init: int = 64, do_plot: bool = False):
    print(f"Seed Criticality Validation: {n_init} nodes, {n_steps} steps")
    print(f"Sparse regime: threshold=0.5, drive=0.2, noise=0.05")
    print(f"Rules active: Rule 2a only")
    print("=" * 60)

    config = SeedConfig(
        n_init=n_init,
        k_neighbors=6,
        noise_floor=0.05,
        drive_amplitude=0.2,
        activation_threshold=0.5,
        learning_rate=0.01,
        w_max=2.0,
        prune_weight_threshold=0.005,
        prune_weight_window=200,
    )

    # Condition 1: Independent random drives (should work)
    def independent_drive_fn(step, nodes, rng):
        return independent_random_drives(list(nodes.keys()), rng)

    result_indep = run_condition(
        "Independent random drives",
        config, n_steps, independent_drive_fn, seed=42
    )

    # Condition 2: Correlated signal drives (known problem)
    def correlated_drive_fn(step, nodes, rng):
        freqs = {nid: n.frequency for nid, n in nodes.items()}
        return correlated_signal_drives(step * config.dt, freqs)

    result_signal = run_condition(
        "Correlated signal drives",
        config, n_steps, correlated_drive_fn, seed=42
    )

    # Summary comparison
    print(f"\n{'=' * 60}")
    print(f"COMPARISON:")
    print(f"  {'Metric':<25} {'Independent':>15} {'Correlated':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'σ (target=1.0)':<25} {result_indep['sigma_mean']:>15.3f} {result_signal['sigma_mean']:>15.3f}")
    print(f"  {'Active fraction':<25} {result_indep['frac_active']:>15.1%} {result_signal['frac_active']:>15.1%}")
    print(f"  {'Cascade power law':<25} {'PASS' if result_indep['cascade_pl'][0] else 'FAIL':>15} {'PASS' if result_signal['cascade_pl'][0] else 'FAIL':>15}")
    print(f"  {'Avalanche power law':<25} {'PASS' if result_indep['avalanche_pl'][0] else 'FAIL':>15} {'PASS' if result_signal['avalanche_pl'][0] else 'FAIL':>15}")

    if do_plot:
        _plot_comparison(result_indep, result_signal)


def _plot_comparison(r1, r2):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for r, color, label in [(r1, "blue", r1["label"]), (r2, "red", r2["label"])]:
        # σ over time
        steps = [s[0] for s in r["sigma_history"]]
        means = [s[1] for s in r["sigma_history"]]
        stds = [s[2] for s in r["sigma_history"]]
        axes[0, 0].plot(steps, means, color=color, lw=2, label=label)
        axes[0, 0].fill_between(steps,
            [m-s for m, s in zip(means, stds)],
            [m+s for m, s in zip(means, stds)], alpha=0.15, color=color)

        # σ distribution
        axes[0, 1].hist(r["final_sigmas"], bins=25, alpha=0.5, color=color, label=label)

        # Cascade sizes
        if r["all_cascades"]:
            unique, counts = np.unique(r["all_cascades"], return_counts=True)
            axes[1, 0].scatter(unique, counts, s=10, alpha=0.5, color=color, label=label)

        # Activity
        window = 100
        smoothed = np.convolve(
            np.array(r["active_counts"]) / len(r["nodes"]),
            np.ones(window)/window, mode="valid")
        axes[1, 1].plot(smoothed, color=color, alpha=0.7, label=label)

    axes[0, 0].axhline(1.0, color="gray", ls="--", alpha=0.5)
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("σ")
    axes[0, 0].set_title("Branching Ratio"); axes[0, 0].legend()

    axes[0, 1].axvline(1.0, color="gray", ls="--")
    axes[0, 1].set_xlabel("σ"); axes[0, 1].set_title("Final σ Distribution")
    axes[0, 1].legend()

    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Size"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Cascade Sizes"); axes[1, 0].legend()

    axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("Fraction active")
    axes[1, 1].set_title("Activity"); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("experiments/seed/01_criticality_validation.png", dpi=150)
    print(f"\nPlot saved to experiments/seed/01_criticality_validation.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run_experiment(n_steps=args.steps, n_init=args.nodes, do_plot=args.plot)
