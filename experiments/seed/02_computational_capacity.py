"""
Seed Architecture — Computational Capacity Tests

Three diagnostic tests to determine whether the self-organized critical
network is a useful computational substrate:

  Test 1: Echo State Property
    Does the network's internal state depend on input?
    Feed two different signals, measure state divergence.

  Test 2: Fading Memory
    How long does the network remember a single pulse?
    Inject a pulse, measure deviation from baseline over time.
    At σ=1 (critical), memory should be maximized.

  Test 3: Nonlinear Separation
    Can a linear readout classify different input patterns?
    Feed two classes of input, train linear regression on reservoir
    state to distinguish them.

If these pass, the network is a functional reservoir. If they fail,
criticality alone isn't enough.

Usage:
    python experiments/seed/02_computational_capacity.py [--nodes N] [--plot]
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from ro_framework.seed.node import OscillatoryNode, SeedConfig


# ---------------------------------------------------------------------------
# Network construction + step (same as experiment 01)
# ---------------------------------------------------------------------------

def build_ring_lattice(
    config: SeedConfig, seed: int = 42
) -> Dict[str, OscillatoryNode]:
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


def step_network(
    nodes: Dict[str, OscillatoryNode],
    drives: Dict[str, float],
    rng: np.random.Generator,
) -> None:
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


def get_state_vector(nodes: Dict[str, OscillatoryNode], node_order: List[str]) -> np.ndarray:
    """Extract ordered activation vector from network."""
    return np.array([nodes[nid].activation for nid in node_order])


def warmup_network(nodes, rng, n_steps=2000):
    """Run network with random drives to reach critical state."""
    for _ in range(n_steps):
        drives = {}
        for nid in nodes:
            if rng.random() < 0.05:
                drives[nid] = float(rng.uniform(0.5, 1.5))
        step_network(nodes, drives, rng)


# ---------------------------------------------------------------------------
# Test 1: Echo State Property
# ---------------------------------------------------------------------------

def test_echo_state(config: SeedConfig, seed: int = 42) -> dict:
    """Do different inputs produce different internal states?

    Method: Run two copies of the network from the same initial state.
    Feed signal A to one, signal B to the other. Measure how quickly
    and how far their internal states diverge.
    """
    print("\n=== Test 1: Echo State Property ===")

    rng = np.random.default_rng(seed)
    nodes = build_ring_lattice(config, seed=seed)
    node_order = sorted(nodes.keys())

    # Warmup to critical state
    warmup_network(nodes, rng, n_steps=2000)

    # Save state for cloning
    state_dict = {nid: n.to_dict() for nid, n in nodes.items()}

    # Clone: build second network from saved state
    nodes_b = {}
    for nid, d in state_dict.items():
        nodes_b[nid] = OscillatoryNode.from_dict(d, config)

    # Choose 4 sensory nodes
    sensory_ids = [node_order[0], node_order[len(node_order)//3],
                   node_order[2*len(node_order)//3], node_order[-1]]

    # Two different signals
    def signal_a(t):
        return 0.8 * np.sin(2 * np.pi * 0.05 * t)

    def signal_b(t):
        return 0.8 * np.sin(2 * np.pi * 0.2 * t)

    n_steps = 500
    divergences = []
    rng_a = np.random.default_rng(seed + 100)
    rng_b = np.random.default_rng(seed + 100)  # same noise sequence

    for step in range(n_steps):
        # Network A: signal A
        drives_a = {sid: signal_a(step) for sid in sensory_ids}
        step_network(nodes, drives_a, rng_a)

        # Network B: signal B
        drives_b = {sid: signal_b(step) for sid in sensory_ids}
        step_network(nodes_b, drives_b, rng_b)

        # Measure state divergence
        state_a = get_state_vector(nodes, node_order)
        state_b = get_state_vector(nodes_b, node_order)
        div = np.linalg.norm(state_a - state_b) / np.sqrt(len(node_order))
        divergences.append(div)

    divergences = np.array(divergences)
    mean_div = np.mean(divergences[100:])  # skip transient
    max_div = np.max(divergences)
    final_div = np.mean(divergences[-50:])

    print(f"  Mean divergence (after transient): {mean_div:.4f}")
    print(f"  Max divergence:  {max_div:.4f}")
    print(f"  Final divergence: {final_div:.4f}")

    passed = final_div > 0.01
    print(f"  {'PASS' if passed else 'FAIL'} — states {'diverge' if passed else 'do NOT diverge'} for different inputs")

    return {
        "divergences": divergences,
        "mean_div": mean_div,
        "max_div": max_div,
        "final_div": final_div,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 2: Fading Memory
# ---------------------------------------------------------------------------

def test_fading_memory(config: SeedConfig, seed: int = 42) -> dict:
    """How long does the network remember a single pulse?

    Method: Run to steady state, record baseline, inject a strong pulse
    into one sensory node, measure how long the network state deviates
    from a no-pulse control run.
    """
    print("\n=== Test 2: Fading Memory ===")

    rng = np.random.default_rng(seed)
    nodes = build_ring_lattice(config, seed=seed)
    node_order = sorted(nodes.keys())

    warmup_network(nodes, rng, n_steps=2000)

    # Clone for control (no pulse)
    state_dict = {nid: n.to_dict() for nid, n in nodes.items()}
    nodes_ctrl = {}
    for nid, d in state_dict.items():
        nodes_ctrl[nid] = OscillatoryNode.from_dict(d, config)

    # Inject pulse into one sensory node
    pulse_node = node_order[len(node_order) // 2]
    pulse_step = 0
    pulse_strength = 3.0

    n_steps = 300
    deviations = []
    rng_pulse = np.random.default_rng(seed + 200)
    rng_ctrl = np.random.default_rng(seed + 200)

    for step in range(n_steps):
        # Pulse network: strong drive on pulse_step, zero otherwise
        drives_p = {}
        if step == pulse_step:
            drives_p[pulse_node] = pulse_strength

        # Control: always zero drive
        drives_c = {}

        step_network(nodes, drives_p, rng_pulse)
        step_network(nodes_ctrl, drives_c, rng_ctrl)

        state_p = get_state_vector(nodes, node_order)
        state_c = get_state_vector(nodes_ctrl, node_order)
        dev = np.linalg.norm(state_p - state_c) / np.sqrt(len(node_order))
        deviations.append(dev)

    deviations = np.array(deviations)

    # Memory length: timesteps until deviation falls below 10% of peak
    peak_dev = np.max(deviations)
    threshold = 0.1 * peak_dev
    above_threshold = np.where(deviations > threshold)[0]
    memory_length = int(above_threshold[-1]) if len(above_threshold) > 0 else 0

    # Also measure half-life
    half_threshold = 0.5 * peak_dev
    above_half = np.where(deviations > half_threshold)[0]
    half_life = int(above_half[-1]) if len(above_half) > 0 else 0

    print(f"  Peak deviation: {peak_dev:.4f}")
    print(f"  Memory length (>10% of peak): {memory_length} steps")
    print(f"  Half-life (>50% of peak): {half_life} steps")
    print(f"  Deviation at step 50: {deviations[min(50, n_steps-1)]:.4f}")
    print(f"  Deviation at step 200: {deviations[min(200, n_steps-1)]:.4f}")

    passed = memory_length > 5
    print(f"  {'PASS' if passed else 'FAIL'} — network {'remembers' if passed else 'forgets immediately'} "
          f"(memory={memory_length} steps)")

    return {
        "deviations": deviations,
        "peak_dev": peak_dev,
        "memory_length": memory_length,
        "half_life": half_life,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 3: Nonlinear Separation
# ---------------------------------------------------------------------------

def _run_frequency_classification(
    config: SeedConfig,
    n_classes: int,
    n_trials_per_class: int,
    trial_length: int,
    seed: int,
) -> dict:
    """Run one frequency classification experiment.

    Args:
        n_classes: Number of distinct frequencies to classify.
        n_trials_per_class: Training+test trials per class.
        trial_length: Steps per trial.

    Returns:
        dict with train/test accuracy and class details.
    """
    rng = np.random.default_rng(seed)
    nodes = build_ring_lattice(config, seed=seed)
    node_order = sorted(nodes.keys())
    n_nodes = len(node_order)

    warmup_network(nodes, rng, n_steps=2000)

    sensory_ids = [node_order[0], node_order[len(node_order)//3],
                   node_order[2*len(node_order)//3], node_order[-1]]

    # Log-spaced frequencies from 0.02 to 0.5
    class_freqs = np.exp(np.linspace(np.log(0.02), np.log(0.5), n_classes))
    amp = 0.8

    # Generate trials: cycle through classes
    n_trials = n_classes * n_trials_per_class
    states = []
    labels = []

    for trial in range(n_trials):
        label = trial % n_classes
        freq = class_freqs[label]

        for step in range(trial_length):
            signal = amp * np.sin(2 * np.pi * freq * step)
            drives = {sid: signal for sid in sensory_ids}
            step_network(nodes, drives, rng)

        state = get_state_vector(nodes, node_order)
        states.append(state.copy())
        labels.append(label)

        # Brief reset between trials
        for _ in range(10):
            drives = {}
            for nid in nodes:
                if rng.random() < 0.05:
                    drives[nid] = float(rng.uniform(0.3, 0.8))
            step_network(nodes, drives, rng)

    states = np.array(states)
    labels = np.array(labels)

    # Train/test split (stratified: last 25% of each class)
    train_mask = np.zeros(n_trials, dtype=bool)
    test_mask = np.zeros(n_trials, dtype=bool)
    n_train_per = n_trials_per_class * 3 // 4
    for c in range(n_classes):
        c_idx = np.where(labels == c)[0]
        train_mask[c_idx[:n_train_per]] = True
        test_mask[c_idx[n_train_per:]] = True

    X_train, X_test = states[train_mask], states[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]

    # One-vs-rest ridge regression readout
    ridge_alpha = 1.0
    XtX_inv = np.linalg.solve(
        X_train.T @ X_train + ridge_alpha * np.eye(n_nodes),
        np.eye(n_nodes),
    )

    # One-hot targets
    Y_train_oh = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train_oh[i, y] = 1.0

    W = XtX_inv @ X_train.T @ Y_train_oh  # (n_nodes, n_classes)

    pred_train = np.argmax(X_train @ W, axis=1)
    pred_test = np.argmax(X_test @ W, axis=1)
    acc_train = np.mean(pred_train == y_train)
    acc_test = np.mean(pred_test == y_test)

    # Per-class accuracy
    per_class_acc = []
    for c in range(n_classes):
        c_mask = y_test == c
        if c_mask.sum() > 0:
            per_class_acc.append(np.mean(pred_test[c_mask] == c))
        else:
            per_class_acc.append(0.0)

    return {
        "n_classes": n_classes,
        "class_freqs": class_freqs,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "per_class_acc": per_class_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pred_test": pred_test,
        "y_test": y_test,
    }


def test_nonlinear_separation(config: SeedConfig, seed: int = 42) -> dict:
    """Can a linear readout classify distinct frequency inputs?

    Scales from 2 to 8 classes (log-spaced frequencies) to find
    where discrimination breaks down. Each class is a pure sine
    wave at a different frequency fed to sensory nodes.
    """
    print("\n=== Test 3: Multi-Class Frequency Discrimination ===")

    class_counts = [2, 4, 5, 6, 8]
    results = []

    for nc in class_counts:
        r = _run_frequency_classification(
            config,
            n_classes=nc,
            n_trials_per_class=40,
            trial_length=50,
            seed=seed,
        )
        results.append(r)

        freq_strs = ", ".join(f"{f:.3f}" for f in r["class_freqs"])
        per_class = " ".join(f"{a:.0%}" for a in r["per_class_acc"])
        print(f"  {nc} classes: train={r['acc_train']:.1%}  test={r['acc_test']:.1%}  "
              f"per-class=[{per_class}]")
        print(f"    freqs: [{freq_strs}]")

    # Summary thresholds

    passed = results[0]["acc_test"] > 0.7  # at least 2-class works
    hard_passed = results[-1]["acc_test"] > 0.5  # 8-class above chance (12.5%)

    print(f"\n  2-class: {'PASS' if passed else 'FAIL'} ({results[0]['acc_test']:.1%})")
    print(f"  8-class: {'PASS' if hard_passed else 'FAIL'} "
          f"({results[-1]['acc_test']:.1%}, chance=12.5%)")

    return {
        "results": results,
        "passed": passed,
        "hard_passed": hard_passed,
        "class_counts": class_counts,
        "accuracies": [r["acc_test"] for r in results],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(n_init: int = 64, do_plot: bool = False):
    print(f"Seed Computational Capacity Tests: {n_init} nodes")
    print(f"Sparse regime: threshold=0.5, drive=0.2, noise=0.05")
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

    t0 = time.time()

    r1 = test_echo_state(config, seed=42)
    r2 = test_fading_memory(config, seed=42)
    r3 = test_nonlinear_separation(config, seed=42)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({elapsed:.1f}s):")
    print(f"  Echo state:     {'PASS' if r1['passed'] else 'FAIL'} (divergence={r1['final_div']:.4f})")
    print(f"  Fading memory:  {'PASS' if r2['passed'] else 'FAIL'} (memory={r2['memory_length']} steps, half-life={r2['half_life']})")
    accs = r3["accuracies"]
    counts = r3["class_counts"]
    acc_summary = ", ".join(f"{c}cl={a:.0%}" for c, a in zip(counts, accs))
    print(f"  Separation:     {'PASS' if r3['passed'] else 'FAIL'} ({acc_summary})")

    all_pass = r1["passed"] and r2["passed"] and r3["passed"]
    print(f"\n  Overall: {'ALL PASS — functional reservoir' if all_pass else 'ISSUES FOUND'}")

    if do_plot:
        _plot_results(r1, r2, r3)

    return r1, r2, r3


def _plot_results(r1, r2, r3):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Echo state: divergence over time
    ax = axes[0]
    ax.plot(r1["divergences"], "b-", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("State divergence (RMS)")
    ax.set_title(f"Echo State ({'PASS' if r1['passed'] else 'FAIL'})")

    # Fading memory: deviation over time
    ax = axes[1]
    ax.plot(r2["deviations"], "g-", alpha=0.7)
    ax.axhline(0.1 * r2["peak_dev"], color="r", ls="--", alpha=0.5, label="10% threshold")
    ax.axvline(r2["memory_length"], color="orange", ls="--", alpha=0.5, label=f"memory={r2['memory_length']}")
    ax.set_xlabel("Steps after pulse")
    ax.set_ylabel("Deviation from control (RMS)")
    ax.set_title(f"Fading Memory ({'PASS' if r2['passed'] else 'FAIL'})")
    ax.legend()

    # Separation: accuracy vs number of classes
    ax = axes[2]
    ax.plot(r3["class_counts"], r3["accuracies"], "ro-", lw=2, markersize=8)
    for nc, acc in zip(r3["class_counts"], r3["accuracies"]):
        chance = 1.0 / nc
        ax.plot(nc, chance, "kx", markersize=8)
    ax.set_xlabel("Number of classes")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Frequency Discrimination ({'PASS' if r3['passed'] else 'FAIL'})")
    ax.legend(["Reservoir", "Chance"], loc="lower left")

    plt.tight_layout()
    plt.savefig("experiments/seed/02_computational_capacity.png", dpi=150)
    print(f"\nPlot saved to experiments/seed/02_computational_capacity.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run_experiment(n_init=args.nodes, do_plot=args.plot)
