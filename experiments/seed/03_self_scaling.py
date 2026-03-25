"""
Seed Architecture — Self-Scaling Validation

Tests whether Rules 4 (recruit) and 5 (release) produce adaptive network
sizing in response to changing input complexity.

Protocol:
  Phase 1 (warmup): Single frequency signal → network should stabilize
  Phase 2 (growth): Add a second frequency → network should recruit nodes
  Phase 3 (release): Remove both signals → idle nodes should be released

We use SeedNetwork directly (not the bare-node loop from experiments 01-02)
since recruitment/release lives in the network orchestrator.

Usage:
    python experiments/seed/03_self_scaling.py [--steps N] [--nodes N]
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from ro_framework.seed.network import SeedNetwork
from ro_framework.seed.node import SeedConfig


# ---------------------------------------------------------------------------
# Sensor / Actuator
# ---------------------------------------------------------------------------

class FrequencySensor:
    """Maps a multi-frequency signal to per-node drives based on frequency proximity."""

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self.signal_freqs: list = []
        self.signal_amps: list = []

    def set_signals(self, freqs: list, amps: list):
        self.signal_freqs = list(freqs)
        self.signal_amps = list(amps)

    def __call__(
        self, external_input: np.ndarray, node_frequencies: Dict[str, float]
    ) -> Dict[str, float]:
        t = float(external_input[0])
        if not self.signal_freqs:
            return {nid: 0.0 for nid in node_frequencies}

        signal_freqs = np.array(self.signal_freqs)
        signal_vals = np.array([
            amp * np.sin(2 * np.pi * freq * t)
            for freq, amp in zip(self.signal_freqs, self.signal_amps)
        ])
        log_sf = np.log(np.clip(signal_freqs, 1e-6, None))

        drives = {}
        for nid, nf in node_frequencies.items():
            log_nf = np.log(max(nf, 1e-6))
            weights = np.exp(-0.5 * ((log_nf - log_sf) / self.bandwidth) ** 2)
            drives[nid] = float(np.dot(weights, signal_vals))
        return drives


class MeanActuator:
    """Returns mean activation as a 1-D output."""

    def __call__(self, node_activations: Dict[str, float]) -> np.ndarray:
        if not node_activations:
            return np.array([0.0])
        return np.array([np.mean(list(node_activations.values()))])


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def snapshot(net: SeedNetwork, label: str = "") -> dict:
    """Capture network state for logging."""
    sigmas = [n.branching_ratio for n in net.nodes.values()]
    n_active = sum(
        1 for n in net.nodes.values()
        if abs(n.activation) > net.config.activation_threshold
    )
    n_conns = sum(len(n.coupling_weights) for n in net.nodes.values()) // 2
    freqs = [n.frequency for n in net.nodes.values()]
    return {
        "label": label,
        "n_nodes": len(net.nodes),
        "n_active": n_active,
        "n_conns": n_conns,
        "sigma_mean": float(np.mean(sigmas)) if sigmas else 0.0,
        "sigma_std": float(np.std(sigmas)) if sigmas else 0.0,
        "freq_min": float(np.min(freqs)) if freqs else 0.0,
        "freq_max": float(np.max(freqs)) if freqs else 0.0,
        "freq_mean": float(np.mean(freqs)) if freqs else 0.0,
    }


def print_snapshot(s: dict, step: int, elapsed: float):
    print(f"  Step {step:5d} | nodes={s['n_nodes']:3d} | "
          f"σ={s['sigma_mean']:.3f}±{s['sigma_std']:.3f} | "
          f"active={s['n_active']:3d} | conns={s['n_conns']:4d} | "
          f"freq=[{s['freq_min']:.3f}, {s['freq_max']:.3f}] | "
          f"t={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_init: int = 16,
    phase1_steps: int = 500,
    phase2_steps: int = 1000,
    phase3_steps: int = 500,
    phase4_steps: int = 1000,
):
    print(f"Seed Self-Scaling Validation")
    print(f"  Initial nodes: {n_init}")
    print(f"  Phase 1 (single freq): {phase1_steps} steps")
    print(f"  Phase 2 (add freq):    {phase2_steps} steps")
    print(f"  Phase 3 (no signal):   {phase3_steps} steps")
    print(f"  Phase 4 (back to 1):   {phase4_steps} steps")
    print("=" * 70)

    config = SeedConfig(
        n_init=n_init,
        k_neighbors=4,
        n_seed_nodes=4,
        max_nodes=128,
        noise_floor=0.05,
        drive_amplitude=0.2,
        activation_threshold=0.5,
        learning_rate=0.01,
        w_max=2.0,
        prune_weight_threshold=0.005,
        prune_weight_window=200,
        # Rule 4/5 parameters
        recruit_mi_threshold=0.1,
        recruit_window=200,
        release_mi_threshold=0.05,
        release_window=200,
    )

    sensor = FrequencySensor(bandwidth=1.0)
    actuator = MeanActuator()
    net = SeedNetwork(config, sensor, actuator, seed=42)

    history = []
    log_interval = 100
    total_steps = phase1_steps + phase2_steps + phase3_steps + phase4_steps
    t0 = time.time()

    # --- Phase 1: Single frequency ---
    print(f"\n--- Phase 1: Single frequency (0.1 Hz) ---")
    sensor.set_signals([0.1], [1.0])

    for step in range(phase1_steps):
        t = step * config.dt
        net.step(np.array([t]))
        if (step + 1) % log_interval == 0:
            s = snapshot(net, "phase1")
            history.append((step + 1, s))
            print_snapshot(s, step + 1, time.time() - t0)

    n_after_phase1 = len(net.nodes)

    # --- Phase 2: Add second frequency ---
    print(f"\n--- Phase 2: Add second frequency (0.1 + 0.4 Hz) ---")
    sensor.set_signals([0.1, 0.4], [1.0, 1.0])

    for step in range(phase2_steps):
        t = (phase1_steps + step) * config.dt
        net.step(np.array([t]))
        global_step = phase1_steps + step + 1
        if global_step % log_interval == 0:
            s = snapshot(net, "phase2")
            history.append((global_step, s))
            print_snapshot(s, global_step, time.time() - t0)

    n_after_phase2 = len(net.nodes)

    # --- Phase 3: Remove all signals ---
    print(f"\n--- Phase 3: No signal (silence) ---")
    sensor.set_signals([], [])

    for step in range(phase3_steps):
        t = (phase1_steps + phase2_steps + step) * config.dt
        net.step(np.array([t]))
        global_step = phase1_steps + phase2_steps + step + 1
        if global_step % log_interval == 0:
            s = snapshot(net, "phase3")
            history.append((global_step, s))
            print_snapshot(s, global_step, time.time() - t0)

    n_after_phase3 = len(net.nodes)

    # --- Phase 4: Back to single frequency ---
    # Nodes recruited for the second frequency should be released
    # if they're not contributing to the single-frequency signal.
    print(f"\n--- Phase 4: Back to single frequency (0.1 Hz only) ---")
    sensor.set_signals([0.1], [1.0])

    offset = phase1_steps + phase2_steps + phase3_steps
    for step in range(phase4_steps):
        t = (offset + step) * config.dt
        net.step(np.array([t]))
        global_step = offset + step + 1
        if global_step % log_interval == 0:
            s = snapshot(net, "phase4")
            history.append((global_step, s))
            print_snapshot(s, global_step, time.time() - t0)

    n_after_phase4 = len(net.nodes)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"  Nodes after Phase 1 (single freq):  {n_after_phase1} (started at {n_init})")
    print(f"  Nodes after Phase 2 (two freqs):    {n_after_phase2}")
    print(f"  Nodes after Phase 3 (silence):      {n_after_phase3}")
    print(f"  Nodes after Phase 4 (single freq):  {n_after_phase4}")

    grew = n_after_phase2 > n_after_phase1
    shrank_silence = n_after_phase3 < n_after_phase2
    shrank_reduced = n_after_phase4 < n_after_phase3
    print(f"\n  Growth on new signal:        {'YES' if grew else 'NO'} ({n_after_phase2 - n_after_phase1:+d} nodes)")
    print(f"  Release on silence:          {'YES' if shrank_silence else 'NO'} ({n_after_phase3 - n_after_phase2:+d} nodes)")
    print(f"  Release on reduced signal:   {'YES' if shrank_reduced else 'NO'} ({n_after_phase4 - n_after_phase3:+d} nodes)")

    return history, net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--phase1", type=int, default=500)
    parser.add_argument("--phase2", type=int, default=1000)
    parser.add_argument("--phase3", type=int, default=500)
    parser.add_argument("--phase4", type=int, default=1000)
    args = parser.parse_args()
    run_experiment(
        n_init=args.nodes,
        phase1_steps=args.phase1,
        phase2_steps=args.phase2,
        phase3_steps=args.phase3,
        phase4_steps=args.phase4,
    )
