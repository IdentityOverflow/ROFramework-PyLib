"""
Activation Tracker Example — Online Feature Discovery During Grokking

Trains the same modular addition MLP as example 08, but instead of
tracking pre-specified Fourier features, uses ActivationTracker to
discover emerging features via PCA on the hidden layer activations.

Demonstrates:
- Eigenvalue trajectory tracking (which directions grow in variance)
- Stability tracking (which directions persist across epochs)
- Readout alignment (which directions are used by the output layer)
- Feature discovery (stable, high-variance directions → candidate DoFs)
- Honest comparison with Phase 8a's sum-averaged per-neuron approach

Key insight:
    89% of raw activation variance is within-sum-class noise (pair-specific
    embedding effects), so PCA on raw activations finds embedding modes,
    not Fourier features directly.  However, the TEMPORAL dynamics — when
    eigenvalues spike and directions stabilize — clearly mark the grokking
    transition.  Readout alignment identifies task-relevant directions.

Requires: PyTorch (pip install torch)
Runtime:  ~30s on GPU, ~3-5 minutes on CPU
"""

import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("This example requires PyTorch. Install with: pip install torch")
    sys.exit(1)

from ro_framework.integration.activation_tracker import ActivationTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model and data (same as example 08)
# ---------------------------------------------------------------------------


class ModularAdditionMLP(nn.Module):
    def __init__(self, p, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.p = p
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, p)

    def forward(self, a, b):
        return self.fc2(self.relu(self.fc1(torch.cat([self.embed_a(a), self.embed_b(b)], -1))))


def make_dataset(p, train_frac=0.5, seed=42):
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)
    split = int(len(all_pairs) * train_frac)

    def to_tensors(pairs):
        a = torch.tensor([x[0] for x in pairs], dtype=torch.long, device=DEVICE)
        b = torch.tensor([x[1] for x in pairs], dtype=torch.long, device=DEVICE)
        return a, b, (a + b) % p

    return to_tensors(all_pairs[:split]), to_tensors(all_pairs[split:])


# ---------------------------------------------------------------------------
# Sum-averaged comparison (from example 08)
# ---------------------------------------------------------------------------


def compute_sum_averaged_correlations(model, p, top_k_freqs=3):
    """Compute per-neuron R on sum-averaged data for comparison."""
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    with torch.no_grad():
        hidden = model.relu(model.fc1(torch.cat([
            model.embed_a(grid_a), model.embed_b(grid_b)
        ], -1))).cpu().numpy()
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    h_avg = np.zeros((p, hidden.shape[1]))
    for s in range(p):
        h_avg[s] = hidden[sums == s].mean(axis=0)

    # DFT to find top frequencies
    dft = np.fft.fft(h_avg, axis=0)
    power = np.abs(dft) ** 2
    total_per_freq = power.sum(axis=1)
    max_k = (p - 1) // 2
    freq_power = [(k, total_per_freq[k]) for k in range(1, max_k + 1)]
    freq_power.sort(key=lambda x: x[1], reverse=True)
    top_freqs = [k for k, _ in freq_power[:top_k_freqs]]

    # Per-neuron R on sum-averaged for top frequencies
    s_vals = np.arange(p)
    results = {}
    for k in top_freqs:
        sin_t = np.sin(2 * np.pi * k * s_vals / p)
        cos_t = np.cos(2 * np.pi * k * s_vals / p)
        max_r_sin = max(abs(np.corrcoef(h_avg[:, j], sin_t)[0, 1]) for j in range(h_avg.shape[1]))
        max_r_cos = max(abs(np.corrcoef(h_avg[:, j], cos_t)[0, 1]) for j in range(h_avg.shape[1]))
        results[k] = (max_r_sin, max_r_cos)
    return results, top_freqs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    p=97,
    embed_dim=128,
    hidden_dim=128,
    num_epochs=7500,
    eval_interval=250,
    top_k=10,
):
    print("=" * 70)
    print("Activation Tracker — Online Feature Discovery")
    print("=" * 70)
    print(f"  p={p}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    print(f"  epochs={num_epochs}, eval_interval={eval_interval}")
    print(f"  Tracking top {top_k} principal directions")
    print(f"  Device: {DEVICE}")
    print()

    torch.manual_seed(42)
    model = ModularAdditionMLP(p, embed_dim, hidden_dim).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0
    )
    criterion = nn.CrossEntropyLoss()

    # Set up activation tracker on the ReLU layer
    tracker = ActivationTracker(
        model, "relu", top_k=top_k, device=str(DEVICE), readout_layer_name="fc2"
    )
    tracker.attach()

    # All input pairs for collection
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)

    print(f"  Training pairs: {len(train_a)}/{p*p}")
    print(f"  Collection: all {p*p} pairs per eval point")
    print()

    # Header
    print(f"{'Epoch':>6} | {'Train':>6} | {'Test':>6} | "
          f"{'EV1':>8} {'EV2':>8} {'EV3':>8} | "
          f"{'Stab1':>6} {'Stab2':>6} {'Stab3':>6} | "
          f"{'Align1':>7} {'Align2':>7} {'Align3':>7}")
    print("-" * 105)

    for epoch in range(num_epochs):
        model.train()
        loss = criterion(model(train_a, train_b), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(train_a, train_b).argmax(1) == train_y).float().mean().item()
                test_acc = (model(test_a, test_b).argmax(1) == test_y).float().mean().item()

            # Collect activations for PCA
            tracker.begin_collection()
            with torch.no_grad():
                model(grid_a, grid_b)  # hook fires, collects stats
            snap = tracker.end_collection(epoch)

            # Print top-3 eigenvalues, stabilities, alignments
            evs = [d.eigenvalue for d in snap.directions[:3]]
            stabs = [d.stability for d in snap.directions[:3]]
            aligns = [d.readout_alignment for d in snap.directions[:3]]

            ev_str = " ".join(f"{v:>8.2f}" for v in evs)
            stab_str = " ".join(f"{v:>6.3f}" if v is not None else f"{'n/a':>6}" for v in stabs)
            align_str = " ".join(f"{v:>7.3f}" if v is not None else f"{'n/a':>7}" for v in aligns)

            print(f"{epoch:>6} | {train_acc:>5.0%} | {test_acc:>5.0%} | "
                  f"{ev_str} | {stab_str} | {align_str}")

    # ===================================================================
    # Post-training analysis
    # ===================================================================

    print()
    print("=" * 70)
    print("Post-Training Analysis")
    print("=" * 70)

    # Eigenvalue spike detection
    print()
    print("Eigenvalue spikes (relative threshold 2.0x):")
    for i in range(min(5, top_k)):
        spike = tracker.detect_eigenvalue_spike(i, relative_threshold=2.0)
        traj = tracker.eigenvalue_trajectory(i)
        if traj:
            last_ev = traj[-1][1]
            if spike is not None:
                print(f"  Direction {i}: spike at epoch {spike}, final EV={last_ev:.2f}")
            else:
                print(f"  Direction {i}: no spike, final EV={last_ev:.2f}")

    # Feature discovery
    print()
    print("Discovered DoFs (min_stability=0.9, min_stable_epochs=3):")
    discovered = tracker.discover_dofs(min_stability=0.9, min_stable_epochs=3)
    if discovered:
        for dd in discovered:
            print(f"  {dd.dof.name}: EV={dd.eigenvalue:.2f}, "
                  f"stable for {dd.stability_epochs} epochs")
    else:
        print("  (none met criteria)")

    # Try with relaxed thresholds
    discovered_relaxed = tracker.discover_dofs(min_stability=0.8, min_stable_epochs=2)
    if len(discovered_relaxed) > len(discovered):
        print()
        print(f"With relaxed thresholds (0.8 stability, 2 epochs): "
              f"{len(discovered_relaxed)} directions found")

    # ===================================================================
    # Comparison with Phase 8a sum-averaged approach
    # ===================================================================

    print()
    print("=" * 70)
    print("Comparison: PCA vs Sum-Averaged (Phase 8a)")
    print("=" * 70)
    print()

    sum_avg_results, top_freqs = compute_sum_averaged_correlations(model, p, top_k_freqs=3)
    print("Sum-averaged per-neuron R (Phase 8a approach):")
    for k in top_freqs:
        r_sin, r_cos = sum_avg_results[k]
        print(f"  k={k}: R(sin)={r_sin:.3f}, R(cos)={r_cos:.3f}")

    print()
    print("PCA directions (this approach):")
    print(f"  Top-3 eigenvalues: {[f'{d.eigenvalue:.1f}' for d in snap.directions[:3]]}")
    print(f"  Top-3 stability:   {[f'{d.stability:.3f}' if d.stability else 'n/a' for d in snap.directions[:3]]}")
    print(f"  Top-3 readout alignment: {[f'{d.readout_alignment:.3f}' if d.readout_alignment else 'n/a' for d in snap.directions[:3]]}")
    print()
    print("Key difference:")
    print("  Sum-averaging removes 89% within-class noise → per-neuron R ≈ 0.97")
    print("  PCA on raw activations captures embedding modes, not Fourier features")
    print("  BUT: eigenvalue dynamics and stability clearly mark grokking transition")
    print("  AND: readout alignment identifies task-relevant directions")

    tracker.detach()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
