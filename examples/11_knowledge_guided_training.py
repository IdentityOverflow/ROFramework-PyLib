"""
Knowledge-Guided Training — Modular Addition Grokking Experiment

Compares standard training vs. knowledge-guided training on modular addition
(a + b mod 97). The KnowledgeRegularizer reads K(d_ext) from a KnowledgeTracker
and adjusts weight decay during training:

- When features are memorized (high ρ, low C): increase weight decay
- When features are generalized (high ρ, high C): decrease weight decay

Hypothesis: selectively increasing regularization pressure during memorization
should accelerate grokking (the transition from memorization to generalization).

Both experiments use the same random seed and initial model weights.

Requires: PyTorch (pip install torch)
Runtime:  ~1-2 minutes on GPU, ~5-10 minutes on CPU (two runs of 7500 epochs)
"""

import copy
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Requires PyTorch: pip install torch")
    sys.exit(1)

from ro_framework import Observer, PolarDoF, State
from ro_framework.knowledge.tracker import KnowledgeTracker
from ro_framework.observer.observer import ObservationPair
from ro_framework.integration.training import KnowledgeRegularizer

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
        return self.fc2(self.relu(self.fc1(
            torch.cat([self.embed_a(a), self.embed_b(b)], -1)
        )))


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
# Observer setup (same pattern as example 08, simplified)
# ---------------------------------------------------------------------------


def compute_sum_averaged_hidden(model, p):
    """Compute sum-class-averaged hidden activations."""
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    model.eval()
    with torch.no_grad():
        h = model.relu(model.fc1(torch.cat([
            model.embed_a(grid_a), model.embed_b(grid_b)
        ], -1))).cpu().numpy()

    h_avg = np.zeros((p, h.shape[1]))
    for s in range(p):
        h_avg[s] = h[sums == s].mean(axis=0)
    return h_avg


def discover_top_frequencies(model, p, top_n=3):
    """Find dominant Fourier frequencies via DFT of sum-averaged activations."""
    h_avg = compute_sum_averaged_hidden(model, p)
    dft = np.fft.fft(h_avg, axis=0)
    power = np.abs(dft) ** 2
    total = power.sum(axis=1)
    max_k = (p - 1) // 2
    freqs = [(k, total[k]) for k in range(1, max_k + 1)]
    freqs.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in freqs[:top_n]]


def setup_tracking(freq_indices, hidden_dim):
    """Create Observer, KnowledgeTracker, and optionally KnowledgeRegularizer."""
    fourier_dofs = []
    for k in freq_indices:
        fourier_dofs.append(PolarDoF(name=f"sin_{k}", pole_negative=-1, pole_positive=1))
        fourier_dofs.append(PolarDoF(name=f"cos_{k}", pole_negative=-1, pole_positive=1))

    neuron_dofs = [PolarDoF(name=f"neuron_{j}") for j in range(hidden_dim)]

    class _Dummy:
        def __call__(self, state):
            return State(values={d: 0.0 for d in neuron_dofs})

    observer = Observer(
        name="tracker",
        internal_dofs=neuron_dofs,
        external_dofs=fourier_dofs,
        world_model=_Dummy(),
        log_capacity=500,
    )
    tracker = KnowledgeTracker(observer, external_dofs=fourier_dofs)
    return observer, tracker, fourier_dofs, neuron_dofs


def populate_observations(observer, h_avg, p, fourier_dofs, neuron_dofs, freq_indices):
    """Populate observation log with sum-averaged data."""
    observer.clear_memory()
    for s in range(p):
        ext_vals = {}
        for fi, k in enumerate(freq_indices):
            angle = 2 * np.pi * k * s / p
            ext_vals[fourier_dofs[2 * fi]] = float(np.sin(angle))
            ext_vals[fourier_dofs[2 * fi + 1]] = float(np.cos(angle))

        int_vals = {neuron_dofs[j]: float(h_avg[s, j]) for j in range(len(neuron_dofs))}

        observer.observation_log.append(ObservationPair(
            external_state=State(values=ext_vals),
            internal_state=State(values=int_vals),
            timestamp=float(s),
        ))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_run(
    model, train_data, test_data, p,
    num_epochs=7500,
    eval_interval=250,
    use_regularizer=False,
    base_wd=1.0,
    freq_indices=None,
):
    """Run one training experiment, return epoch-by-epoch metrics."""
    if freq_indices is None:
        freq_indices = [1, 2, 3]

    train_a, train_b, train_y = train_data
    test_a, test_b, test_y = test_data
    hidden_dim = model.fc1.out_features

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=base_wd
    )
    criterion = nn.CrossEntropyLoss()

    observer, tracker, fourier_dofs, neuron_dofs = setup_tracking(freq_indices, hidden_dim)

    regularizer = None
    if use_regularizer:
        regularizer = KnowledgeRegularizer(
            tracker,
            base_weight_decay=base_wd,
            memorized_multiplier=3.0,
            generalized_multiplier=0.5,
            # Use slightly relaxed thresholds to detect memorization earlier
            memorized_min_correlation=0.4,
            memorized_max_calibration=0.4,
        )

    metrics = []
    freq_switch_epoch = None

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

            # Auto-discover frequencies after generalization starts
            if test_acc > 0.90 and freq_switch_epoch is None:
                new_freqs = discover_top_frequencies(model, p, top_n=3)
                if new_freqs != freq_indices:
                    freq_indices = new_freqs
                    observer, tracker, fourier_dofs, neuron_dofs = setup_tracking(freq_indices, hidden_dim)
                    if regularizer is not None:
                        regularizer = KnowledgeRegularizer(
                            tracker,
                            base_weight_decay=base_wd,
                            memorized_multiplier=3.0,
                            generalized_multiplier=0.5,
                            memorized_min_correlation=0.4,
                            memorized_max_calibration=0.4,
                        )
                    freq_switch_epoch = epoch

            # Compute K
            h_avg = compute_sum_averaged_hidden(model, p)
            populate_observations(observer, h_avg, p, fourier_dofs, neuron_dofs, freq_indices)
            tracker.step(epoch)

            # Get knowledge state
            best_rho = 0.0
            best_type = "n/a"
            for dof in fourier_dofs:
                latest = tracker.latest(dof)
                if latest and latest.correlation > best_rho:
                    best_rho = latest.correlation
                    best_type = latest.knowledge_type

            # Update regularizer and apply
            current_wd = base_wd
            wd_mult = 1.0
            if regularizer is not None:
                regularizer.update(epoch)
                current_wd = regularizer.get_weight_decay()
                wd_mult = regularizer.current_multiplier
                for pg in optimizer.param_groups:
                    pg["weight_decay"] = current_wd

            metrics.append({
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "best_rho": best_rho,
                "best_type": best_type,
                "wd": current_wd,
                "wd_mult": wd_mult,
            })

    return metrics, freq_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(p=97, num_epochs=7500):
    print("=" * 80)
    print("Knowledge-Guided Training Experiment")
    print(f"Modular addition (a + b mod {p}), {num_epochs} epochs")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # Create dataset (shared between both runs)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    train_data = (train_a, train_b, train_y)
    test_data = (test_a, test_b, test_y)

    # Create model with fixed seed (save initial state for both runs)
    torch.manual_seed(42)
    model_init = ModularAdditionMLP(p).to(DEVICE)
    init_state = copy.deepcopy(model_init.state_dict())

    # --- Run 1: Baseline ---
    print("\n--- Run 1: Baseline (constant weight decay = 1.0) ---\n")
    model_base = ModularAdditionMLP(p).to(DEVICE)
    model_base.load_state_dict(copy.deepcopy(init_state))

    t0 = time.time()
    base_metrics, base_freqs = train_run(
        model_base, train_data, test_data, p,
        num_epochs=num_epochs, use_regularizer=False,
    )
    t_base = time.time() - t0

    # --- Run 2: K-guided ---
    print("\n--- Run 2: K-guided (KnowledgeRegularizer) ---\n")
    model_guided = ModularAdditionMLP(p).to(DEVICE)
    model_guided.load_state_dict(copy.deepcopy(init_state))

    t0 = time.time()
    guided_metrics, guided_freqs = train_run(
        model_guided, train_data, test_data, p,
        num_epochs=num_epochs, use_regularizer=True,
    )
    t_guided = time.time() - t0

    # --- Results ---
    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs K-Guided")
    print("=" * 80)
    print(f"\n  {'Epoch':>6} | {'Base test':>9} | {'K-guided test':>13} | "
          f"{'WD mult':>7} | {'Best ρ':>6} | {'K type':>10}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*13}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}")

    # Align by epoch
    base_by_epoch = {m["epoch"]: m for m in base_metrics}
    guided_by_epoch = {m["epoch"]: m for m in guided_metrics}

    for epoch in sorted(set(base_by_epoch) | set(guided_by_epoch)):
        b = base_by_epoch.get(epoch, {})
        g = guided_by_epoch.get(epoch, {})
        b_test = f"{b.get('test_acc', 0):.0%}" if b else "—"
        g_test = f"{g.get('test_acc', 0):.0%}" if g else "—"
        wd_m = f"{g.get('wd_mult', 1.0):.1f}" if g else "—"
        rho = f"{g.get('best_rho', 0):.3f}" if g else "—"
        ktype = g.get("best_type", "—") if g else "—"
        print(f"  {epoch:>6} | {b_test:>9} | {g_test:>13} | {wd_m:>7} | {rho:>6} | {ktype:>10}")

    # Find grokking epoch (first epoch with test_acc >= 95%)
    def grok_epoch(metrics):
        for m in metrics:
            if m["test_acc"] >= 0.95:
                return m["epoch"]
        return None

    base_grok = grok_epoch(base_metrics)
    guided_grok = grok_epoch(guided_metrics)

    print(f"\n  Baseline grokking epoch (≥95% test):  {base_grok or 'never'}")
    print(f"  K-guided grokking epoch (≥95% test):  {guided_grok or 'never'}")

    if base_grok and guided_grok:
        diff = base_grok - guided_grok
        if diff > 0:
            print(f"  K-guided grokked {diff} epochs earlier ({diff/base_grok:.0%} faster)")
        elif diff < 0:
            print(f"  K-guided grokked {-diff} epochs later ({-diff/base_grok:.0%} slower)")
        else:
            print("  Same grokking epoch")

    print(f"\n  Baseline time:  {t_base:.1f}s")
    print(f"  K-guided time:  {t_guided:.1f}s")

    print(f"\n  Baseline final test acc:  {base_metrics[-1]['test_acc']:.1%}")
    print(f"  K-guided final test acc:  {guided_metrics[-1]['test_acc']:.1%}")

    print(f"\n  Baseline frequencies:  {base_freqs}")
    print(f"  K-guided frequencies:  {guided_freqs}")
    print()


if __name__ == "__main__":
    main()
