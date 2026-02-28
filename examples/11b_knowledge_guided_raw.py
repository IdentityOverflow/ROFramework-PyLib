"""
Knowledge-Guided Training — Raw Data Experiment

Compares standard training vs. knowledge-guided training on modular addition.
Unlike example 11 (which failed because it used sum-averaged data), this version
feeds RAW per-pair activations to the KnowledgeTracker.

On raw data, the embedding noise causes calibration (C) to stay low even as 
correlation (ρ) rises, accurately reflecting the "memorized" state. 
The KnowledgeRegularizer should detect this state and increase weight decay,
hopefully accelerating grokking.

Requires: PyTorch
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
# Model and data
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
# Observer setup 
# ---------------------------------------------------------------------------

def compute_raw_hidden(model, p):
    """Compute raw hidden activations for all p^2 pairs."""
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    
    model.eval()
    with torch.no_grad():
        h = model.relu(model.fc1(torch.cat([
            model.embed_a(grid_a), model.embed_b(grid_b)
        ], -1))).cpu().numpy()

    return h, grid_a.cpu().numpy(), grid_b.cpu().numpy()

def setup_tracking(freq_indices, hidden_dim):
    """Create Observer and KnowledgeTracker."""
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
        log_capacity=10000, # Big enough to hold all 9409 pairs
    )
    tracker = KnowledgeTracker(observer, external_dofs=fourier_dofs)
    return observer, tracker, fourier_dofs, neuron_dofs

def populate_observations_raw(observer, h_raw, a_raw, b_raw, p, fourier_dofs, neuron_dofs, freq_indices):
    """Populate observation log with RAW per-pair data."""
    observer.clear_memory()
    sums = (a_raw + b_raw) % p
    
    # Process in batches to avoid slow python loops where possible
    # But since we have to make State objects, it's unavoidable.
    # To keep it fast enough for an eval loop, we'll subsample if needed,
    # but 9409 isn't too bad.
    
    # Subsample to speed up eval loop (take 1000 random pairs)
    idx = np.random.choice(len(h_raw), size=1000, replace=False)
    
    for i in idx:
        s = sums[i]
        ext_vals = {}
        for fi, k in enumerate(freq_indices):
            angle = 2 * np.pi * k * s / p
            ext_vals[fourier_dofs[2 * fi]] = float(np.sin(angle))
            ext_vals[fourier_dofs[2 * fi + 1]] = float(np.cos(angle))

        int_vals = {neuron_dofs[j]: float(h_raw[i, j]) for j in range(len(neuron_dofs))}

        observer.observation_log.append(ObservationPair(
            external_state=State(values=ext_vals),
            internal_state=State(values=int_vals),
            timestamp=float(i),
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
    if freq_indices is None:
        freq_indices = [7, 9, 22] # Use the known true frequencies

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
            memorized_multiplier=3.0,     # Push harder when memorized
            generalized_multiplier=1.0,   # Return to normal when generalized
            # Adjusted thresholds for raw data (correlations are lower)
            memorized_min_correlation=0.15,
            memorized_max_calibration=0.30,
        )

    metrics = []

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

            # Compute K on RAW data
            h_raw, a_raw, b_raw = compute_raw_hidden(model, p)
            populate_observations_raw(observer, h_raw, a_raw, b_raw, p, fourier_dofs, neuron_dofs, freq_indices)
            tracker.step(epoch)

            # Get knowledge state
            best_rho = 0.0
            best_cal = 0.0
            best_type = "n/a"
            for dof in fourier_dofs:
                latest = tracker.latest(dof)
                if latest and latest.correlation > best_rho:
                    best_rho = latest.correlation
                    best_cal = latest.calibration
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
                "best_cal": best_cal,
                "best_type": best_type,
                "wd": current_wd,
                "wd_mult": wd_mult,
            })

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(p=97, num_epochs=7500):
    print("=" * 80)
    print("Knowledge-Guided Training — RAW DATA Experiment")
    print("=" * 80)

    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    train_data = (train_a, train_b, train_y)
    test_data = (test_a, test_b, test_y)

    torch.manual_seed(42)
    model_init = ModularAdditionMLP(p).to(DEVICE)
    init_state = copy.deepcopy(model_init.state_dict())

    # --- Run 1: Baseline ---
    print("\n--- Run 1: Baseline (constant weight decay = 1.0) ---\n")
    model_base = ModularAdditionMLP(p).to(DEVICE)
    model_base.load_state_dict(copy.deepcopy(init_state))

    t0 = time.time()
    base_metrics = train_run(
        model_base, train_data, test_data, p,
        num_epochs=num_epochs, use_regularizer=False,
    )
    t_base = time.time() - t0

    # --- Run 2: K-guided (Raw) ---
    print("\n--- Run 2: K-guided (KnowledgeRegularizer on RAW data) ---\n")
    model_guided = ModularAdditionMLP(p).to(DEVICE)
    model_guided.load_state_dict(copy.deepcopy(init_state))

    t0 = time.time()
    guided_metrics = train_run(
        model_guided, train_data, test_data, p,
        num_epochs=num_epochs, use_regularizer=True,
    )
    t_guided = time.time() - t0

    # --- Results ---
    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs RAW K-Guided")
    print("=" * 80)
    print(f"\n  {'Epoch':>6} | {'Base test':>9} | {'K-guided test':>13} | "
          f"{'WD mult':>7} | {'K-Raw ρ':>7} | {'K-Raw C':>7} | {'K type':>10}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*13}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

    base_by_epoch = {m["epoch"]: m for m in base_metrics}
    guided_by_epoch = {m["epoch"]: m for m in guided_metrics}

    for epoch in sorted(set(base_by_epoch) | set(guided_by_epoch)):
        b = base_by_epoch.get(epoch, {})
        g = guided_by_epoch.get(epoch, {})
        b_test = f"{b.get('test_acc', 0):.0%}" if b else "—"
        g_test = f"{g.get('test_acc', 0):.0%}" if g else "—"
        wd_m = f"{g.get('wd_mult', 1.0):.1f}" if g else "—"
        rho = f"{g.get('best_rho', 0):.3f}" if g else "—"
        cal = f"{g.get('best_cal', 0):.3f}" if g else "—"
        ktype = g.get("best_type", "—") if g else "—"
        print(f"  {epoch:>6} | {b_test:>9} | {g_test:>13} | {wd_m:>7} | {rho:>7} | {cal:>7} | {ktype:>10}")

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
            print(f"  K-guided grokked {diff} epochs earlier ({diff/base_grok:.0%} faster) 🎉")
        elif diff < 0:
            print(f"  K-guided grokked {-diff} epochs later ({-diff/base_grok:.0%} slower) ❌")
        else:
            print("  Same grokking epoch")

if __name__ == "__main__":
    main()
