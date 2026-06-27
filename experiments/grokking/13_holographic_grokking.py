"""
Holographic Grokking — handing the model the resonant code for free.

Hypothesis (from the holography ↔ HRR connection):
    Grokking on modular addition is the *search* for a holographic/resonant
    encoding.  A standard MLP spends thousands of epochs discovering that
    integers want to live as phasors e^{i·2πk·n/p} on a circle, because in
    that code addition becomes phase addition (rotation), which a linear
    readout can solve.  Nanda et al. (2023) and He et al. (2026) showed the
    grokked network reconstructs exactly this Fourier structure.

    Holographic Reduced Representations (Plate, 1995) — specifically the
    frequency-domain variant FHRR — *build that code in by construction*:
    items are unit-magnitude phasors, binding is phase addition.  So if we
    encode a and b as phasors and BIND them (circular convolution = complex
    multiply = phase add), the bound vector is already the phasor of (a+b).
    The only thing left to learn is a linear readout — a convex problem.

    Prediction: K(d_ext) of the pre-readout representation should be "strong"
    at epoch 0 (the resonant basis is built in, not discovered), and test
    accuracy should follow within tens of epochs — versus thousands for the
    standard MLP.  We quantify that gap.

Honest framing:
    The holographic arm is handed the answer (the Fourier basis), so ρ≈1 is
    true *by construction*, not a discovery.  The scientific content is the
    comparison of trajectories: epochs-to-strong and epochs-to-grok with the
    code discovered (baseline) vs. supplied (holographic).  This measures the
    cost of the search that grokking performs.  It also removes the 89%
    within-sum-class activation noise the baseline fights — the holographic
    features are an *exact* function of the sum class.

This connects to:
    - Phase 8a: feature-level knowledge precedes behavioral generalization.
      Here that lag collapses to almost nothing once the search is removed.
    - Phase 8c (negative result): K-guided training was fooled by the
      feature/behavioral lag.  That lag is the search cost we delete here.

Requires: PyTorch.   Runtime: baseline ~30-60s on GPU, holographic ~1s.
"""

import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("This example requires PyTorch. Install with: pip install torch")
    sys.exit(1)

from ro_framework import Observer, PolarDoF, State
from ro_framework.knowledge.tracker import KnowledgeTracker
from ro_framework.observer.observer import ObservationPair

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# FHRR encoding and binding  (the holographic core)
# ---------------------------------------------------------------------------


def phasor_encode(n: np.ndarray, freqs: list, p: int) -> np.ndarray:
    """Encode integers as concatenated phasors: [cos(2πk·n/p), sin(2πk·n/p)].

    This is the frequency-domain HRR (FHRR) representation of an integer on
    the cyclic group Z/pZ.  Shape: (len(n), 2*len(freqs)).
    """
    ang = 2 * np.pi * np.outer(n, freqs) / p          # (N, K)
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)  # (N, 2K)


def bind_features(a: np.ndarray, b: np.ndarray, freqs: list, p: int) -> np.ndarray:
    """Bind encodings of a and b by phase addition (circular convolution).

    e^{iθ_a}·e^{iθ_b} = e^{i(θ_a+θ_b)}.  With θ = 2πk·n/p, the bound phase is
    2πk·(a+b)/p — i.e. the bound vector is the phasor of (a+b) mod p, WITHOUT
    the model ever being shown the sum.  We compute it from a and b alone via
    the angle-addition identity, to stay faithful to "binding".
    """
    ta = 2 * np.pi * np.outer(a, freqs) / p           # (N, K)
    tb = 2 * np.pi * np.outer(b, freqs) / p
    s = ta + tb
    return np.concatenate([np.cos(s), np.sin(s)], axis=1)  # (N, 2K)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_dataset(p: int, train_frac: float = 0.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    return pairs[:split], pairs[split:]


# ---------------------------------------------------------------------------
# Shared knowledge-probe scaffolding
# ---------------------------------------------------------------------------


def make_observer_tracker(freq_indices: list, n_internal: int):
    """Observer with Fourier external DoFs and `n_internal` internal probes."""
    fourier_dofs = []
    for k in freq_indices:
        fourier_dofs.append(PolarDoF(name=f"sin_{k}", pole_negative=-1.0, pole_positive=1.0))
        fourier_dofs.append(PolarDoF(name=f"cos_{k}", pole_negative=-1.0, pole_positive=1.0))
    internal_dofs = [PolarDoF(name=f"unit_{j}") for j in range(n_internal)]

    class _Dummy:
        def __call__(self, state: State) -> State:
            return State(values={d: 0.0 for d in internal_dofs})

    observer = Observer(
        name="holo_probe",
        internal_dofs=internal_dofs,
        external_dofs=fourier_dofs,
        world_model=_Dummy(),
        log_capacity=500,
    )
    return observer, KnowledgeTracker(observer, external_dofs=fourier_dofs), fourier_dofs, internal_dofs


def populate(observer, repr_by_sum, p, fourier_dofs, internal_dofs, freq_indices):
    """One observation per sum class: (ideal Fourier features, representation)."""
    observer.clear_memory()
    for s in range(p):
        ext = {}
        for fi, k in enumerate(freq_indices):
            ang = 2 * np.pi * k * s / p
            ext[fourier_dofs[2 * fi]] = float(np.sin(ang))
            ext[fourier_dofs[2 * fi + 1]] = float(np.cos(ang))
        internal = {internal_dofs[j]: float(repr_by_sum[s, j]) for j in range(len(internal_dofs))}
        observer.observation_log.append(ObservationPair(
            external_state=State(values=ext),
            internal_state=State(values=internal),
            timestamp=float(s),
        ))


def epochs_to_strong(tracker, dof):
    for tp in tracker.trajectory(dof):
        if tp.assessment.knowledge_type == "strong":
            return tp.epoch
    return None


# ---------------------------------------------------------------------------
# Holographic arm
# ---------------------------------------------------------------------------


def run_holographic(p, train, test, probe_freqs, num_epochs=200, eval_interval=10, lr=1e-2):
    """Fixed FHRR binding → trainable linear readout.  No learned encoder."""
    print("\n" + "=" * 70)
    print("HOLOGRAPHIC ARM — FHRR binding (fixed) + linear readout (trained)")
    print("=" * 70)

    # Full holographic code: all unique frequencies k = 1..(p-1)/2.
    freqs = list(range(1, (p - 1) // 2 + 1))

    # --- Sanity check: binding a,b reproduces the phasor of (a+b) mod p ---
    a_all = np.array([x for x in range(p) for _ in range(p)])
    b_all = np.array([y for _ in range(p) for y in range(p)])
    bound = bind_features(a_all, b_all, freqs, p)
    direct = phasor_encode((a_all + b_all) % p, freqs, p)
    bind_err = float(np.abs(bound - direct).max())
    # Within-sum-class variance of the bound features (the noise baseline fights)
    sums = (a_all + b_all) % p
    within = np.mean([bound[sums == s].var(axis=0).mean() for s in range(p)])
    print(f"  bind(a,b) == phasor(a+b) ?   max abs error = {bind_err:.2e}")
    print(f"  within-sum-class feature variance = {within:.2e}  (baseline ≈ 0.89 of total)")
    print(f"  feature dim (2 × {len(freqs)} freqs) = {bound.shape[1]}")

    def feats(pairs):
        a = np.array([x for x, _ in pairs]); b = np.array([y for _, y in pairs])
        X = bind_features(a, b, freqs, p)
        y = (a + b) % p
        return (torch.tensor(X, dtype=torch.float32, device=DEVICE),
                torch.tensor(y, dtype=torch.long, device=DEVICE))

    Xtr, ytr = feats(train)
    Xte, yte = feats(test)

    torch.manual_seed(0)
    readout = nn.Linear(Xtr.shape[1], p).to(DEVICE)
    opt = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    # Pre-readout representation by sum class (exact — depends only on s)
    repr_by_sum = phasor_encode(np.arange(p), freqs, p)  # (p, 2K)
    observer, tracker, fourier_dofs, internal_dofs = make_observer_tracker(
        probe_freqs, repr_by_sum.shape[1]
    )

    print(f"\n  {'Epoch':>6} | {'Train':>6} | {'Test':>6} | {'R(cos1)':>8} | Type(cos_1)")
    print("  " + "-" * 52)

    grok_epoch = None
    for epoch in range(num_epochs + 1):
        readout.train()
        loss = crit(readout(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % eval_interval == 0:
            readout.eval()
            with torch.no_grad():
                tr = (readout(Xtr).argmax(1) == ytr).float().mean().item()
                te = (readout(Xte).argmax(1) == yte).float().mean().item()
            populate(observer, repr_by_sum, p, fourier_dofs, internal_dofs, probe_freqs)
            tracker.step(epoch)
            if te > 0.95 and grok_epoch is None:
                grok_epoch = epoch
            a = tracker.latest(fourier_dofs[1])  # cos_1
            print(f"  {epoch:>6} | {tr:>5.0%} | {te:>5.0%} | {a.correlation:>8.3f} | {a.knowledge_type}")

    strong = epochs_to_strong(tracker, fourier_dofs[1])
    return {"arm": "holographic", "grok_epoch": grok_epoch, "strong_epoch": strong,
            "within_var": within, "tracker": tracker, "fourier_dofs": fourier_dofs}


# ---------------------------------------------------------------------------
# Baseline arm  (standard MLP — the code is discovered, not given)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, p, embed=128, hidden=128):
        super().__init__()
        self.p, self.hidden_dim = p, hidden
        self.ea = nn.Embedding(p, embed); self.eb = nn.Embedding(p, embed)
        self.fc1 = nn.Linear(2 * embed, hidden); self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, a, b):
        return self.fc2(self.relu(self.fc1(torch.cat([self.ea(a), self.eb(b)], -1))))

    def hidden_act(self, a, b):
        with torch.no_grad():
            return self.relu(self.fc1(torch.cat([self.ea(a), self.eb(b)], -1)))


def sum_averaged_hidden(model, p):
    ga = torch.arange(p, device=DEVICE).repeat_interleave(p)
    gb = torch.arange(p, device=DEVICE).repeat(p)
    h = model.hidden_act(ga, gb).cpu().numpy()
    s = ((ga + gb) % p).cpu().numpy()
    out = np.zeros((p, model.hidden_dim))
    for k in range(p):
        out[k] = h[s == k].mean(axis=0)
    return out


def run_baseline(p, train, test, probe_freqs, num_epochs=5000, eval_interval=250,
                 lr=1e-3, weight_decay=1.0):
    print("\n" + "=" * 70)
    print("BASELINE ARM — standard MLP (resonant code must be discovered)")
    print("=" * 70)

    def to_t(pairs):
        a = torch.tensor([x for x, _ in pairs], dtype=torch.long, device=DEVICE)
        b = torch.tensor([y for _, y in pairs], dtype=torch.long, device=DEVICE)
        return a, b, (a + b) % p

    tra, trb, try_ = to_t(train)
    tea, teb, tey = to_t(test)

    torch.manual_seed(42)
    model = MLP(p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    observer, tracker, fourier_dofs, internal_dofs = make_observer_tracker(probe_freqs, model.hidden_dim)

    print(f"\n  {'Epoch':>6} | {'Train':>6} | {'Test':>6} | {'R(cos1)':>8} | Type(cos_1)")
    print("  " + "-" * 52)

    grok_epoch = None
    for epoch in range(num_epochs + 1):
        model.train()
        loss = crit(model(tra, trb), try_)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                tr = (model(tra, trb).argmax(1) == try_).float().mean().item()
                te = (model(tea, teb).argmax(1) == tey).float().mean().item()
            populate(observer, sum_averaged_hidden(model, p), p, fourier_dofs, internal_dofs, probe_freqs)
            tracker.step(epoch)
            if te > 0.95 and grok_epoch is None:
                grok_epoch = epoch
            a = tracker.latest(fourier_dofs[1])
            print(f"  {epoch:>6} | {tr:>5.0%} | {te:>5.0%} | {a.correlation:>8.3f} | {a.knowledge_type}")

    strong = epochs_to_strong(tracker, fourier_dofs[1])
    return {"arm": "baseline", "grok_epoch": grok_epoch, "strong_epoch": strong,
            "within_var": None, "tracker": tracker, "fourier_dofs": fourier_dofs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(p=97, run_baseline_arm=True):
    probe_freqs = [1, 2, 3]
    print("=" * 70)
    print("Holographic Grokking — resonant code: discovered vs. supplied")
    print("=" * 70)
    print(f"  p={p}   device={DEVICE}   probing frequencies k={probe_freqs}")

    train, test = make_dataset(p)

    holo = run_holographic(p, train, test, probe_freqs)
    base = run_baseline(p, train, test, probe_freqs) if run_baseline_arm else None

    print("\n" + "=" * 70)
    print("SUMMARY — cost of discovering the holographic code")
    print("=" * 70)
    print(f"\n  {'arm':>14} | {'epochs→K-strong':>16} | {'epochs→grok':>12}")
    print("  " + "-" * 48)
    for r in [holo, base]:
        if r is None:
            continue
        s = r["strong_epoch"]; g = r["grok_epoch"]
        print(f"  {r['arm']:>14} | {('—' if s is None else s):>16} | {('—' if g is None else g):>12}")

    print("\n  Reading:")
    print("  • Holographic: K is strong from epoch 0 — the Fourier basis is built in")
    print("    by construction (ρ≈1), and the linear readout generalizes almost at")
    print("    once.  Within-sum-class feature variance is ~0; no noise to fight.")
    if base is not None:
        print("  • Baseline: the same K-strong state and generalization arrive thousands")
        print("    of epochs later — that gap IS the search holography hands you for free.")
    print("\n  Caveat: the holographic arm is given the answer; ρ≈1 is by construction,")
    print("  not a discovery.  The result is the *trajectory gap*, which quantifies the")
    print("  cost of the resonance search that grokking performs.")
    print("\nDone.")


if __name__ == "__main__":
    main()
