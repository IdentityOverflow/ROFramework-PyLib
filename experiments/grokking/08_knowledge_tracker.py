"""
Knowledge Tracker Example — Modular Addition & Grokking

Trains an MLP on modular addition (a + b mod p) and uses KnowledgeTracker
to monitor how the model's knowledge of Fourier features evolves during
training.  Demonstrates:

- Knowledge trajectory tracking over training epochs
- Grokking detection (memorization → generalization phase transition)
- Resonance detection (feature locking in before full generalization)
- Auto-discovery of which Fourier frequencies the model selects

Background:
    In modular addition, the "correct" internal representation uses Fourier
    features: sin(2πk·(a+b)/p) and cos(2πk·(a+b)/p) for frequency indices
    k = 1, 2, ...  During training with weight decay, the model first
    memorizes the training set (high train acc, low test acc), then
    "grokks" — suddenly generalizing as Fourier features lock in.

    After grokking, individual neurons become nearly pure cosine waves as
    functions of (a+b) mod p ("On the Mechanism and Dynamics of Modular
    Addition", He et al. 2025).  However, raw per-pair activations have
    ~89% within-sum-class variance from pair-specific embedding effects.

    We average activations by sum class s = (a+b) mod p before probing.
    This removes the irrelevant pair-specific noise and reveals the clean
    Fourier structure — giving per-neuron R ≈ 0.97 (vs 0.32 on raw pairs).

Requires: PyTorch (pip install torch)
Runtime:  ~30s on GPU, ~3-5 minutes on CPU (p=97, 7500 epochs)
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

# Auto-select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ModularAdditionMLP(nn.Module):
    """MLP for modular addition: (a, b) → (a + b) mod p.

    Architecture: Embedding(a) ⊕ Embedding(b) → Linear → ReLU → Linear → logits.
    Uses learned embeddings (not one-hot) following standard grokking setups.
    """

    def __init__(self, p: int, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.p = p
        self.hidden_dim = hidden_dim
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, p)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(torch.cat([self.embed_a(a), self.embed_b(b)], -1))))

    def get_hidden(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Extract hidden layer activations (after ReLU)."""
        with torch.no_grad():
            return self.relu(self.fc1(torch.cat([self.embed_a(a), self.embed_b(b)], -1)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_dataset(p: int, train_frac: float = 0.5, seed: int = 42):
    """Create train/test split for modular addition."""
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)

    split = int(len(all_pairs) * train_frac)
    train = all_pairs[:split]
    test = all_pairs[split:]

    def to_tensors(pairs):
        a = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=DEVICE)
        b = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=DEVICE)
        y = (a + b) % p
        return a, b, y

    return to_tensors(train), to_tensors(test)


# ---------------------------------------------------------------------------
# Sum-class averaging and frequency discovery
# ---------------------------------------------------------------------------


def compute_sum_averaged_hidden(model: ModularAdditionMLP, p: int) -> np.ndarray:
    """Compute hidden activations averaged by sum class.

    For each s in 0..p-1, averages the post-ReLU hidden activations over
    all p input pairs (a, b) with (a + b) ≡ s (mod p).  This removes
    pair-specific embedding noise (~89% of activation variance) and reveals
    the clean Fourier structure that neurons lock into after grokking.

    Returns:
        h_avg: Shape (p, hidden_dim) — one row per sum class.
    """
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    hidden = model.get_hidden(grid_a, grid_b).cpu().numpy()
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    h_avg = np.zeros((p, model.hidden_dim))
    for s in range(p):
        h_avg[s] = hidden[sums == s].mean(axis=0)
    return h_avg


def discover_top_frequencies(h_avg: np.ndarray, p: int, top_n: int = 5) -> list:
    """Discover which Fourier frequencies the model uses most.

    Computes the DFT of each neuron's sum-averaged activation pattern and
    ranks frequencies by total spectral power across all neurons.

    Returns:
        List of (frequency_k, power_fraction) tuples, sorted by power.
    """
    dft = np.fft.fft(h_avg, axis=0)
    power = np.abs(dft) ** 2

    # Sum power across all neurons for each frequency
    total_per_freq = power.sum(axis=1)

    # Frequencies 1..(p-1)/2 are unique (rest are conjugate mirrors)
    max_k = (p - 1) // 2
    non_dc_total = total_per_freq[1:max_k + 1].sum()

    results = []
    for k in range(1, max_k + 1):
        frac = total_per_freq[k] / non_dc_total if non_dc_total > 0 else 0
        results.append((k, frac))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Observer setup
# ---------------------------------------------------------------------------


def setup_observer_and_tracker(freq_indices: list, hidden_dim: int):
    """Create Observer and KnowledgeTracker for per-neuron Fourier tracking.

    External DoFs: Fourier features sin(2πk·s/p), cos(2πk·s/p).
    Internal DoFs: Individual neuron activations (sum-averaged).

    After grokking, individual neurons lock into clean cosine waves at
    specific frequencies, so per-neuron Pearson correlation on sum-averaged
    data reaches R ≈ 0.97 — well above the "strong" threshold.
    """
    fourier_dofs = []
    for k in freq_indices:
        fourier_dofs.append(PolarDoF(name=f"sin_{k}", pole_negative=-1.0, pole_positive=1.0))
        fourier_dofs.append(PolarDoF(name=f"cos_{k}", pole_negative=-1.0, pole_positive=1.0))

    neuron_dofs = [PolarDoF(name=f"neuron_{j}") for j in range(hidden_dim)]

    class _DummyMapping:
        def __call__(self, state: State) -> State:
            return State(values={d: 0.0 for d in neuron_dofs})

    observer = Observer(
        name="mod_add_tracker",
        internal_dofs=neuron_dofs,
        external_dofs=fourier_dofs,
        world_model=_DummyMapping(),
        log_capacity=500,  # p sum classes per step, plenty of room
    )

    tracker = KnowledgeTracker(observer, external_dofs=fourier_dofs)
    return observer, tracker, fourier_dofs, neuron_dofs


def populate_observations(
    observer: Observer,
    h_avg: np.ndarray,
    p: int,
    fourier_dofs: list,
    neuron_dofs: list,
    freq_indices: list,
):
    """Populate observation log with sum-averaged data.

    Each observation corresponds to one sum class s ∈ {0, ..., p-1}:
    - External state: Fourier feature values sin(2πk·s/p), cos(2πk·s/p)
    - Internal state: Sum-averaged neuron activations h_avg[s, :]
    """
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
# Printing helpers
# ---------------------------------------------------------------------------


def print_header(freq_indices: list) -> None:
    print(f"{'Epoch':>6} | {'Train':>6} | {'Test':>6} | ", end="")
    for k in freq_indices:
        print(f"{'R(sin'+str(k)+')':>8} {'R(cos'+str(k)+')':>8} | ", end="")
    kfirst = freq_indices[0]
    print(f"Type(sin_{kfirst})")
    print("-" * (36 + 20 * len(freq_indices)))


def print_row(epoch, train_acc, test_acc, tracker, fourier_dofs):
    print(f"{epoch:>6} | {train_acc:>5.0%} | {test_acc:>5.0%} | ", end="")
    for dof in fourier_dofs:
        latest = tracker.latest(dof)
        rho = latest.correlation if latest else 0.0
        print(f"{rho:>8.3f}", end=" ")
        if dof.name.startswith("cos_"):
            print("| ", end="")
    latest_first = tracker.latest(fourier_dofs[0])
    ktype = latest_first.knowledge_type if latest_first else "n/a"
    print(ktype)


def print_summary(tracker, fourier_dofs):
    print()
    print("=" * 70)
    print("Phase Transition Analysis")
    print("=" * 70)
    print()

    for dof in fourier_dofs:
        grok = tracker.detect_grokking(dof)
        resonance = tracker.detect_resonance(dof)
        forgetting = tracker.detect_forgetting(dof)

        print(f"  {dof.name}:")
        if grok is not None:
            print(f"    Grokking detected at epoch {grok}")
        else:
            print("    No grokking detected")
        if resonance:
            epochs_str = str(resonance[:5])
            print(f"    Resonance at epochs: {epochs_str}{'...' if len(resonance) > 5 else ''}")
        if forgetting:
            epochs_str = str(forgetting[:5])
            print(f"    Forgetting at epochs: {epochs_str}{'...' if len(forgetting) > 5 else ''}")

        traj = tracker.trajectory(dof)
        if traj:
            a = traj[-1].assessment
            best = a.best_internal_dof
            best_name = best.name if best else "none"
            print(f"    Final: R={a.correlation:.3f}, ε={a.systematic_error:.3f}, "
                  f"σ={a.random_error:.3f}, type={a.knowledge_type}, best_neuron={best_name}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    p: int = 97,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_freqs: int = 3,
    train_frac: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    num_epochs: int = 7500,
    eval_interval: int = 250,
) -> None:
    """Train modular addition MLP and track knowledge trajectory."""
    print("=" * 70)
    print("Knowledge Tracker — Modular Addition Grokking")
    print("=" * 70)
    print(f"  p={p}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    print(f"  train_frac={train_frac}, lr={lr}, weight_decay={weight_decay}")
    print(f"  epochs={num_epochs}, eval_interval={eval_interval}")
    print(f"  Device: {DEVICE}")
    print(f"  Method: per-neuron correlation on sum-averaged activations")
    print()

    torch.manual_seed(42)
    model = ModularAdditionMLP(p, embed_dim, hidden_dim).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p, train_frac)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Discover which frequencies the model will use (checked after first eval).
    # Start with k=1..num_freqs, then update after grokking if frequencies shift.
    freq_indices = list(range(1, num_freqs + 1))

    observer, tracker, fourier_dofs, neuron_dofs = setup_observer_and_tracker(
        freq_indices, hidden_dim
    )

    print(f"  Training pairs: {len(train_a)}/{p*p}")
    print(f"  Initial tracked frequencies: k={freq_indices}")
    print(f"  Observations per step: {p} (one per sum class)")
    print()

    print_header(freq_indices)

    discovered = False

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

            h_avg = compute_sum_averaged_hidden(model, p)

            # After generalization, discover which frequencies the model chose
            if test_acc > 0.95 and not discovered:
                discovered = True
                top_freqs = discover_top_frequencies(h_avg, p, top_n=num_freqs)
                new_indices = [k for k, _ in top_freqs]
                if new_indices != freq_indices:
                    print("\n  >>> Frequencies discovered after generalization: ", end="")
                    for k, frac in top_freqs:
                        print(f"k={k} ({frac:.0%})", end="  ")
                    print()
                    print(f"  >>> Switching tracked frequencies from {freq_indices} to {new_indices}")
                    print()

                    freq_indices = new_indices
                    observer, tracker, fourier_dofs, neuron_dofs = setup_observer_and_tracker(
                        freq_indices, hidden_dim
                    )
                    print_header(freq_indices)

            populate_observations(observer, h_avg, p, fourier_dofs, neuron_dofs, freq_indices)
            tracker.step(epoch)
            print_row(epoch, train_acc, test_acc, tracker, fourier_dofs)

    print_summary(tracker, fourier_dofs)
    print("Done.")


if __name__ == "__main__":
    main()
