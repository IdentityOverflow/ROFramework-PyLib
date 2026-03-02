"""
Denoising Experiment 10d — Cointegration / Slow Feature Analysis (SFA)

Compares approaches to discovering features in grokked modular addition,
focusing on the econometrics/SFA approach. Finds directions that have
high variance in the final state but low variance in their recent temporal change
(i.e., features that have "settled" while embedding noise continues to drift).

1. Raw PCA (baseline)
2. Readout-projected PCA
3. Cointegration / SFA (Final Variance vs Recent Change Variance)
"""

import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    from scipy.linalg import eigh
except ImportError:
    print("Requires PyTorch and SciPy")
    sys.exit(1)

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

def collect_activations(model, p):
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    model.eval()
    with torch.no_grad():
        h = model.relu(model.fc1(torch.cat([
            model.embed_a(grid_a), model.embed_b(grid_b)
        ], -1))).cpu().numpy()

    return h, sums, grid_a.cpu().numpy(), grid_b.cpu().numpy()

def discover_frequencies(h, sums, p, top_n=5):
    h_avg = np.zeros((p, h.shape[1]))
    for s in range(p):
        h_avg[s] = h[sums == s].mean(axis=0)

    dft = np.fft.fft(h_avg, axis=0)
    power = np.abs(dft) ** 2
    total_per_freq = power.sum(axis=1)

    max_k = (p - 1) // 2
    freq_power = [(k, total_per_freq[k]) for k in range(1, max_k + 1)]
    freq_power.sort(key=lambda x: x[1], reverse=True)
    top_freqs = [k for k, _ in freq_power[:top_n]]

    return h_avg, top_freqs

# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

def raw_pca(h, n_components=20):
    h_centered = h - h.mean(axis=0)
    cov = (h_centered.T @ h_centered) / (len(h) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, idx].T, eigenvalues[idx]

def readout_projected_pca(h, model, n_components=20):
    W = model.fc2.weight.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    coords = (h - h.mean(axis=0)) @ Vt.T
    cov = (coords.T @ coords) / (len(coords) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]

    directions_in_proj = eigenvectors[:, idx].T
    directions_original = directions_in_proj @ Vt
    norms = np.linalg.norm(directions_original, axis=1, keepdims=True)
    directions_original = directions_original / np.clip(norms, 1e-12, None)

    return directions_original, eigenvalues[idx], Vt

def cointegration_sfa(h_final, h_earlier, n_components=20):
    """
    Finds features that have high variance in the final state (they matter)
    but LOW variance in their change over recent epochs (they have settled).
    """
    # Covariance of final state
    h_centered = h_final - h_final.mean(axis=0)
    cov_final = (h_centered.T @ h_centered) / (len(h_final) - 1)
    
    # Covariance of temporal change
    diff = h_final - h_earlier
    diff_centered = diff - diff.mean(axis=0)
    cov_diff = (diff_centered.T @ diff_centered) / (len(diff) - 1)
    
    # Regularize cov_diff to avoid singularity
    cov_diff += np.eye(cov_diff.shape[0]) * 1e-6
    
    # Solve Generalized Eigenvalue Problem: C_final * v = lambda * C_diff * v
    eigenvalues, eigenvectors = eigh(cov_final, cov_diff)
    
    # Largest eigenvalues mean high final variance relative to change variance
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    directions = eigenvectors[:, idx].T
    
    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    
    return directions, eigenvalues[idx]

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_directions(directions, h, sums, p, top_freqs, limit=None):
    if limit is not None:
        directions = directions[:limit]
    
    s_vals = np.arange(p)
    results = []

    for i, d in enumerate(directions):
        proj = h @ d
        proj_avg = np.zeros(p)
        for s in range(p):
            proj_avg[s] = proj[sums == s].mean()

        best_raw_r = 0.0
        best_raw_k = 0
        for k in top_freqs:
            sin_t = np.sin(2 * np.pi * k * sums / p)
            cos_t = np.cos(2 * np.pi * k * sums / p)
            r_sin = abs(np.corrcoef(proj, sin_t)[0, 1])
            r_cos = abs(np.corrcoef(proj, cos_t)[0, 1])
            r = max(r_sin, r_cos)
            if r > best_raw_r:
                best_raw_r = r
                best_raw_k = k

        best_avg_r = 0.0
        best_avg_k = 0
        for k in top_freqs:
            sin_t = np.sin(2 * np.pi * k * s_vals / p)
            cos_t = np.cos(2 * np.pi * k * s_vals / p)
            r_sin = abs(np.corrcoef(proj_avg, sin_t)[0, 1])
            r_cos = abs(np.corrcoef(proj_avg, cos_t)[0, 1])
            r = max(r_sin, r_cos)
            if r > best_avg_r:
                best_avg_r = r
                best_avg_k = k

        results.append({
            "idx": i,
            "raw_r": best_raw_r,
            "raw_k": best_raw_k,
            "avg_r": best_avg_r,
            "avg_k": best_avg_k,
            "proj_var": np.var(proj),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(p=97, num_epochs=7500):
    print("=" * 70)
    print("Denoising Experiment 10d: Cointegration / Slow Feature Analysis")
    print("=" * 70)

    torch.manual_seed(42)
    model = ModularAdditionMLP(p).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0
    )
    criterion = nn.CrossEntropyLoss()

    snapshot_epochs = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
    snapshots = {}
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)

    for epoch in range(num_epochs):
        model.train()
        loss = criterion(model(train_a, train_b), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                h_snap = model.relu(model.fc1(torch.cat([
                    model.embed_a(grid_a), model.embed_b(grid_b)
                ], -1))).cpu().numpy()
            snapshots[epoch] = h_snap

        if epoch % 1000 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(train_a, train_b).argmax(1) == train_y).float().mean().item()
                test_acc = (model(test_a, test_b).argmax(1) == test_y).float().mean().item()
            print(f"  Epoch {epoch:>5}: train={train_acc:.0%}, test={test_acc:.0%}")

    print()

    h, sums, _, _ = collect_activations(model, p)
    h_avg, top_freqs = discover_frequencies(h, sums, p, top_n=5)
    total_var = np.var(h, axis=0).sum()

    n_components = 20

    print("Approach 1: Raw PCA")
    pca_dirs, pca_evals = raw_pca(h, n_components)
    pca_results = evaluate_directions(pca_dirs, h, sums, p, top_freqs)
    
    print("Approach 2: Readout-Projected PCA")
    rp_dirs, _, _ = readout_projected_pca(h, model, n_components)
    rp_results = evaluate_directions(rp_dirs, h, sums, p, top_freqs)

    print()
    print("=" * 70)
    print("Approach 3: Cointegration / Slow Feature Analysis")
    print("=" * 70)
    
    best_c_sfa_overall = 0.0
    best_c_sfa_results = None
    best_window = ""
    
    # We compare h to earlier snapshots to find which window isolates settled features best
    for earlier_epoch in [2000, 3000, 4000, 5000, 6000]:
        h_earlier = snapshots[earlier_epoch]
        c_dirs, c_evals = cointegration_sfa(h, h_earlier, n_components)
        c_results = evaluate_directions(c_dirs, h, sums, p, top_freqs)
        
        best_r = max(r["raw_r"] for r in c_results)
        best_avg = max(r["avg_r"] for r in c_results)
        window_name = f"{earlier_epoch} → Final"
        
        print(f"  Window {window_name:>15}: best raw R = {best_r:.3f}, best avg R = {best_avg:.3f}")
        
        if best_r > best_c_sfa_overall:
            best_c_sfa_overall = best_r
            best_c_sfa_results = c_results
            best_window = window_name

    print()
    print(f"  Best window: {best_window}")
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Settle Score':>12} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}")
    
    # Re-run for best window to get eigenvalues/scores
    earlier_epoch = int(best_window.split(' ')[0])
    c_dirs, c_evals = cointegration_sfa(h, snapshots[earlier_epoch], n_components)
    c_results = evaluate_directions(c_dirs, h, sums, p, top_freqs)
    
    c_sorted = sorted(c_results, key=lambda x: x["raw_r"], reverse=True)
    for r in c_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        score = c_evals[r['idx']]
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {score:>12.2f} | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | {r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    print()
    print("=" * 70)
    print("Summary: Best Raw R")
    print("=" * 70)
    best_pca = max(r["raw_r"] for r in pca_results)
    best_rp = max(r["raw_r"] for r in rp_results)

    print(f"  Raw PCA:               best raw R = {best_pca:.3f}")
    print(f"  Readout-projected PCA: best raw R = {best_rp:.3f}")
    print(f"  Cointegration (SFA):   best raw R = {best_c_sfa_overall:.3f}  ({best_window})")
    print()

if __name__ == "__main__":
    main()
