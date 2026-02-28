"""
Denoising Experiment 10c — Full-Rank DMD and Projected DMD

Compares:
1. Raw PCA (baseline)
2. Readout-projected PCA
3. Full-Rank DMD (no variance bottleneck)
4. Readout-Projected DMD
"""

import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Requires PyTorch: pip install torch")
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

def full_rank_dmd(snapshots_dict):
    epochs = sorted(snapshots_dict.keys())
    X1_list, X2_list = [], []
    for i in range(len(epochs) - 1):
        X1_list.append(snapshots_dict[epochs[i]].T) 
        X2_list.append(snapshots_dict[epochs[i+1]].T)
        
    X1 = np.hstack(X1_list)  
    X2 = np.hstack(X2_list)  
    
    # SVD without truncation (keep all 128 dimensions)
    # Using small threshold for S to prevent division by zero for numerical stability
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    
    r = np.sum(S > 1e-10)
    Ur = U[:, :r]
    Sr = S[:r]
    Vtr = Vt[:r, :]
    
    # Compute the full temporal operator
    Atilde = Ur.T @ X2 @ Vtr.T @ np.diag(1.0 / Sr)
    
    eigenvalues, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vtr.T @ np.diag(1.0 / Sr) @ W 
    
    directions = np.real(Phi).T  
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    
    growth_rates = np.abs(eigenvalues)
    idx = np.argsort(growth_rates)[::-1]
    
    return directions[idx], growth_rates[idx], eigenvalues[idx]

def projected_dmd(snapshots_dict, model):
    W = model.fc2.weight.detach().cpu().numpy()
    U_readout, S_readout, Vt = np.linalg.svd(W, full_matrices=False)

    epochs = sorted(snapshots_dict.keys())
    X1_list, X2_list = [], []
    for i in range(len(epochs) - 1):
        # Project each snapshot onto the readout space before tracking dynamics
        proj_h1 = (snapshots_dict[epochs[i]] - snapshots_dict[epochs[i]].mean(axis=0)) @ Vt.T
        proj_h2 = (snapshots_dict[epochs[i+1]] - snapshots_dict[epochs[i+1]].mean(axis=0)) @ Vt.T
        X1_list.append(proj_h1.T) 
        X2_list.append(proj_h2.T)
        
    X1 = np.hstack(X1_list)  
    X2 = np.hstack(X2_list)  
    
    # DMD on the projected space
    U, S, V_dmd = np.linalg.svd(X1, full_matrices=False)
    
    r = np.sum(S > 1e-10)
    Ur = U[:, :r]
    Sr = S[:r]
    Vtr = V_dmd[:r, :]
    
    Atilde = Ur.T @ X2 @ Vtr.T @ np.diag(1.0 / Sr)
    eigenvalues, W_dmd = np.linalg.eig(Atilde)
    Phi = X2 @ Vtr.T @ np.diag(1.0 / Sr) @ W_dmd 
    
    directions_in_proj = np.real(Phi).T  
    directions_original = directions_in_proj @ Vt
    norms = np.linalg.norm(directions_original, axis=1, keepdims=True)
    directions_original = directions_original / np.clip(norms, 1e-12, None)
    
    growth_rates = np.abs(eigenvalues)
    idx = np.argsort(growth_rates)[::-1]
    
    return directions_original[idx], growth_rates[idx], eigenvalues[idx]


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
    print("Denoising Experiment 10c: Full-Rank and Projected DMD")
    print("=" * 70)

    torch.manual_seed(42)
    model = ModularAdditionMLP(p).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0
    )
    criterion = nn.CrossEntropyLoss()

    snapshot_epochs = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
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
    print("Approach 3: Full-Rank DMD")
    print("=" * 70)
    fr_dmd_dirs, fr_dmd_growth, _ = full_rank_dmd(snapshots)
    fr_dmd_results = evaluate_directions(fr_dmd_dirs, h, sums, p, top_freqs)
    
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Growth':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    
    # Sort results by raw_r to find if there are ANY good directions hidden in the 128
    fr_sorted = sorted(fr_dmd_results, key=lambda x: x["raw_r"], reverse=True)
    for r in fr_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        growth = fr_dmd_growth[r['idx']]
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {growth:>6.3f} | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | {r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    print()
    print("=" * 70)
    print("Approach 4: Readout-Projected DMD")
    print("=" * 70)
    p_dmd_dirs, p_dmd_growth, _ = projected_dmd(snapshots, model)
    p_dmd_results = evaluate_directions(p_dmd_dirs, h, sums, p, top_freqs)
    
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Growth':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    
    p_sorted = sorted(p_dmd_results, key=lambda x: x["raw_r"], reverse=True)
    for r in p_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        growth = p_dmd_growth[r['idx']]
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {growth:>6.3f} | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | {r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    print()
    print("=" * 70)
    print("Summary: Best Raw R")
    print("=" * 70)
    best_pca = max(r["raw_r"] for r in pca_results)
    best_rp = max(r["raw_r"] for r in rp_results)
    best_fr_dmd = max(r["raw_r"] for r in fr_dmd_results)
    best_p_dmd = max(r["raw_r"] for r in p_dmd_results)

    print(f"  Raw PCA:               best raw R = {best_pca:.3f}")
    print(f"  Readout-projected PCA: best raw R = {best_rp:.3f}")
    print(f"  Full-Rank DMD:         best raw R = {best_fr_dmd:.3f}")
    print(f"  Readout-Projected DMD: best raw R = {best_p_dmd:.3f}")
    print()

if __name__ == "__main__":
    main()
