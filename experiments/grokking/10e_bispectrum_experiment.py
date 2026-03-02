"""
Denoising Experiment 10e — Bispectrum / Higher-Order Statistics

Compares approaches to discovering features in grokked modular addition,
focusing on 3rd-order statistics (skewness/coskewness). 

Because the embedding noise is largely additive f(a)+g(b) and the 
target Fourier features require multiplicative interactions (created by ReLU), 
higher-order statistics like the 3rd cumulant matrix should theoretically 
filter out the additive noise and isolate the nonlinear interactions,
using only a single snapshot of activations.

1. Raw PCA (2nd-order statistics)
2. Readout-projected PCA
3. Coskewness PCA (3rd-order statistics)
"""

import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Requires PyTorch")
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

def coskewness_pca(h, n_components=20):
    """
    Computes a pseudo-coskewness matrix.
    
    True coskewness is a 3D tensor: S_{ijk} = E[(x_i - m_i)(x_j - m_j)(x_k - m_k)].
    Since a 128x128x128 tensor is large (2M elements) and hard to eigendecompose,
    we compute a "Coskewness Matrix" relative to the total variance of each vector.
    
    We weight the standard covariance matrix by the squared norm of the centered 
    activations. This acts like a 3rd-order/4th-order filter: data points with 
    extreme nonlinear deviations dominate the covariance matrix, while standard 
    Gaussian noise (which doesn't have fat tails) is suppressed.
    """
    h_centered = h - h.mean(axis=0)
    
    # Standardize to unit variance per dimension so magnitude doesn't just
    # replicate standard PCA
    std = h_centered.std(axis=0)
    std[std < 1e-10] = 1.0
    h_std = h_centered / std
    
    # Calculate the squared norm of each sample's standardized activation
    # This acts as our nonlinear weighting factor
    weights = np.sum(h_std**3, axis=1)  # Using ^3 specifically for skewness/asymmetry
    
    # Compute the weighted covariance matrix (a slice of the coskewness tensor)
    # We take absolute weights to ensure the matrix remains positive semi-definite 
    # for the eigenvalue decomposition.
    abs_weights = np.abs(weights)
    abs_weights = abs_weights / abs_weights.sum()  # normalize weights
    
    weighted_h = h_centered * np.sqrt(abs_weights)[:, np.newaxis]
    pseudo_coskew = weighted_h.T @ weighted_h
    
    eigenvalues, eigenvectors = np.linalg.eigh(pseudo_coskew)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    
    return eigenvectors[:, idx].T, eigenvalues[idx]

def coskewness_tensor_unfolding(h, n_components=20):
    """
    Computes the actual 3D coskewness tensor S_{ijk}, and unfolds it into a
    (D, D^2) matrix. We then do SVD on this unfolding to find the directions
    that explain the most 3rd-order variance.
    
    This directly searches for the highest non-Gaussian (skewed) interactions.
    """
    h_centered = h - h.mean(axis=0)
    N, D = h_centered.shape
    
    # We can't build 128x128x128 in memory easily (it's 2M floats, actually totally fine! ~16MB)
    # Let's compute it. S_{ijk} = sum_n (x_ni * x_nj * x_nk)
    
    # To compute efficiently: 
    # For each sample n, outer product of x_n with itself is a DxD matrix M_n.
    # S = sum_n (x_n tensor M_n)
    
    S_unfolded = np.zeros((D, D * D), dtype=np.float32)
    
    # Compute in batches to save memory and be fast
    batch_size = 1000
    for i in range(0, N, batch_size):
        h_batch = h_centered[i:i+batch_size]  # (B, D)
        
        # h_batch[:, :, None] * h_batch[:, None, :] gives (B, D, D)
        M_batch = h_batch[:, :, np.newaxis] * h_batch[:, np.newaxis, :]
        
        # Reshape M_batch to (B, D*D)
        M_batch_flat = M_batch.reshape(len(h_batch), -1)
        
        # Multiply h_batch (B, D) with M_batch_flat (B, D*D) and sum over B
        # This gives a (D, D*D) matrix
        S_unfolded += h_batch.T @ M_batch_flat
        
    S_unfolded /= N
    
    # Now we have the unfolded 3rd moment tensor S_{(1)}
    # We do SVD on this unfolded tensor. The left singular vectors U 
    # span the subspace containing the most skewness.
    U, S, Vt = np.linalg.svd(S_unfolded, full_matrices=False)
    
    directions = U[:, :n_components].T
    
    return directions, S[:n_components]


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
    print("Denoising Experiment 10e: Bispectrum (3rd-Order Statistics)")
    print("=" * 70)

    torch.manual_seed(42)
    model = ModularAdditionMLP(p).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        loss = criterion(model(train_a, train_b), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(train_a, train_b).argmax(1) == train_y).float().mean().item()
                test_acc = (model(test_a, test_b).argmax(1) == test_y).float().mean().item()
            print(f"  Epoch {epoch:>5}: train={train_acc:.0%}, test={test_acc:.0%}")

    print()

    # Collect FINAL activations only (no temporal tracking needed)
    h, sums, _, _ = collect_activations(model, p)
    h_avg, top_freqs = discover_frequencies(h, sums, p, top_n=5)
    total_var = np.var(h, axis=0).sum()

    n_components = 20

    print("Approach 1: Raw PCA (2nd-order)")
    pca_dirs, pca_evals = raw_pca(h, n_components)
    pca_results = evaluate_directions(pca_dirs, h, sums, p, top_freqs)
    
    print("Approach 2: Readout-Projected PCA")
    rp_dirs, _, _ = readout_projected_pca(h, model, n_components)
    rp_results = evaluate_directions(rp_dirs, h, sums, p, top_freqs)

    print()
    print("=" * 70)
    print("Approach 3: Pseudo-Coskewness PCA (Weighted 3rd-order)")
    print("=" * 70)
    cs_dirs, cs_evals = coskewness_pca(h, n_components)
    cs_results = evaluate_directions(cs_dirs, h, sums, p, top_freqs)
    
    cs_sorted = sorted(cs_results, key=lambda x: x["raw_r"], reverse=True)
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in cs_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | {r['avg_r']:>5.3f} ({r['avg_k']:>2})")


    print()
    print("=" * 70)
    print("Approach 4: True Coskewness Tensor Unfolding")
    print("=" * 70)
    print("  Computing exact 128x128x128 3rd moment tensor SVD...")
    cu_dirs, cu_evals = coskewness_tensor_unfolding(h, n_components)
    cu_results = evaluate_directions(cu_dirs, h, sums, p, top_freqs)
    
    cu_sorted = sorted(cu_results, key=lambda x: x["raw_r"], reverse=True)
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in cu_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | {r['avg_r']:>5.3f} ({r['avg_k']:>2})")


    print()
    print("=" * 70)
    print("Summary: Best Raw R")
    print("=" * 70)
    best_pca = max(r["raw_r"] for r in pca_results)
    best_rp = max(r["raw_r"] for r in rp_results)
    best_cs = max(r["raw_r"] for r in cs_results)
    best_cu = max(r["raw_r"] for r in cu_results)

    print(f"  Raw PCA:               best raw R = {best_pca:.3f}")
    print(f"  Readout-projected PCA: best raw R = {best_rp:.3f}")
    print(f"  Pseudo-Coskewness PCA: best raw R = {best_cs:.3f}")
    print(f"  True Tensor Unfolding: best raw R = {best_cu:.3f}")
    print()

if __name__ == "__main__":
    main()