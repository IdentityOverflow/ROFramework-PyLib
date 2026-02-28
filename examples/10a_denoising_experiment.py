"""
Denoising Experiment — Finding Fourier Features Without Task Knowledge

Compares five approaches to discovering features in grokked modular addition:

1. Raw PCA (baseline) — finds max-variance directions → embedding noise
2. Readout-projected PCA — project onto output weight row space, then PCA
3. ICA (FastICA) — find maximally non-Gaussian / independent directions
4. Temporal differencing — PCA on (h_now - h_earlier), finds fast-changing directions
5. Sum-averaged per-neuron (oracle) — requires task knowledge (ground truth)

The question: can approaches 2, 3, or 4 find Fourier features without knowing
anything about modular addition or sum classes?

Requires: PyTorch, scikit-learn
Runtime: ~30s GPU, ~3 min CPU
"""

import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Requires PyTorch: pip install torch")
    sys.exit(1)

try:
    from sklearn.decomposition import FastICA
except ImportError:
    print("Requires scikit-learn: pip install scikit-learn")
    sys.exit(1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model and data (same as examples 08/09)
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
# Collect activations
# ---------------------------------------------------------------------------


def collect_activations(model, p):
    """Get post-ReLU hidden activations for all p^2 pairs."""
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    model.eval()
    with torch.no_grad():
        h = model.relu(model.fc1(torch.cat([
            model.embed_a(grid_a), model.embed_b(grid_b)
        ], -1))).cpu().numpy()

    return h, sums, grid_a.cpu().numpy(), grid_b.cpu().numpy()


# ---------------------------------------------------------------------------
# Ground truth: discover what frequencies the model actually uses
# ---------------------------------------------------------------------------


def discover_frequencies(h, sums, p, top_n=5):
    """DFT on sum-averaged activations to find model's dominant frequencies."""
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
# Approach 1: Raw PCA
# ---------------------------------------------------------------------------


def raw_pca(h, n_components=20):
    """Standard PCA on raw activations."""
    h_centered = h - h.mean(axis=0)
    cov = (h_centered.T @ h_centered) / (len(h) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, idx].T, eigenvalues[idx]  # (K, D), (K,)


# ---------------------------------------------------------------------------
# Approach 2: Readout-projected PCA
# ---------------------------------------------------------------------------


def readout_projected_pca(h, model, n_components=20):
    """Project activations onto fc2 row space, then PCA."""
    W = model.fc2.weight.detach().cpu().numpy()  # (97, 128)

    # SVD of W to get row space basis
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    # Vt rows are the right singular vectors = basis of row space
    # W has rank min(97, 128) = 97, so Vt is (97, 128)
    # Project activations: H_proj = H @ Vt.T @ Vt (project then back)
    # Or just work in the 97-d coordinate system: coords = H @ Vt.T

    coords = (h - h.mean(axis=0)) @ Vt.T  # (N, 97)

    # PCA in the projected space
    cov = (coords.T @ coords) / (len(coords) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]

    # Map directions back to original 128-d space
    directions_in_proj = eigenvectors[:, idx].T  # (K, 97)
    directions_original = directions_in_proj @ Vt  # (K, 128)
    # Normalize
    norms = np.linalg.norm(directions_original, axis=1, keepdims=True)
    directions_original = directions_original / np.clip(norms, 1e-12, None)

    return directions_original, eigenvalues[idx], Vt


# ---------------------------------------------------------------------------
# Approach 3: ICA
# ---------------------------------------------------------------------------


def ica_decomposition(h, n_components=20, max_iter=1000):
    """FastICA to find independent components."""
    ica = FastICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=42,
        whiten="unit-variance",
    )
    sources = ica.fit_transform(h)  # (N, n_components)
    # Mixing matrix columns are the directions in activation space
    mixing = ica.mixing_  # (128, n_components)
    directions = mixing.T  # (n_components, 128)
    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    return directions, sources


# ---------------------------------------------------------------------------
# Approach 4: Temporal differencing
# ---------------------------------------------------------------------------


def temporal_diff_pca(h_now, h_earlier, n_components=20):
    """PCA on the difference between two activation snapshots.

    Finds directions of maximum *change* rather than maximum variance.
    During grokking, changing directions should be the task-relevant features.
    """
    diff = h_now - h_earlier  # (N, D)
    diff_centered = diff - diff.mean(axis=0)
    cov = (diff_centered.T @ diff_centered) / (len(diff) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, idx].T, eigenvalues[idx]


def dual_window_pca(h_now, h_narrow, h_wide, n_components=20):
    """Dual-window temporal differencing.

    Computes change-covariance for both a narrow and wide window, then finds
    directions where the narrow window has disproportionately high change
    relative to the wide window.  This isolates features that are *currently*
    active rather than historically changed.

    Method: compute narrow_cov and wide_cov, then find the generalized
    eigenvectors of (narrow_cov, wide_cov) — directions that maximize
    the ratio narrow_variance / wide_variance.
    """
    diff_narrow = h_now - h_narrow
    diff_wide = h_now - h_wide

    dn = diff_narrow - diff_narrow.mean(axis=0)
    dw = diff_wide - diff_wide.mean(axis=0)

    cov_narrow = (dn.T @ dn) / (len(dn) - 1)
    cov_wide = (dw.T @ dw) / (len(dw) - 1)

    # Regularize wide cov to avoid singularity
    cov_wide += np.eye(cov_wide.shape[0]) * 1e-6

    # Generalized eigenvalue problem: narrow_cov @ v = lambda * wide_cov @ v
    # Directions with highest lambda changed most in the narrow window
    # relative to the wide window — i.e., recently accelerating features
    from scipy.linalg import eigh as scipy_eigh
    eigenvalues, eigenvectors = scipy_eigh(cov_narrow, cov_wide)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    directions = eigenvectors[:, idx].T
    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    return directions, eigenvalues[idx]


# ---------------------------------------------------------------------------
# Evaluate: how well does each direction correlate with Fourier features?
# ---------------------------------------------------------------------------


def evaluate_directions(directions, h, sums, p, top_freqs):
    """For each direction, project activations and check correlation with
    Fourier features on both raw and sum-averaged data."""
    s_vals = np.arange(p)
    results = []

    for i, d in enumerate(directions):
        proj = h @ d  # (N,) projection of all activations onto this direction

        # Sum-average the projections
        proj_avg = np.zeros(p)
        for s in range(p):
            proj_avg[s] = proj[sums == s].mean()

        # Best Fourier correlation (raw)
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

        # Best Fourier correlation (sum-averaged projection)
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
    print("Denoising Experiment: Can We Find Fourier Features Without Task Knowledge?")
    print("=" * 70)
    print(f"  p={p}, epochs={num_epochs}, device={DEVICE}")
    print()

    # --- Train model, saving activation snapshots for temporal differencing ---
    print("Training model to grokking...")
    torch.manual_seed(42)
    model = ModularAdditionMLP(p).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0
    )
    criterion = nn.CrossEntropyLoss()

    # Save activation snapshots at key epochs for temporal differencing
    snapshot_epochs = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    snapshots = {}  # epoch -> activations array
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

    # --- Collect activations ---
    h, sums, _, _ = collect_activations(model, p)
    print(f"Activations: {h.shape} (p²={p*p} pairs, {h.shape[1]} hidden dims)")

    # --- Ground truth frequencies ---
    h_avg, top_freqs = discover_frequencies(h, sums, p, top_n=5)
    print(f"Top frequencies (from sum-averaged DFT): {top_freqs}")
    print()

    # --- Variance decomposition ---
    total_var = np.var(h, axis=0).sum()
    between_var = np.var(h_avg, axis=0).sum()
    within_frac = 1.0 - between_var / total_var
    print(f"Variance decomposition: {within_frac:.0%} within-sum-class, "
          f"{1-within_frac:.0%} between-sum-class")
    print()

    n_components = 20

    # === Approach 1: Raw PCA ===
    print("=" * 70)
    print("Approach 1: Raw PCA")
    print("=" * 70)
    pca_dirs, pca_evals = raw_pca(h, n_components)
    pca_results = evaluate_directions(pca_dirs, h, sums, p, top_freqs)

    total_eig = sum(pca_evals)
    print(f"  Top-{n_components} directions explain "
          f"{sum(pca_evals)/total_var:.0%} of total variance")
    print()
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in pca_results[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | "
              f"{r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    # === Approach 2: Readout-projected PCA ===
    print()
    print("=" * 70)
    print("Approach 2: Readout-Projected PCA")
    print("=" * 70)
    rp_dirs, rp_evals, Vt = readout_projected_pca(h, model, n_components)
    rp_results = evaluate_directions(rp_dirs, h, sums, p, top_freqs)

    # How much variance is in the readout subspace vs orthogonal?
    h_centered = h - h.mean(axis=0)
    h_in_readout = h_centered @ Vt.T @ Vt
    h_orthogonal = h_centered - h_in_readout
    var_in_readout = np.var(h_in_readout, axis=0).sum()
    var_orthogonal = np.var(h_orthogonal, axis=0).sum()
    print(f"  Readout subspace: rank {Vt.shape[0]}, "
          f"captures {var_in_readout/total_var:.0%} of variance")
    print(f"  Orthogonal complement: {var_orthogonal/total_var:.0%} of variance")
    print()
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in rp_results[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | "
              f"{r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    # === Approach 3: ICA ===
    print()
    print("=" * 70)
    print("Approach 3: ICA (FastICA)")
    print("=" * 70)
    ica_dirs, ica_sources = ica_decomposition(h, n_components)
    ica_results = evaluate_directions(ica_dirs, h, sums, p, top_freqs)

    # Sort by raw R descending for readability
    ica_results_sorted = sorted(ica_results, key=lambda r: r["raw_r"], reverse=True)
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in ica_results_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | "
              f"{r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    # === Approach 4: Temporal differencing ===
    print()
    print("=" * 70)
    print("Approach 4: Temporal Differencing")
    print("=" * 70)

    # Try multiple offset windows to find which works best
    # Compare final activations (h) to earlier snapshots
    td_best_overall = 0.0
    td_best_results = None
    td_best_window = ""

    for earlier_epoch in sorted(snapshots.keys()):
        h_earlier = snapshots[earlier_epoch]
        td_dirs, _ = temporal_diff_pca(h, h_earlier, n_components)
        td_results_i = evaluate_directions(td_dirs, h, sums, p, top_freqs)
        best_r = max(r["raw_r"] for r in td_results_i)
        best_avg = max(r["avg_r"] for r in td_results_i)
        window = f"epoch {earlier_epoch} → {num_epochs}"
        print(f"  {window:>25s}: best raw R = {best_r:.3f}, best avg R = {best_avg:.3f}")

        if best_r > td_best_overall:
            td_best_overall = best_r
            td_best_results = td_results_i
            td_best_window = window

    print()
    print(f"  Best window: {td_best_window}")
    print()
    td_results_sorted = sorted(td_best_results, key=lambda r: r["raw_r"], reverse=True)
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in td_results_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | "
              f"{r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    # === Approach 4b: Dual-window temporal differencing ===
    print()
    print("=" * 70)
    print("Approach 4b: Dual-Window Temporal Differencing")
    print("=" * 70)
    print("  Generalized eigenvalue: finds directions that changed MORE in the")
    print("  narrow window than the wide window (recently accelerating features)")
    print()

    # Try several narrow/wide combinations
    dw_best_overall = 0.0
    dw_best_results = None
    dw_best_label = ""

    wide_epochs = [500, 1000, 1500]
    narrow_epochs = [3000, 4000, 5000]

    for wide_ep in wide_epochs:
        for narrow_ep in narrow_epochs:
            if wide_ep >= narrow_ep:
                continue
            if wide_ep not in snapshots or narrow_ep not in snapshots:
                continue
            dw_dirs, _ = dual_window_pca(h, snapshots[narrow_ep], snapshots[wide_ep], n_components)
            dw_results_i = evaluate_directions(dw_dirs, h, sums, p, top_freqs)
            best_r = max(r["raw_r"] for r in dw_results_i)
            best_avg = max(r["avg_r"] for r in dw_results_i)
            label = f"wide={wide_ep}, narrow={narrow_ep}"
            print(f"  {label:>30s}: best raw R = {best_r:.3f}, best avg R = {best_avg:.3f}")

            if best_r > dw_best_overall:
                dw_best_overall = best_r
                dw_best_results = dw_results_i
                dw_best_label = label

    print()
    print(f"  Best combo: {dw_best_label}")
    print()
    dw_results_sorted = sorted(dw_best_results, key=lambda r: r["raw_r"], reverse=True)
    print(f"  {'Dir':>4} | {'Var%':>6} | {'Raw R':>6} (k) | {'Avg R':>6} (k)")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for r in dw_results_sorted[:10]:
        var_pct = r["proj_var"] / total_var * 100
        print(f"  {r['idx']:>4} | {var_pct:>5.1f}% | {r['raw_r']:>5.3f} ({r['raw_k']:>2}) | "
              f"{r['avg_r']:>5.3f} ({r['avg_k']:>2})")

    # === Approach 5: Oracle (sum-averaged per-neuron) ===
    print()
    print("=" * 70)
    print("Approach 5: Oracle — Sum-Averaged Per-Neuron (requires task knowledge)")
    print("=" * 70)
    s_vals = np.arange(p)
    for k in top_freqs:
        sin_t = np.sin(2 * np.pi * k * s_vals / p)
        cos_t = np.cos(2 * np.pi * k * s_vals / p)
        best_r_sin = max(abs(np.corrcoef(h_avg[:, j], sin_t)[0, 1])
                         for j in range(h_avg.shape[1]))
        best_r_cos = max(abs(np.corrcoef(h_avg[:, j], cos_t)[0, 1])
                         for j in range(h_avg.shape[1]))
        print(f"  k={k:>2}: R(sin)={best_r_sin:.3f}, R(cos)={best_r_cos:.3f}")

    # === Summary ===
    print()
    print("=" * 70)
    print("Summary: Best Raw R across all directions (no sum-averaging)")
    print("=" * 70)

    best_pca = max(r["raw_r"] for r in pca_results)
    best_rp = max(r["raw_r"] for r in rp_results)
    best_ica = max(r["raw_r"] for r in ica_results)
    best_td = td_best_overall
    best_dw = dw_best_overall
    print(f"  Raw PCA:               best raw R = {best_pca:.3f}")
    print(f"  Readout-projected PCA: best raw R = {best_rp:.3f}")
    print(f"  ICA:                   best raw R = {best_ica:.3f}")
    print(f"  Temporal diff (single): best raw R = {best_td:.3f}  ({td_best_window})")
    print(f"  Temporal diff (dual):  best raw R = {best_dw:.3f}  ({dw_best_label})")
    print()

    # Also check: what if we sum-average the PROJECTED signal?
    print("Summary: Best Avg R (sum-averaged projections)")
    print("-" * 70)
    best_pca_avg = max(r["avg_r"] for r in pca_results)
    best_rp_avg = max(r["avg_r"] for r in rp_results)
    best_ica_avg = max(r["avg_r"] for r in ica_results)
    best_td_avg = max(r["avg_r"] for r in td_best_results)
    best_dw_avg = max(r["avg_r"] for r in dw_best_results)
    print(f"  Raw PCA:               best avg R = {best_pca_avg:.3f}")
    print(f"  Readout-projected PCA: best avg R = {best_rp_avg:.3f}")
    print(f"  ICA:                   best avg R = {best_ica_avg:.3f}")
    print(f"  Temporal diff (single): best avg R = {best_td_avg:.3f}")
    print(f"  Temporal diff (dual):  best avg R = {best_dw_avg:.3f}")
    print(f"  Oracle (per-neuron):   R = ~0.97")
    print()

    # Key question: which approaches found Fourier features on RAW data?
    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    approaches = [("Readout projection", best_rp),
                  ("ICA", best_ica),
                  ("Temporal diff (single)", best_td),
                  ("Temporal diff (dual)", best_dw)]
    for name, best_r in approaches:
        if best_r > best_pca + 0.05:
            print(f"  {name}: IMPROVED over raw PCA (R={best_r:.3f} vs {best_pca:.3f})")
        else:
            print(f"  {name}: no improvement over raw PCA")

    winners = [(n, r) for n, r in approaches if r > 0.7]
    if winners:
        print(f"  >>> 'Strong' features (R > 0.7) found by: "
              f"{', '.join(f'{n} ({r:.3f})' for n, r in winners)}")
    else:
        print("  >>> No approach reached 'strong' (R > 0.7) on raw data")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
