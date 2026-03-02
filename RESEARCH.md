# Research Findings

Empirical results from using the Recursive Observer Framework on training-time knowledge dynamics, motivated by [He et al. 2025](https://arxiv.org/abs/2306.12034) ("On the Mechanism and Dynamics of Modular Addition").

## Knowledge Trajectory Tracking (Phase 8a)

`KnowledgeTracker` records K(d_ext) = (ρ, ε, σ, C) over training epochs, enabling temporal analysis of how knowledge forms. Applied to modular addition (a+b mod 97) with grokking.

### Setup

An MLP learns modular addition: given inputs a and b (integers mod 97), predict (a+b) mod 97. The architecture is `Embedding(a) + Embedding(b) → Linear(256→128) → ReLU → Linear(128→97) → logits`. Training uses AdamW with strong weight decay (1.0), which drives grokking — the model memorizes first, then slowly generalizes.

After grokking, the theory predicts that hidden neurons lock into Fourier features: cos(2πk·s/p + φ) where s = (a+b) mod p. We use the framework's knowledge assessment to track whether and when this happens.

### Trajectory tracking results

**Feature-level knowledge precedes behavioral generalization.** Fourier features reach "strong" knowledge (R > 0.7) hundreds of epochs before test accuracy rises. At epoch 500, sin(k=1) features have R=0.765 while test accuracy is 0%. The observer "knows" the feature before the model can use it for classification.

**Neurons lock into clean cosine waves.** After grokking, individual neurons are 98.8% pure sinusoids cos(2πk·s/p + φ), confirming the resonance selection theory from He et al.

**Models select their own frequencies.** The model uses k=7 (10% of power), k=22 (9%), k=9 (7%) — not k=1,2,3 as might be expected. Which frequencies win is determined by initial weight alignment. Auto-discovery via DFT of sum-class-averaged activations reveals the winners.

**Phase detection works.** `detect_grokking()` correctly identifies the epoch where knowledge transitions from weak to strong. `detect_resonance()` finds the pre-grokking period where correlation rises but noise is still high.

### The sum-averaging requirement

Raw per-pair activations have 89% within-sum-class variance (pair-specific embedding noise) and only 11% between-sum-class variance (the Fourier signal). This means:

- Per-neuron correlation on raw (a,b) pairs: R = 0.32
- Linear probe on raw pairs: R = 0.66 (for rare frequencies like k=1)
- Per-neuron correlation on sum-class-averaged data: R = 0.97

**Why**: Pre-ReLU activations are f(a) + g(b) — additive. The target sin(k(a+b)/p) requires multiplicative interaction. These are mathematically orthogonal over all p² pairs (Fourier completeness). ReLU creates the needed nonlinearity, but the resulting signal is swamped by pair-specific embedding noise. Averaging over all pairs with the same sum s removes this noise.

**Implication**: Sum-averaging is task-specific — it requires knowing the relevant grouping variable ((a+b) mod p). For real models, you don't know what to average over. This is where feature discovery (Phase 8b) and SAEs become necessary.

See [examples/08_knowledge_tracker.py](examples/08_knowledge_tracker.py).

## Online Feature Discovery (Phase 8b — experimental)

`ActivationTracker` uses Welford's online algorithm for streaming covariance estimation and PCA to find emerging directions in activation space during training. The goal: discover features without task-specific knowledge.

### Setup

Same modular addition MLP. At evaluation intervals, collect all p² activations, compute running covariance, extract top PCA directions. Track which directions persist across epochs (stability) and how eigenvalues evolve.

### Feature discovery results

**Stability is a task-agnostic grokking detector.** PCA direction stability (cosine similarity of top eigenvectors across epochs) jumps from 0.4 to 0.999 during the memorization-to-generalization transition. No task knowledge needed — just watching whether the principal directions are settling.

**Eigenvalue spikes mark memorization onset.** Sharp eigenvalue increases at ~epoch 250 coincide with the model starting to memorize training data.

**Top-variance directions are NOT task-relevant.** Readout alignment (how much a PCA direction projects onto the output weight matrix) drops from 0.5 to 0.04 during grokking. PCA finds embedding modes (89% of variance), not Fourier features (11% of variance). The directions that explain the most variance are not the directions that matter for the task.

### What works and what doesn't

| Signal | Task-agnostic? | Detects grokking? | Finds features? |
|--------|---------------|-------------------|-----------------|
| PCA stability | Yes | Yes | No |
| Eigenvalue trajectory | Yes | Partially (memorization) | No |
| Readout alignment | Yes | Yes (inverse signal) | No |
| Sum-class averaging | No (task-specific) | Yes | Yes |
| Per-neuron DFT | No (assumes Fourier) | Yes | Yes |

**Conclusion**: Temporal dynamics (stability, eigenvalue changes) are reliable task-agnostic detectors of phase transitions. But finding *what* the model learned — the actual features — requires either task-specific denoising or learned decomposition (SAEs).

See [examples/09_activation_tracker.py](examples/09_activation_tracker.py).

## The Denoising Problem

The central challenge for unsupervised feature discovery in neural networks:

In modular addition, 89% of hidden activation variance is noise relative to the task-relevant Fourier features. Any method that optimizes for total variance (PCA, autoencoder reconstruction) will find the noise. Methods that find the signal require either:

1. **Task knowledge** — knowing what to average over (sum-class averaging), or what functional form to look for (DFT)
2. **Sparsity constraints** — assuming features activate sparsely, so a sparse decomposition separates signal from noise (SAEs)
3. **Structured scanning** — systematically probing different "filter orientations" to discover which ones reveal coherent structure (an open direction)

Option 3 is the most interesting theoretically — it connects to matched filtering, canonical correlation analysis, and the question of whether there's a general method between "know nothing" (PCA) and "know everything" (task-specific probes).

### Analogy

Imagine a crowd where several small groups are singing different songs simultaneously. You hear a wall of noise. PCA finds the loudest sounds — which are the crowd murmur, not any song. Sum-averaging is like already knowing which people are in each group and listening to them separately. Readout projection is like asking the conductor "which instruments are you listening to?" and filtering for just those — you don't need to know the songs, the conductor already knows which sounds matter.

## Denoising Experiment

Given the 89/11 variance split, we tested several task-agnostic approaches to find Fourier features without sum-averaging.

### Approaches tested

1. **Raw PCA** (baseline) — standard PCA on all 9409 activation vectors
2. **Readout-projected PCA** — project activations onto fc2's row space (the subspace the output layer reads from), then PCA within that constrained space
3. **ICA (FastICA)** — find maximally independent/non-Gaussian directions
4. **Temporal differencing (single-window)** — PCA on (h_now - h_earlier) to find directions that changed most between two training snapshots
5. **Temporal differencing (dual-window)** — generalized eigenvalue problem (scipy.linalg.eigh) to find directions that changed more in a narrow window than a wide window, isolating recently accelerating features
6. **Dynamic Mode Decomposition (DMD)** — treats training epochs as a temporal DoF in a dynamical system. Tested in both full-rank and readout-projected variants.
7. **Cointegration / Slow Feature Analysis (SFA)** — computes generalized eigenvalues of Cov(Final) vs Cov(Change) to find features that have high variance but have stopped changing.
8. **3rd-order tensor unfolding** — SVD of the mode-1 unfolded coskewness tensor $S_{ijk} = E[x_i x_j x_k]$ to find directions maximizing 3rd-order variance
9. **Sum-averaged per-neuron** (oracle) — requires task knowledge

### Denoising results

| Method | Best Raw R | Best Avg R | Task knowledge needed? |
| ------ | ---------- | ---------- | ---------------------- |
| Readout-projected PCA | **0.907** | **0.994** | No |
| 3rd-order tensor unfolding | 0.850 | 0.892 | No |
| Cointegration / SFA | 0.806 | 0.972 | No |
| Temporal diff (single) | 0.521 | 0.892 | No |
| Temporal diff (dual) | 0.521 | 0.892 | No |
| Readout-Projected DMD | 0.410 | 0.833 | No |
| Full-Rank DMD | 0.316 | 0.785 | No |
| Raw PCA | 0.070 | 0.881 | No |
| ICA | 0.055 | 0.686 | No |
| Oracle (sum-averaged) | — | 0.97 | Yes |

### Analysis

**Cointegration / Slow Feature Analysis finds settled features.** The generalized eigenvalue problem $C_{final} \cdot v = \lambda \cdot C_{\Delta} \cdot v$ finds directions with high final variance but low recent change — features that have "crystallized." This discovered the k=7 Fourier feature with R=0.806 raw correlation, without task knowledge or readout projection. The insight: after grokking, Fourier features stabilize while embedding noise keeps drifting, so the variance-to-change ratio naturally separates them. Caveats: this is a post-hoc detector (features must have already settled), and the window choice (which "earlier" epoch to compare against) matters without a principled way to select it. It would also find any stable noise direction, not just task-relevant features.

**3rd-order tensor unfolding finds features from a single snapshot.** SVD of the mode-1 unfolded coskewness tensor achieves R=0.850 using only final activations — no temporal snapshots, no architectural information. The 3rd moment captures coordinated skewness patterns across neurons: after ReLU, neurons tuned to the same Fourier frequency have correlated higher-order statistics, while pair-specific embedding noise creates less coordinated skewness. Like readout projection and SFA, the signal hides in low-variance directions (1.1% of total variance). However, **this method does not scale**: the unfolded tensor is (D, D²), requiring O(D³) memory and O(D⁴) compute. For D=128 this is trivial (16 MB), but for D=4096 (Llama-scale) it requires ~537 GB. Approximations (randomized SVD, tensor sketching) exist but tend to discard exactly the low-variance signal we need. This result confirms that 3rd-order statistics separate signal from noise in principle, but SFA (O(D²) memory, O(D³) compute) remains the practical choice for scalable feature discovery without architectural information.

**DMD shows modest improvement over PCA on raw data.** Dynamic Mode Decomposition treats training as a dynamical system, fitting a linear operator to the epoch-to-epoch evolution of activations. Standard DMD's SVD truncation discards the 11% feature signal along with the 89% noise. Full-rank DMD (R=0.316) and readout-projected DMD (R=0.410) improve over raw PCA's R=0.070 on raw data. However, on sum-averaged data, DMD's avg R (0.785–0.833) is actually *worse* than raw PCA (0.881), suggesting DMD directions partially overlap with the signal but don't cleanly isolate it. The fundamental limitation: DMD assumes linear dynamics, but neural network training is highly nonlinear.

**Readout projection works.** By projecting onto the output weight matrix's row space (rank 97, captures 49% of total variance), the embedding noise orthogonal to the output is discarded. PCA within this constrained subspace finds a direction at rank 9 (only 1.3% of total variance) with raw R = 0.888 for k=7. Sum-averaged, it reaches R = 0.994 — better than the oracle per-neuron approach.

**Temporal differencing partially works.** PCA on activation differences (h_epoch5000 - h_epoch3000) finds directions that changed most during grokking. Best raw R = 0.521 — much better than raw PCA (0.070) but far below readout projection (0.907). Later windows (closer to grokking) work better. The method finds different frequencies than readout projection (k=12,42 vs k=7), suggesting it captures different aspects of the learning dynamics. The dual-window variant (generalized eigenvalue problem for narrow-vs-wide change ratio) ties with single-window — the noise is present equally in both windows, so the ratio doesn't improve separation.

**ICA fails.** The non-Gaussianity criterion doesn't separate Fourier features from embedding noise. The embedding noise comprises hundreds of pair-specific effects that dominate the independence structure. FastICA finds noise modes, not signal.

**Why temporal differencing is limited.** The 89% within-sum-class noise is not static — it evolves across epochs as embeddings change. Taking diffs cancels only the *static* component of noise, not the evolving part. The Fourier signal also evolves, so it partially survives differencing, but the signal-to-noise ratio improves only modestly (from 11/89 to roughly 30/70).

**Key insight: the model is its own best radio tuner.** The output weight matrix defines exactly which directions are task-relevant. This is architectural information, not task labels — the model tells us what it cares about through its own weights. Low-variance directions within the readout subspace are precisely the features that generalization selected for, rather than the high-variance noise PCA naturally gravitates toward.

**Implications for real models.** Every neural network with a readout layer has this structure. In transformers, the unembedding matrix plays this role. Projecting intermediate activations onto the unembedding row space before analysis should similarly filter noise, though transformers also use residual streams and attention, complicating the picture.

See [examples/10a_denoising_experiment.py](examples/10a_denoising_experiment.py) (readout projection, ICA, temporal diff), [examples/10b_dmd_experiment.py](examples/10b_dmd_experiment.py) (DMD), [examples/10c_dmd_full_experiment.py](examples/10c_dmd_full_experiment.py) (full-rank and projected DMD), [examples/10d_cointegration_experiment.py](examples/10d_cointegration_experiment.py) (SFA/cointegration), [examples/10e_bispectrum_experiment.py](examples/10e_bispectrum_experiment.py) (3rd-order tensor unfolding).

## Knowledge-Guided Training (Phase 8c — negative result)

Hypothesis: if K(d_ext) can detect memorization (high ρ, low C), we can selectively increase weight decay on memorized features to accelerate grokking.

### Setup

`KnowledgeRegularizer` reads K from a `KnowledgeTracker` at eval intervals and adjusts the optimizer's global weight decay:
- Memorized features (high ρ, low C): multiply weight decay by 3.0
- Generalized features (high ρ, high C): multiply weight decay by 0.5
- Uncertain features (low ρ): leave weight decay unchanged

Two identical runs (same seed, same initial weights): baseline (constant wd=1.0) vs K-guided (KnowledgeRegularizer adjusts wd).

### Result

**K-guided training made grokking 81% slower** (epoch 7250 vs baseline epoch 4000).

### Why it failed

**Feature-behavioral lag undermines the approach.** On sum-averaged data, K(d_ext) measures feature-level knowledge — whether hidden neurons track Fourier components. This reaches "strong" (ρ > 0.7, C > 0.7) hundreds of epochs before test accuracy rises. At epoch 500, features are already "strong" while test accuracy is 0%.

The regularizer interprets "strong" features as generalization and reduces weight decay. But the model hasn't actually generalized — the output layer hasn't learned to compose the features yet. Reducing weight decay removes the regularization pressure that drives grokking, slowing it down.

The "memorized" state (high ρ, low C) that the regularizer targets never occurs on clean sum-averaged data. Features jump directly from "weak" to "strong" because sum-averaging removes the noise that would lower calibration.

### Raw-data variant

A follow-up experiment tracked K on raw per-pair activations instead of sum-averaged data, hoping that the lower per-pair correlations (ρ ≈ 0.36 max) would keep features in the "memorized" classification longer. Result: **no effect** — raw ρ never exceeded the memorized threshold, so the regularizer never fired. Both runs grokked at the same epoch (4000).

This confirms that the problem is not just sum-averaging: per-DoF K(d_ext) is fundamentally the wrong signal for steering grokking dynamics, regardless of the data preprocessing.

### Conclusions

K(d_ext) is a feature-level metric, not a model-level one. Using it to steer training confuses the two levels. The feature-behavioral lag (features form hundreds of epochs before the model generalizes) means K cannot distinguish "features are forming" from "the model has generalized."

Global weight decay modulation is also too blunt — grokking depends on the competition between all features simultaneously, and changing the pressure mid-race disrupts the resonance selection dynamics rather than steering them.

See [examples/11_knowledge_guided_training.py](examples/11_knowledge_guided_training.py) (sum-averaged), [examples/11b_knowledge_guided_raw.py](examples/11b_knowledge_guided_raw.py) (raw data).

## SAE Knowledge Assessment on GPT-2 (Phase 9)

`SAEObserver` bridges the framework from toy models to real ones by integrating pre-trained Sparse Autoencoders (SAELens) with language models (TransformerLens). External DoFs are user-provided labels (sentiment, is_code). Internal DoFs are SAE feature activations. Knowledge assessment answers: which SAE features track each label, and with what correlation, bias, noise, and calibration?

### Setup

GPT-2 small wrapped with pre-trained SAEs from the `gpt2-small-res-jb` release (24,576 features per layer). Dataset: 420 labeled texts across 7 categories (positive sentiment, negative sentiment, code, questions, statements, formal, casual), with 5 label DoFs: is_code, is_question, formality, sentiment, random_label. SAE features extracted at layers 0, 4, 8, 11 with mean-pooling across tokens.

Multi-feature assessment (max_features=10) uses OLS multiple regression with the top-k most correlated SAE features jointly, so ρ reflects the observer's *combined* knowledge — not just one feature's tracking.

### Results (420 texts, multi-feature)

| Label | Layer 8 ρ | ε | σ | C | Type |
|-------|-----------|------|------|------|------|
| is_code | 0.949 | 0.287 | 0.315 | 0.666 | **strong** |
| formality | 0.735 | 0.106 | 0.678 | 0.615 | **strong** |
| is_question | 0.515 | 0.586 | 0.857 | 0.285 | weak |
| sentiment | 0.291 | 0.016 | 0.957 | 0.000 | weak |
| random_label | 0.245 | 0.045 | 0.969 | 0.457 | weak |

**Code detection shows strong knowledge at all layers.** ρ = 0.844 at layer 0, rising to 0.949 at layer 8. However, at layer 0 the knowledge type is "false" (ρ=0.844, ε=0.315) — the early representations correlate with code but with systematic bias. By layer 4 the bias is resolved and knowledge becomes "strong." Code detection is biased toward Python-like syntax: SQL and bash code weakens detection because SAE features are trained on internet text which is Python-heavy.

**Formality is strong at deeper layers.** Formal vs. casual register crosses the "strong" threshold at layer 8 (ρ=0.735, C=0.615). At layer 0 it's only ρ=0.480 — the model builds formality representations progressively through its layers.

**Question detection is weak despite clear signal.** ρ=0.515 with ε=0.586 — the model partially tracks questions but with strongly heteroscedastic errors. Best result is at layer 4 (ρ=0.672). The high ε suggests the multi-feature regression finds features that correlate with questions but whose errors vary systematically with the label. Mean-pooling likely loses the positional signal (question marks at sentence end).

**Sentiment remains weak.** ρ=0.291 across layers — GPT-2 small's residual stream SAE features don't strongly encode sentiment polarity. The signal may be more distributed than 10 features can capture, or it may require attention-head SAEs rather than residual-stream ones.

**Random label is correctly low.** ρ=0.245 with no meaningful signal — the regression finds only spurious correlations that don't survive the error metrics.

### K tuple error decomposition

The original `compute_knowledge()` used z-score normalization, which made ε always ≈ 0 and C a function of ρ alone — "false" and "uncertain" knowledge types were unreachable. The fix uses OLS regression residuals:
- **ε** = |Spearman(external, |residuals|)| — measures heteroscedasticity (do errors depend on the label value?)
- **C** = 1 - CV(binned |residuals|) — measures error uniformity (are errors consistent across bins?)

This produces meaningful four-way classification: strong (high ρ, low ε, high C), false (high ρ, high ε), uncertain (low ρ, high C), weak (everything else).

### Multi-feature assessment

Single-feature K assessment finds only the one internal DoF most correlated with the external label. For distributed representations (common in real models), this severely underestimates knowledge:

| Label | Single ρ | Multi ρ (k=10) | Improvement |
|-------|----------|----------------|-------------|
| is_code | 0.719 | 0.949 | +32% |
| formality | 0.546 | 0.735 | +35% |
| is_question | 0.397 | 0.515 | +30% |
| sentiment | 0.178 | 0.291 | +63% |

Multi-feature assessment uses OLS multiple regression: preselect top-k features by |Pearson ρ|, build design matrix X (n × k), solve β = (X'X)⁻¹X'y, compute ρ = |pearsonr(y, Xβ)|. k is capped at n_samples // 10 to prevent overfitting.

### Implications

The framework's abstractions (DoFs, observers, knowledge assessment) transfer directly to real models via SAE decomposition. The key bridge: SAE features are atomic DoFs in the framework's sense — they have magnitude, can be tracked over time, and their correlation with external properties can be assessed.

The graded K tuple provides interpretability beyond linear probes: "false" knowledge at layer 0 for code (high ρ but biased) is a result a probe accuracy number wouldn't flag. The layer-by-layer progression from false → strong tells a real story about representation formation.

Current limitations: mean-pooling across tokens loses positional information (hurting question detection). SAE feature sparsity creates heteroscedastic errors even for meaningless labels. The framework correctly reports "weak" in ambiguous cases — it means "I can't confirm strong knowledge with this setup," not "the model doesn't know."

See [examples/12_sae_knowledge.py](examples/12_sae_knowledge.py) (basic), [examples/12b_sae_knowledge_types.py](examples/12b_sae_knowledge_types.py) (expanded, all knowledge types).

## Open Questions

1. **Readout projection during training.** Does readout-projected PCA find emerging features during training, not just post-hoc? The output weights themselves are changing, so the projection subspace evolves. Does this help or hurt? Need to test temporal dynamics with the evolving readout space.

2. **Multi-layer readout projection.** In deeper networks, which readout layer matters? Can we chain projections through multiple layers? In transformers, the relevant readout might be attention heads, not just the final unembedding.

3. **When does PCA suffice?** In models where task-relevant features dominate variance (unlike modular addition's 89/11 split), PCA-discovered directions may be directly useful as DoFs. Characterizing when this holds would determine when ActivationTracker is sufficient without readout projection.

4. **Pre-ReLU structural constraint.** Before ReLU, activations are f(a)+g(b) — additive and mathematically orthogonal to any function of (a+b) mod p. ReLU creates the multiplicative interaction. Does this generalize to other architectures?

5. **Feature-behavioral lag as diagnostic.** Phase 8c showed that feature-level K reaches "strong" hundreds of epochs before test accuracy rises. This gap makes K unsuitable as a training signal, but it remains potentially useful as a diagnostic: if features are "strong" but accuracy is low, the bottleneck is feature composition in the output layer, not feature formation in hidden layers.


**Bispectrum / 3rd-Order Statistics works perfectly, but is fundamentally unscalable.** Because the embedding noise $f(a) + g(b)$ is purely additive and symmetric, its 3rd moment (skewness) is mathematically zero. The Fourier feature requires multiplication (created by the ReLU breaking symmetry). By computing the exact 3D Coskewness Tensor $S_{ijk}$ and unfolding it to find the principal directions of skewness, the algorithm reached an $0.850$ correlation with the Fourier feature using only a **single snapshot** of data. However, while mathematically elegant, this approach is dead on arrival for deep learning: building and decomposing the full $D \times D \times D$ tensor scales as $\mathcal{O}(D^3)$ memory and $\mathcal{O}(D^4)$ compute. It works flawlessly for our $D=128$ toy model (tensor size ~2 million elements), but would be computationally impossible for an LLM where $D=4096$ (tensor size ~68 billion elements).