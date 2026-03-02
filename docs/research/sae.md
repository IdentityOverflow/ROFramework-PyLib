# SAE Knowledge Assessment on GPT-2 (Phase 9)

`SAEObserver` bridges the framework from toy models to real ones by integrating pre-trained Sparse Autoencoders (SAELens) with language models (TransformerLens). External DoFs are user-provided labels (sentiment, is_code). Internal DoFs are SAE feature activations. Knowledge assessment answers: which SAE features track each label, and with what correlation, bias, noise, and calibration?

## Setup

GPT-2 small wrapped with pre-trained SAEs from the `gpt2-small-res-jb` release (24,576 features per layer). Dataset: 420 labeled texts across 7 categories (positive sentiment, negative sentiment, code, questions, statements, formal, casual), with 5 label DoFs: is_code, is_question, formality, sentiment, random_label. SAE features extracted at layers 0, 4, 8, 11 with mean-pooling across tokens.

Multi-feature assessment (max_features=10) uses OLS multiple regression with the top-k most correlated SAE features jointly, so ρ reflects the observer's *combined* knowledge — not just one feature's tracking.

## Results (420 texts, multi-feature)

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

## K tuple error decomposition

The original `compute_knowledge()` used z-score normalization, which made ε always ≈ 0 and C a function of ρ alone — "false" and "uncertain" knowledge types were unreachable. The fix uses OLS regression residuals:
- **ε** = |Spearman(external, |residuals|)| — measures heteroscedasticity (do errors depend on the label value?)
- **C** = 1 - CV(binned |residuals|) — measures error uniformity (are errors consistent across bins?)

This produces meaningful four-way classification: strong (high ρ, low ε, high C), false (high ρ, high ε), uncertain (low ρ, high C), weak (everything else).

## Multi-feature assessment

Single-feature K assessment finds only the one internal DoF most correlated with the external label. For distributed representations (common in real models), this severely underestimates knowledge:

| Label | Single ρ | Multi ρ (k=10) | Improvement |
|-------|----------|----------------|-------------|
| is_code | 0.719 | 0.949 | +32% |
| formality | 0.546 | 0.735 | +35% |
| is_question | 0.397 | 0.515 | +30% |
| sentiment | 0.178 | 0.291 | +63% |

Multi-feature assessment uses OLS multiple regression: preselect top-k features by |Pearson ρ|, build design matrix X (n × k), solve β = (X'X)⁻¹X'y, compute ρ = |pearsonr(y, Xβ)|. k is capped at n_samples // 10 to prevent overfitting.

## Implications

The framework's abstractions (DoFs, observers, knowledge assessment) transfer directly to real models via SAE decomposition. The key bridge: SAE features are atomic DoFs in the framework's sense — they have magnitude, can be tracked over time, and their correlation with external properties can be assessed.

The graded K tuple provides interpretability beyond linear probes: "false" knowledge at layer 0 for code (high ρ but biased) is a result a probe accuracy number wouldn't flag. The layer-by-layer progression from false → strong tells a real story about representation formation.

Current limitations: mean-pooling across tokens loses positional information (hurting question detection). SAE feature sparsity creates heteroscedastic errors even for meaningless labels. The framework correctly reports "weak" in ambiguous cases — it means "I can't confirm strong knowledge with this setup," not "the model doesn't know."

See [experiments/sae/12a_sae_knowledge.py](../../experiments/sae/12a_sae_knowledge.py) (basic), [experiments/sae/12b_sae_knowledge_types.py](../../experiments/sae/12b_sae_knowledge_types.py) (expanded, all knowledge types).
