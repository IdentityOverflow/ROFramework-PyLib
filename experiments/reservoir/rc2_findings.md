# RC-2 Findings: Noise-to-Signal Generalization via Behavioral Imprinting

Empirical results from the second reservoir computing experiment.
See `rc2_noise_to_signal.py` for full implementation.

## Setup

- Markov chain: V=10 alphabet, k=3 order, Dirichlet(α=0.5) transitions
  (sparse, peaked distributions; V^k = 1000 distinct contexts)
- Target model: GRU (hidden=128) trained until val_CE ≈ 1.91
  (ground truth entropy: 1.667 nats, gap ≈ 0.25 nats)
- Reservoir: ESN with fixed random weights, ridge regression readout (λ=1e-4)
  Sizes: [64, 256, 1024], spectral radius 0.97
- Training conditions: Random, Coherent, Mixed (80/20), Oracle-Random, Oracle-Coherent
- N sweep: [25, 75, 250, 750, 2500, 7500] sequences × 21 pairs/seq
- Evaluation: 500 held-out coherent sequences

## Main Result

**The core hypothesis is supported. Random training generalizes as well as coherent training.**

KL(Markov‖ESN) ratio at best N, largest reservoir (N=1024):

| Condition pair | KL ratio |
|---|---|
| Random(GRU) / Coherent(GRU) | 1.02× |
| Random(Oracle) / Coherent(Oracle) | 1.00× |

The conditions trained on random sequences generalize to coherent inputs
essentially identically to the conditions trained on coherent sequences.
This holds for both the GRU teacher (behavioral imprinting) and the Oracle
teacher (direct Markov distributions).

## Complexity Probe

PCA on GRU output distributions across 3000 input sequences:

| Input type | d_90 | d_95 | d_99 |
|---|---|---|---|
| Random | 8 | 9 | 9 |
| Coherent | 8 | 9 | 9 |

**Effective dimensionality = 9 = V−1 in both cases.**
This matches the V−1 simplex prediction exactly, not V^k = 1000 (the nominal
context count). The 1000 distinct conditional distributions, each lying in the
9-dimensional probability simplex, collapse to a 9-dimensional manifold.

This cleanly separates effective complexity (9) from nominal input space (1000),
and predicts that N_crit should scale with ~9, not 1000.

## Learning Curves (N=1024 reservoir, GRU teacher)

| N seqs | Pairs | Random KL | Coherent KL | Uniform KL |
|---|---|---|---|---|
| 25 | 525 | 1.556 | 1.635 | 0.635 |
| 75 | 1575 | 1.313 | 1.348 | 0.635 |
| 250 | 5250 | 0.677 | 0.672 | 0.635 |
| 750 | 15750 | 0.605 | 0.594 | 0.635 |
| 2500 | 52500 | 0.583 | 0.571 | 0.635 |
| 7500 | 157500 | 0.577 | 0.565 | 0.635 |

KL values > 0.635 (uniform) indicate the ESN is actively wrong — ridge regression
is underdetermined and produces bad predictions. The transition below the uniform
baseline (N_crit) happens around N=250–750 sequences (5000–15000 pairs).

The learning curve is **gradual**, not a sharp phase transition. Improvement
continues monotonically but slowly — no grokking-like jump.

## Accuracy vs. KL Discrepancy

Despite KL remaining close to the uniform baseline (0.635), accuracy is
significantly above chance:

| Condition | KL | Accuracy | Chance |
|---|---|---|---|
| Random (GRU), N=1024, N=7500 | 0.577 | 0.212 | 0.10 |
| Coherent (GRU), N=1024, N=7500 | 0.565 | 0.229 | 0.10 |

Accuracy is 2× above chance even when KL barely improves over uniform. This
happens because KL heavily penalizes distribution confidence mismatch: the ESN
learns to predict the correct mode (acc gain) but can't sharpen the distribution
enough (KL penalty). The linear readout learns the identity of the most likely
character but not the full probability mass concentration.

## Oracle Teacher is Harder than GRU Teacher

Surprisingly, training with exact Markov distributions (Oracle) performs worse
than training with the imperfect GRU distributions:

| Teacher | N=1024, N=7500 | KL | Acc |
|---|---|---|---|
| GRU (CE≈1.91) | Random | 0.577 | 0.212 |
| Oracle (exact) | Random | 0.684 | 0.160 |

**Reason**: Dirichlet(0.5) distributions have near-zero probability on most
characters, creating extreme log-prob targets (log(p) → −∞). Ridge regression
on these extreme values overfits severely at small N, and the large-N asymptote
is also worse due to the high condition number. The GRU's smoother approximation
acts as a natural regularizer — the log-prob range is narrower, the regression
problem is better-conditioned.

Practical implication: **smoothness of the teacher signal matters for imprinting**.
A noisy-but-smooth teacher can be a better substrate for behavioral cloning than
an exact-but-sharp target, especially with linear readout learners.

## Reservoir Size Effect

| Reservoir | Best condition | Best KL | Acc |
|---|---|---|---|
| N=64 | Coherent (GRU), N=7500 | 0.625 | 0.143 |
| N=256 | Coherent (GRU), N=7500 | 0.612 | 0.153 |
| N=1024 | Coherent (GRU), N=7500 | 0.565 | 0.229 |

Larger reservoir → better performance. N=1024 still hasn't saturated. This
suggests the reservoir needs enough neurons to maintain linearly separable
representations of the 1000 distinct k-gram contexts. A linear readout can
decode which k-gram is active when the representations are well-separated.

## What This Means

**Signal is organized noise — the premise holds at small scale.**

A reservoir trained exclusively on random (unstructured) I/O pairs from a target
model generalizes to coherent inputs at the same level as one trained on coherent
pairs. The structure of the target function is recoverable from random sampling
because the output manifold has low effective dimensionality (V−1 = 9) relative
to the nominal input space (V^k = 1000).

**But the linear readout is the bottleneck, not the training data.**

Even the coherent condition fails to recover the full Markov structure with a
linear readout. The ESN state represents a complex, mixed history of past
characters (exponentially decaying), and the linear projection can't fully
separate 1000 distinct k-gram contexts from this. The structure is recoverable
(accuracy 2× above chance) but not fully — KL remains far from optimal.

## Open Questions

1. **Why doesn't the learning curve phase-transition?** Modular addition (RC-1)
   showed gradual accuracy growth too, but with grokking-like dynamics in the
   K trajectory. Here there's no sharp transition. Is k-gram extraction
   inherently gradual for ESNs?

2. **What's the GRU teacher's actual accuracy?** The GRU's CE=1.91 suggests it
   hasn't fully memorized 1000 contexts. Does the GRU itself show gradual
   learning of k-gram contexts, or is there a structural limit?

3. **Does a nonlinear readout fix the ceiling?** RC-1 showed that nonlinear
   readout is necessary for modular addition. For Markov chains, the function
   is simpler — but the extraction of k-gram from fading memory state may still
   require nonlinearity.

## Next Steps (Phase 2)

1. **Reduce k to 2**: V^k = 100 contexts (vs 1000). GRU should converge fully.
   Check if the phase transition becomes sharper and N_crit matches effective
   complexity predictions.

2. **Nonlinear readout**: Replace ridge regression with a small MLP readout
   trained with cross-entropy (as in RC-1). Expected to dramatically improve KL
   while keeping the reservoir fixed. Core hypothesis test would be cleaner.

3. **Adaptive reservoir (Seed architecture)**: Replace fixed random reservoir
   with Hebbian+contrastive learning. Tests whether self-organization finds
   structure faster than random projections — the question RC-2 was designed
   to motivate.

4. **N_crit vs effective complexity scaling**: Run with multiple k values
   (k=1,2,3,4) and measure N_crit for each. If N_crit scales with V−1 (not V^k),
   it supports the effective complexity interpretation from the complexity probe.
