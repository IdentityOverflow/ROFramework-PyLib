# Reservoir Computing Research

Experimental research applying the Recursive Observer Framework to reservoir computing architectures. Motivated by the structural alignment between the RO framework's observer theory and RC's computational paradigm.

## Why RO and RC Work Well Together

The Recursive Observer Framework defines an observer as O = (B, M, R, Mem) — boundary, mapping, resolution, memory. Reservoir computing implements each of these as a first-class architectural property, not an emergent training artifact.

### Observer structure maps directly onto a reservoir

**Boundary (B):** A reservoir has a crisp boundary. Fixed input weights define what external DoFs enter. The recurrent state is "inside." The trained readout defines what gets communicated out. This is cleaner than a transformer where the boundary is blurred across residual streams, attention heads, and layer norms.

**Mapping (M):** The reservoir's fixed recurrent dynamics ARE M_world — they transform external configurations into high-dimensional internal states. Crucially, M is not learned. The RO framework defines M as "a structural relation between external and internal DoF configurations" — it never requires M to be trained. A random reservoir satisfies this definition directly.

**Resolution (R):** A reservoir with N neurons has N internal DoFs. Its spectral radius determines temporal resolution — how many timesteps back it can distinguish input history. This is exactly "minimum distinguishable difference on each internal DoF" from the theory (§3.5). Near criticality (spectral radius ≈ 1.0), temporal resolution is maximized.

**Memory (Mem):** The framework defines memory as "correlation constraint across the temporal DoF — internal states at different temporal positions show correlation not reducible to instantaneous external correlations" (§3.6). This is the literal definition of an echo state network. Recurrent connections create fading echoes: h(t) depends on h(t-1), h(t-2), ... The reservoir's entire value proposition IS the framework's definition of memory.

### Knowledge K(d_ext) maps onto readout training

In a standard neural network, knowledge is entangled between feature extraction (hidden weights) and readout (output weights). In RC, only the readout trains. This creates a clean separation:

- **Before readout training:** K(d_ext) on reservoir states measures representational capacity — what correlations exist in the high-dimensional expansion, waiting to be used. This is "latent knowledge."
- **After readout training:** K(d_ext) measures actual knowledge — what the system has learned to extract.
- **The readout weights ARE the knowledge.** No ambiguity. No feature-behavioral lag. No hidden-layer overfitting confound.

The library's `compute_knowledge(max_features=N)` directly answers: "how many reservoir neurons does the readout need to combine to track this external DoF?"

### Spectral modes are natural DoFs

A reservoir's eigenmodes (from eigendecomposition of W_reservoir) are its natural frequency basis. Each eigenvalue λ_i defines a temporal mode: |λ_i| determines decay rate, arg(λ_i) determines oscillation frequency. The reservoir is literally a bank of damped oscillators.

This connects directly to the Phase 8a finding where MLP neurons locked into Fourier modes during grokking. In RC, these modes exist from initialization (fixed weights). The readout learns which modes to attend to. K(d_ext) before training tells you which modes correlate with the task; the readout after training tells you which ones the system chose.

The "model is its own best radio tuner" insight (Phase 8b) becomes concrete: the readout IS the tuner, and the reservoir IS the radio.

### Problems from Phase 8 that RC resolves

**No denoising problem.** In the MLP grokking experiments, 89% of activation variance was embedding noise. In RC, the reservoir state IS the feature space — there's no separate embedding to create noise. The readout projection trick becomes trivial because the readout IS the only trained component.

**No feature-behavioral lag.** Phase 8c (K-guided training) failed because K reached "strong" 500 epochs before the model generalized — feature formation preceded readout learning. In RC, there are no hidden-layer features to form. When the readout learns to use a reservoir mode, that IS generalization. K(d_ext) should track behavior directly, which means K-guided training might actually work here.

**No SAE needed.** The "distributed representations require feature extraction" limitation (README Known Limitations) applies to transformers because interpretable features are directions across neurons in superposition. In a reservoir, neurons are fixed random projections. Their activations are directly interpretable as DoFs — no decomposition step required.

### Multi-reservoir OCA as multi-observer system

The OCA architecture (see [organic_cognitive_architecture_oca.md](../organic_cognitive_architecture_oca.md)) defines multiple interconnected reservoirs with trainable bottleneck pipes. In RO framework terms:

- Each reservoir is an observer with its own (B, M, R, Mem)
- Bottleneck pipes are inter-observer communication channels
- K(d_ext) can be assessed independently at each reservoir
- The Salience/Value reservoir observing other reservoirs' states is M_self — structural consciousness
- The Logic/Simulation reservoir forward-simulating dynamics is M_meta — metacognition

This maps directly to the framework's recursive depth levels:
- Sensory reservoirs = Level 0 (external → internal only)
- Salience/Value reservoir = Level 1 (internal → internal)
- Logic/Simulation reservoir = Level 2 (meta-cognitive)

The "single-observer only" limitation in the current library would be addressed by multi-reservoir experiments.

---

## Research Roadmap

### RC-1: Single Reservoir on Modular Addition

The same task as Phase 8a, but with a fixed random reservoir instead of a fully trainable MLP. This became the first successful RC baseline for modular addition, but only after relaxing the original "pure linear readout" constraint.

**Final setup that worked:**
- Reservoir: 729 neurons, fixed random `W_in`, fixed random recurrent `W_res`, fixed random bias
- Dynamics: sequential ESN update, one-hot `a` then one-hot `b`, with 5 recurrent settle steps per symbol
- Spectral radius: 0.99
- Input scaling: 0.1
- Train/test split: 75/25 over all `(a, b)` pairs mod 97
- Readout: trainable nonlinear head `Linear(729→729) → ReLU → Linear(729→97)`
- Optimizer: AdamW, lr = `5e-3`, weight decay = `1.2`
- Knowledge tracking: `K(d_ext)` on sum-averaged readout hidden features, not raw logits

**What failed first:**
- A linear readout on top of a fixed reservoir memorized the training set but generalized at only `0.0%–0.3%` test accuracy, even when ridge regression nearly saturated train accuracy.
- Tracking `K(d_ext)` on frozen reservoir states during readout training was a measurement bug: since the reservoir never changed, the trajectory was constant by construction.
- A two-step sequential ESN with a linear readout was still too weak: it improved over the feedforward random-feature baseline, but remained far from solving the task.

**What changed the result:**
- Adding recurrent settle steps increased the effective nonlinear expansion of the fixed reservoir.
- Adding a nonlinear readout head let the trained component compose the latent reservoir features rather than only linearly selecting them.
- Tracking `K` on the readout hidden layer made the knowledge signal follow what was actually learning.

**Result:**
- Test accuracy rose smoothly from near chance to `99%` by epoch `6750`, reaching `95%` around epoch `3000–3250`.
- The model hit `100%` train accuracy by epoch `250` but continued improving on test accuracy for thousands of epochs after that, showing a clear memorization-to-generalization transition.
- Initial latent reservoir knowledge was only moderate: single-feature `ρ ≈ 0.36–0.44`, multi-feature `ρ ≈ 0.65–0.74` on the discovered Fourier frequencies.
- After training, readout hidden features reached effectively perfect Fourier correlation: all tracked `sin`/`cos` features ended at `ρ = 0.999` with low noise.

**Interpretation:**
- The reservoir does contain useful latent Fourier signal before training, but not in a form that a linear readout can exploit well enough for modular addition at `p=97`.
- The successful model is still a valid RC result because the reservoir remains fixed and all learning is confined to the readout head. But it is no longer a "pure ESN + linear probe" result.
- This means the cleanest current claim is: **fixed reservoirs can support modular-addition generalization, but the readout may need nonlinear composition to extract the latent structure.**

**What we learn:**
- RC is viable for this task, but the original linear-readout hypothesis was too optimistic.
- There is still a form of feature-behavior lag inside RC once the readout becomes multi-layer: latent reservoir signal exists before the trained head can use it behaviorally.
- `K(d_ext)` remains useful, but the probe location matters. On fixed reservoirs it measures capacity; on readout hidden features it measures extracted knowledge.
- This gives a practical baseline for later multi-reservoir and K-guided experiments: keep reservoir dynamics fixed, but do not artificially cripple the readout.

### RC-2: Reservoir Spectral Analysis

Deep dive into the reservoir's eigenmodes and their relationship to task-relevant features.

**Experiments:**
- Eigendecompose W_reservoir, characterize the mode spectrum
- Measure K(d_ext) per eigenmode: which modes correlate with Fourier features?
- Compare reservoirs with different spectral radii: does criticality matter for K?
- Test whether reservoir size (N) determines which Fourier frequencies are representable
- Spectral radius sweep: how does the edge of chaos relate to knowledge quality?

**What we learn:**
- The relationship between reservoir architecture and representational capacity
- Whether K(d_ext) can predict which tasks a reservoir can solve before training
- Optimal reservoir design principles from a knowledge-assessment perspective

### RC-3: K-Guided Readout Training

Retry Phase 8c's failed experiment in the RC context where feature-behavioral lag shouldn't exist.

**Hypothesis:** Since the readout IS behavior in RC, K(d_ext) should be a valid training signal. Features that show "strong" K in the reservoir should be prioritized in the readout.

**Approaches:**
- K-weighted ridge regression: weight reservoir neurons by their K correlation with the target
- Iterative readout refinement: assess K, prune low-K neurons from readout, retrain
- Adaptive regularization: increase L2 on readout weights for low-K reservoir neurons

**Success criterion:** K-guided readout training should match or beat standard ridge regression on modular addition.

### RC-4: Multi-Reservoir Knowledge Assessment

First multi-observer experiment. Two or more reservoirs with trainable bottleneck connections.

**Setup:**
- Reservoir A: receives input (a, b)
- Reservoir B: receives state of Reservoir A through a trainable bottleneck
- Readout: from Reservoir B to output logits

**Key questions:**
- Does K(d_ext) at Reservoir A differ from K at Reservoir B?
- Does the bottleneck learn to transmit the task-relevant modes?
- Can we measure "knowledge transfer" between observers via K trajectories?
- Is Reservoir B's knowledge of the external label mediated entirely by Reservoir A's state?

**What we learn:**
- How knowledge propagates across observer boundaries
- Whether the bottleneck acts as an information filter (transmitting high-K modes)
- Foundation for the full OCA architecture

### RC-5: Reservoir Self-Model (Consciousness Structure)

A reservoir that models its own state — the structural definition of consciousness from §5.1.

**Setup:**
- World reservoir: maps external input to internal state
- Self reservoir: maps world reservoir's state to a prediction of that state
- Compare M_self architecture to M_world architecture (same architectural type per §5.1)

**Key questions:**
- Can a reservoir predict its own state? (Trivially yes for linear readout, interesting for nonlinear)
- Does K(d_self) (knowledge of own internal state) differ from K(d_ext)?
- Does recursive depth > 1 emerge when the self-reservoir models its own modeling?
- Can we measure metacognitive accuracy using the framework's ConsciousnessEvaluator?

**What we learn:**
- Whether RC provides a cleaner substrate for structural consciousness experiments
- How self-modeling quality relates to reservoir properties (spectral radius, size)
- First step toward the full OCA architecture with self-aware components

### RC-6: Toward OCA

Scale up to the full Organic Cognitive Architecture with multiple specialized reservoirs, RPE-gated plasticity, and circadian consolidation.

**This phase depends on results from RC-1 through RC-5.**

---

## Experiments

*Results will be added as experiments are completed.*

### RC-1: Modular Addition with Echo State Network

**Status:** Completed

**Summary:**
- Fixed reservoir + linear readout: failed to generalize (`≤ 0.3%` test)
- Fixed reservoir + nonlinear readout head: succeeded, reaching `99%` test accuracy on `p=97`
- Latent Fourier knowledge exists in the reservoir before training, but the trained head must learn to compose it

See [experiments/reservoir/rc1_modular_addition.py](../../experiments/reservoir/rc1_modular_addition.py) and [experiments/reservoir/rc1_findings.md](../../experiments/reservoir/rc1_findings.md).

---

## References

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- Lukoševičius, M. & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
- He, Z. et al. (2025). On the Mechanism and Dynamics of Modular Addition. (Phase 8a reference)
- [organic_cognitive_architecture_oca.md](../organic_cognitive_architecture_oca.md) — The OCA design document
- [ro_framework.md](../ro_framework.md) — The Recursive Observer Framework theory
- [grokking.md](grokking.md) — Phase 8 empirical findings (grokking, denoising, K-guided training)
