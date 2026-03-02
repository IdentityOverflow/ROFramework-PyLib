# Recursive Observer Framework

A Python library for wrapping any model as an **Observer** and asking structured questions about what it knows, how well-calibrated it is, and whether it can model itself.

## What is this good for?

Most ML tools focus on *training* models. This library focuses on *understanding* them after the fact.

**Graded knowledge assessment** — Go beyond accuracy. When you wrap a model and feed it data, the library tracks paired (input, output) history and computes a four-dimensional knowledge profile:

- Is the model's internal state *correlated* with the input? (not just "right or wrong")
- Is there *systematic bias*? (consistently wrong in one direction)
- How *noisy* is the mapping? (inconsistent outputs for similar inputs)
- Is *uncertainty calibrated*? (when it says "80% confident", is it right 80% of the time?)

A model can be 90% accurate but systematically biased — this library tells you the difference between "strong", "weak", "false", and "uncertain" knowledge.

**Structural self-modeling evaluation** — If you give a model a second model that predicts its own internal states (a "self-model"), the library evaluates how good that self-modeling is: does it actually predict its own state? Does its stated uncertainty match real errors? Does it know what it doesn't know? These are measurable engineering properties, not philosophical claims.

**Saliency and uncertainty** — Per-input-dimension importance scoring via gradients. MC Dropout uncertainty quantification. All integrated into the same observer abstraction.

### Where it sits in the tool stack

It's not a training framework. It sits *on top* of PyTorch, sklearn, or any callable:

```
Training frameworks (PyTorch, JAX, sklearn)
    ↓ produce models
RO Framework wraps them as Observers
    ↓ provides
Graded knowledge assessment · Calibration auditing
Structural self-modeling evaluation · Saliency analysis
Paired observation history · Uncertainty quantification
```

Closest existing tools: Uncertainty Toolbox (calibration), Captum/SHAP (saliency), sklearn.metrics (evaluation). The RO Framework unifies these through a single observer abstraction with typed input/output dimensions.

### Can it be used on LLMs?

Yes. `SAEObserver` wraps a transformer model (via TransformerLens) with a pre-trained Sparse Autoencoder (via SAELens). You provide labeled texts, and it assesses whether SAE features at any layer track your labels — returning `K(d_ext) = (ρ, ε, σ, C)` instead of just a probe accuracy number. Validated on GPT-2 small with 420 texts across 5 labels (see [docs/research/sae.md](docs/research/sae.md)).

This is useful for:

- **Probing with error decomposition**: Does layer N "know" about code? Formality? You get correlation, systematic bias, noise, and calibration — not just accuracy. GPT-2 layer 0 shows "false" knowledge for code (high ρ but biased), resolved to "strong" by layer 4.
- **Multi-feature assessment**: Single-feature probes miss distributed representations. `max_features=10` uses multiple regression — code detection jumps from ρ=0.72 (single feature) to ρ=0.95 (10 features jointly).
- **Layer-by-layer comparison**: Assess knowledge across layers 0, 4, 8, 11 to see how representations form through the model's depth.
- **Comparing fine-tuned variants**: Same base model, different fine-tunes — which one has stronger/more biased knowledge of a particular feature?

It doesn't work on raw text — it operates at the SAE feature level. Pre-trained SAEs are required (available for GPT-2, Pythia, Gemma). It's a research tool for interpretability, not a drop-in LLM evaluator.

## Installation

### Using pip

```bash
# Core (numpy + scipy)
pip install -e .

# With PyTorch integration
pip install -e ".[torch]"

# Everything
pip install -e ".[all]"
```

### Using conda

If you prefer isolated environments via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or [Miniforge](https://github.com/conda-forge/miniforge)):

```bash
# Create and activate environment
conda create -n ro-framework python=3.10
conda activate ro-framework

# Install the library
pip install -e ".[all]"

# Run tests
pytest tests/ -v
```

## Quick Start

### Wrap a function as an Observer

```python
from ro_framework import PolarDoF, State
from ro_framework.integration.wrappers import wrap_callable, create_dofs_for_vector

# Create DoFs for a 3→2 function
input_dofs = create_dofs_for_vector(3, prefix="sensor", pole_negative=-1.0, pole_positive=1.0)
output_dofs = create_dofs_for_vector(2, prefix="latent", pole_negative=-5.0, pole_positive=5.0)

# Wrap any numpy function as an Observer
import numpy as np
observer = wrap_callable(
    fn=lambda x: np.array([x[0] + x[1], x[0] - x[2]]),
    input_dofs=input_dofs,
    output_dofs=output_dofs,
    name="my_observer",
)

# Observe
external = State(values={d: 0.5 for d in input_dofs})
internal = observer.observe(external)
```

### Assess knowledge

```python
# After making observations, assess what the observer knows
for i in range(50):
    ext = State(values={d: np.random.uniform(-1, 1) for d in input_dofs})
    observer.observe(ext)

assessment = observer.assess_knowledge(input_dofs[0])
print(f"Knowledge type: {assessment.knowledge_type}")  # "strong", "weak", "false", "uncertain"
print(f"Correlation: {assessment.correlation:.3f}")
print(f"Calibration: {assessment.calibration:.3f}")
```

### Wrap a PyTorch model

```python
from ro_framework.integration.wrappers import wrap_torch_model
from ro_framework.integration.torch import create_mlp

world_nn = create_mlp(3, 2, hidden_dims=[16, 8], dropout=0.2)
self_nn = create_mlp(2, 2, hidden_dims=[16, 8], dropout=0.2)

observer = wrap_torch_model(
    model=world_nn,
    input_dofs=input_dofs,
    output_dofs=output_dofs,
    self_model=self_nn,     # adds self-modeling capability
    name="self_aware_model",
    use_dropout_uncertainty=True,
)

# Evaluate self-modeling quality
metrics = observer.get_consciousness_metrics()
print(f"Self-accuracy: {metrics.self_accuracy:.3f}")
print(f"Calibration error: {metrics.calibration_error:.3f}")
print(f"Limitation awareness: {metrics.limitation_awareness:.3f}")
```

## Project Structure

```
ROFramework-PyLib/
├── src/ro_framework/          # Library
│   ├── core/                  #   DoF, Value, State (typed data model)
│   ├── observer/              #   Observer, ObservationLog, Mapping
│   ├── knowledge/             #   KnowledgeAssessment, compute_knowledge, KnowledgeTracker
│   ├── correlation/           #   Pearson, MI, temporal, causal detection
│   ├── consciousness/         #   ConsciousnessEvaluator, ConsciousnessMetrics
│   └── integration/           #   PyTorch bridge, wrappers, SAE, activation analysis
├── tests/                     # Unit tests (331 tests)
├── examples/                  # Library usage demos (01-07)
├── experiments/               # Research experiments
│   ├── grokking/              #   Phase 8: knowledge trajectories, denoising, K-guided training
│   ├── sae/                   #   Phase 9: GPT-2 + SAE knowledge assessment
│   └── reservoir/             #   Direction C: reservoir computing (planned)
└── docs/                      # Documentation
    ├── ro_framework.md        #   Theoretical framework (1500+ lines)
    ├── organic_cognitive_architecture_oca.md  # OCA multi-reservoir design
    └── research/              #   Research findings (mirrors experiments/)
        ├── grokking.md
        ├── sae.md
        └── reservoir.md
```

## Key Features

### Knowledge Assessment — `K(d_ext) = (ρ, ε, σ, C)`

Graded, observer-relative knowledge. After observation history accumulates, assess how well the observer tracks any external DoF:

- **ρ** (correlation): How strongly an internal DoF tracks the external DoF
- **ε** (systematic_error): Consistent bias in the mapping
- **σ** (random_error): Noise / inconsistency
- **C** (calibration): Whether stated uncertainty matches actual error

Knowledge is classified as `"strong"`, `"weak"`, `"false"`, or `"uncertain"`.

### Structural Self-Modeling Evaluation

A model with a self-model (internal → internal mapping) is evaluated on:

- **Self-accuracy**: How well the self-model predicts actual internal states
- **Architectural similarity**: Structural comparison of world and self models
- **Calibration**: Expected Calibration Error of self-model uncertainty
- **Metacognition**: Behavioral tests for self-awareness capability
- **Limitation awareness**: Whether uncertainty increases on harder inputs

### Observation Log

Every `observe()` call records an `ObservationPair(external_state, internal_state, timestamp)`. This paired history drives both knowledge assessment and temporal memory analysis — no separate buffer needed.

### PyTorch Integration

- `TorchNeuralMapping`: Wraps `nn.Module` with automatic state <-> tensor conversion
- `TorchObserver`: Batched inference, gradient-based saliency, MC Dropout uncertainty
- `create_mlp()`: Quick MLP construction with dropout and batch norm options

## Running Tests

```bash
# All tests (331 tests)
pytest tests/ -v

# Specific module
pytest tests/unit/test_knowledge.py -v

# Run examples (library demos, no GPU needed)
python examples/01_basic_observer.py
python examples/03_knowledge_assessment.py

# Run experiments (research scripts, some require GPU / SAE models)
python experiments/grokking/08_knowledge_tracker.py
python experiments/sae/12b_sae_knowledge_types.py
```

## Known Limitations

This is a v0.2.1-dev research library. Be aware of these issues:

**Distributed representations require feature extraction.** In real neural networks, interpretable features are directions across many neurons (superposition), not individual neuron activations. The framework's DoFs are scalar values, which is correct *after* feature extraction. For transformer models, `SAEObserver` provides this bridge: it integrates pre-trained Sparse Autoencoders (via SAELens/TransformerLens) to decompose activations into monosemantic SAE features that become internal DoFs. Multi-feature assessment (`max_features=N`) then uses multiple regression to capture knowledge distributed across features. Pre-trained SAEs exist for GPT-2, Pythia, and Gemma but are model-specific and layer-specific — an SAE trained on one model cannot be reused for another. For models without pre-trained SAEs, you'd need to train one first (not yet supported by the library).

**Validated on GPT-2 small only.** SAE integration (`SAEObserver`) has been tested on GPT-2 small with pre-trained SAEs from the `gpt2-small-res-jb` release and 420 labeled texts. Multi-feature assessment (max_features=10) yields: code detection strong (ρ=0.95), formality strong at deeper layers (ρ=0.74), question detection weak with high systematic error (ρ=0.52, ε=0.59), sentiment weak (ρ=0.29). Code detection is biased toward Python-like syntax — SQL/bash weaken detection. Larger models have not been tested.

**Token aggregation loses positional information.** SAEObserver mean-pools SAE features across the token sequence. This discards positional signal — e.g., a question mark at the final position is highly informative but gets averaged away. This likely explains why question detection underperforms despite GPT-2 clearly "knowing" questions.

**Single-observer only.** The theory describes observers observing each other and observer-relative knowledge. The library only supports individual observers — no multi-observer comparison or ensemble analysis.

## Roadmap

### Phase 9: Interpretability Dashboard (Direction A)

Make the library work on real models, not just toys.

- SAE training tools — train SAEs on arbitrary model activations

### Phase 10: Self-Aware Training (Direction B)

Use the framework as introspection machinery inside a training loop.

- Training-time feature introspection — periodically assess K(d_ext) on SAE-extracted features during fine-tuning
- Self-model integration — model maintains a structural map of what it knows as it learns
- Multi-observer comparison — compare knowledge profiles across model checkpoints, fine-tune variants, or ensemble members
- Multimodal bridge — assess alignment between visual and linguistic feature spaces using observer-relative knowledge

### Reservoir Computing Research (Direction C)

Apply the framework to reservoir computing — the most structurally natural substrate for RO observers. RC implements O = (B, M, R, Mem) as first-class architectural properties: fixed mapping (reservoir), clear boundary (input/readout), explicit memory (fading echoes), measurable resolution (spectral radius). See [docs/research/reservoir.md](docs/research/reservoir.md) for full rationale.

- RC-1: Single reservoir on modular addition — ESN baseline with K trajectory tracking
- RC-2: Reservoir spectral analysis — eigenmodes as natural DoFs
- RC-3: K-guided readout training — retry Phase 8c where feature-behavioral lag shouldn't exist
- RC-4: Multi-reservoir knowledge assessment — first multi-observer experiment
- RC-5: Reservoir self-model — structural consciousness via self-observing reservoir
- RC-6: Toward OCA — full multi-reservoir architecture

### Research directions

- Causal vs. correlational knowledge distinction
- Information-theoretic knowledge bounds given observer resolution and boundary
- Automatic DoF discovery — combine ActivationTracker feature emergence with SAE decomposition

Completed (Phase 9):

- SAE integration — `SAEObserver` wraps TransformerLens + SAELens, SAE feature activations become internal DoFs
- Multi-feature knowledge assessment — `compute_knowledge(max_features=N)` uses multiple regression to capture distributed representations
- K tuple error decomposition — OLS residual-based ε (heteroscedasticity) and C (error uniformity); all four knowledge types reachable
- GPT-2 validation (420 texts, 5 labels) — code: strong (ρ=0.95), formality: strong at L8+ (ρ=0.74), questions: weak with high ε, sentiment: weak, random: weak
- Feature-level knowledge profiles — `top_features_for()` ranks SAE features by correlation with any label
- Multi-layer comparison — knowledge assessed across layers 0, 4, 8, 11; "false" knowledge at layer 0 for code (bias resolved by layer 4)

Completed (Phase 8):

- Knowledge trajectory tracking — `KnowledgeTracker` records K(d_ext) over training epochs, detects grokking/resonance/forgetting
- Online feature discovery — `ActivationTracker` with Welford's online covariance, PCA stability tracking, eigenvalue spike detection
- Denoising experiments — readout projection (R=0.907), SFA (R=0.806), tensor unfolding (R=0.850), and others
- Knowledge-guided training — negative result: K(d_ext) is a feature-level metric unsuitable for steering training dynamics (see [docs/research/grokking.md](docs/research/grokking.md))

## Documentation

- [Theoretical Framework](docs/ro_framework.md) — Complete theoretical foundation (1500+ lines)
- [OCA Architecture](docs/organic_cognitive_architecture_oca.md) — Multi-reservoir cognitive architecture design
- [Grokking & Feature Discovery](docs/research/grokking.md) — Phase 8 experiments: knowledge trajectories, denoising, K-guided training
- [SAE Knowledge Assessment](docs/research/sae.md) — Phase 9 experiments: GPT-2 + SAE knowledge profiles
- [Reservoir Computing](docs/research/reservoir.md) — Direction C: RO-RC alignment and RC research roadmap
- [Examples](examples/) — Library usage demos (01-07)
- [Experiments](experiments/) — Research experiment scripts (grokking, SAE, reservoir)

## License

Apache License 2.0 — see LICENSE file for details.

## Citation

```bibtex
@software{ro_framework,
  title = {Recursive Observer Framework},
  author = {RO Framework Contributors},
  year = {2026},
  url = {https://github.com/IdentityOverflow/ROFramework}
}
```

---

**Version**: 0.2.1-dev | **Python**: 3.9+
