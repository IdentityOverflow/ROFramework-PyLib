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

Yes, at the embedding/representation level. You'd wrap a model's encoder or a specific layer, feed token embeddings through `observe()`, and assess whether internal representations reliably track specific input features. This is useful for:

- **Probing**: Does layer N "know" about sentiment? Factuality? You get `(correlation, bias, noise, calibration)` instead of just a probe accuracy number.
- **Calibration auditing**: Are the model's confidence scores actually calibrated?
- **Comparing fine-tuned variants**: Same base model, different fine-tunes — which one has stronger/more biased knowledge of a particular feature?

It doesn't work on raw text — you need to pick a numeric representation layer. It's a research tool for interpretability, not a drop-in LLM evaluator.

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
src/ro_framework/
├── core/              # DoF, Value, State (typed data model)
│   ├── dof.py
│   ├── value.py
│   └── state.py
├── observer/          # Observer, ObservationLog, Mapping
│   ├── observer.py
│   └── mapping.py
├── knowledge/         # KnowledgeAssessment, compute_knowledge, trajectory tracking
│   ├── assessment.py
│   └── tracker.py
├── correlation/       # Pearson, MI, temporal, causal detection
│   └── measures.py
├── consciousness/     # ConsciousnessEvaluator, ConsciousnessMetrics
│   └── evaluation.py
└── integration/       # PyTorch bridge, wrappers, activation analysis, training
    ├── torch.py
    ├── wrappers.py
    ├── activation_tracker.py
    └── training.py
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
# All tests
pytest tests/ -v

# Specific module
pytest tests/unit/test_knowledge.py -v

# Run examples
python examples/01_basic_observer.py
python examples/02_pytorch_conscious_observer.py
python examples/03_knowledge_assessment.py
```

## Known Limitations

This is a v0.2.1-dev research library. Be aware of these issues:

**Distributed representations require feature extraction.** In real neural networks, interpretable features are not stored in individual neurons — they are directions across many neurons (superposition). The framework assumes each DoF is a single scalar value, which is correct *after* feature extraction but not for raw activations. To wrap a real model, the world_model mapping must include a feature extraction step (e.g., a Sparse Autoencoder or linear probe) that decomposes distributed activations into monosemantic features. See [Anthropic's work on dictionary learning](https://www.anthropic.com/research/mapping-mind-language-model) for the approach this framework is designed to integrate with. Pre-trained SAEs exist for some open models (GPT-2, Pythia, Gemma) but are model-specific and layer-specific — an SAE trained on one model cannot be reused for another.

**Only tested on toy models.** All examples and tests use small MLPs, identity mappings, and synthetic data. There is no validated example of wrapping a real pre-trained model with SAE feature extraction and producing meaningful knowledge assessments.

**Single-observer only.** The theory describes observers observing each other and observer-relative knowledge. The library only supports individual observers — no multi-observer comparison or ensemble analysis.

## Roadmap

Next:

- SAE integration — load pre-trained SAEs, compose with model layer as world_model mapping
- SAE training tools — train SAEs on arbitrary model activations

Longer-term (research directions):

- Multi-observer systems — comparing knowledge across model ensembles
- Causal vs. correlational knowledge distinction
- Information-theoretic knowledge bounds given observer resolution and boundary
- Automatic DoF discovery — combine ActivationTracker feature emergence with SAE decomposition

Completed (Phase 8):

- Knowledge trajectory tracking — `KnowledgeTracker` records K(d_ext) over training epochs, detects grokking/resonance/forgetting
- Online feature discovery — `ActivationTracker` with Welford's online covariance, PCA stability tracking, eigenvalue spike detection
- Denoising experiments — readout projection (R=0.907), SFA (R=0.806), tensor unfolding (R=0.850), and others
- Knowledge-guided training — negative result: K(d_ext) is a feature-level metric unsuitable for steering training dynamics (see [RESEARCH.md](RESEARCH.md))

## Documentation

- [Theoretical Framework](ro_framework.md) — Complete theoretical foundation (1500+ lines)
- [Research Findings](RESEARCH.md) — Empirical results from training-time knowledge dynamics experiments
- [Examples](examples/) — Runnable demonstrations of each feature

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
