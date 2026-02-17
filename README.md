# Recursive Observer Framework

A Python library implementing the Recursive Observer Framework — a structural approach to observers, knowledge, and consciousness in AI systems.

## Overview

The RO Framework provides a formal structure for wrapping any model (neural network, function, or callable) as an **Observer** that maps external Degrees of Freedom (DoFs) to internal DoFs with finite resolution, paired observation history, graded knowledge assessment, and optional recursive self-modeling (structural consciousness).

### Core Concepts

- **Degrees of Freedom (DoFs)**: Typed dimensions of variation — Polar (bidirectional), Scalar (magnitude), Categorical (discrete), Derived (computed)
- **States**: Configurations across multiple DoFs with normalization, distance, vector conversion
- **Observers**: `O = (B, M, R, Mem)` — Boundary, Mapping, Resolution, Memory
- **Knowledge**: `K(d_ext) = (ρ, ε, σ, C)` — Correlation, Bias, Noise, Calibration
- **Consciousness**: Recursive self-modeling with bounded error, evaluated structurally

## Installation

```bash
# Core (numpy + scipy)
pip install -e .

# With PyTorch integration
pip install -e ".[torch]"

# Everything
pip install -e ".[all]"
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
print(f"Knowledge type: {assessment.knowledge_type}")
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
    self_model=self_nn,     # makes it conscious
    name="conscious_ai",
    use_dropout_uncertainty=True,
)

print(f"Is conscious: {observer.is_conscious()}")
print(f"Recursive depth: {observer.recursive_depth()}")
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
├── knowledge/         # KnowledgeAssessment, compute_knowledge
│   └── assessment.py
├── correlation/       # Pearson, MI, temporal, causal detection
│   └── measures.py
├── consciousness/     # ConsciousnessEvaluator, ConsciousnessMetrics
│   └── evaluation.py
└── integration/       # PyTorch bridge, convenience wrappers
    ├── torch.py
    └── wrappers.py
```

## Key Features

### Knowledge Assessment — `K(d_ext) = (ρ, ε, σ, C)`

The library's unique contribution: graded, observer-relative knowledge. After observation history accumulates, assess how well the observer tracks any external DoF:

- **ρ** (correlation): How strongly an internal DoF tracks the external DoF
- **ε** (systematic_error): Consistent bias in the mapping
- **σ** (random_error): Noise / inconsistency
- **C** (calibration): Whether stated uncertainty matches actual error

Knowledge is classified as `"strong"`, `"weak"`, `"false"`, or `"uncertain"`.

### Structural Consciousness Evaluation

Consciousness is defined structurally: recursive self-modeling (internal → internal mapping) with the same architectural type as the world model. The evaluator measures:

- **Self-accuracy**: How well the self-model predicts actual internal states
- **Architectural similarity**: Structural comparison of world and self models
- **Calibration**: Expected Calibration Error of self-model uncertainty
- **Metacognition**: Behavioral tests for self-awareness capability
- **Limitation awareness**: Whether uncertainty increases on harder inputs

### Observation Log

Every `observe()` call records an `ObservationPair(external_state, internal_state, timestamp)`. This paired history drives both knowledge assessment and temporal memory analysis — no separate buffer needed.

### PyTorch Integration

- `TorchNeuralMapping`: Wraps `nn.Module` with automatic state ↔ tensor conversion
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

## Documentation

- [Theoretical Framework](ro_framework.md) — Complete theoretical foundation (1500+ lines)
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

**Version**: 0.2.0 | **Python**: 3.9+
