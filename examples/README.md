# RO Framework Examples

Library usage demos showing how to use the Recursive Observer Framework API.

For research experiments (grokking, SAE, reservoir computing), see [experiments/](../experiments/).

## Running Examples

```bash
# Activate conda environment
conda activate ro-framework

# Run an example
python examples/01_basic_observer.py
```

## Available Examples

### 01_basic_observer.py

Defining Degrees of Freedom (Polar DoFs), creating States, building an Observer with a world model, performing observations, computing state distances, DoF normalization.

### 02_pytorch_conscious_observer.py

PyTorch neural network integration: world model (MLP: external → internal), self-model (MLP: internal → internal), recursive self-observation, MC Dropout uncertainty, consciousness evaluation metrics, correlation analysis.

Requires: `pip install -e ".[torch]"`

### 03_knowledge_assessment.py

Knowledge assessment K(d_ext) = (ρ, ε, σ, C): wrapping a function as an Observer, accumulating observations, assessing what the observer knows about external DoFs, knowledge type classification (strong/weak/false/uncertain).

### 04_memory_temporal_correlation.py

Memory detection via temporal correlation (not just buffering): autocorrelated vs. random sequences, multi-DoF memory analysis, temporal correlation profiles at multiple lags.

### 05_consciousness_evaluation.py

Structural consciousness evaluation: `is_conscious()` with custom thresholds, `get_consciousness_metrics()`, recursive depth calculation, multiple consciousness levels, threshold testing.

### 06_serialization.py

Save/load workflow: `Observer.save()` / `Observer.load()` via JSON, serializing DoFs, States, ObservationLog, and Observer configuration.

### 07_wrappers.py

Convenience wrappers: `wrap_callable()`, `wrap_torch_model()`, `create_dofs_for_vector()`, batch observation, knowledge assessment on wrapped models.

## Next Steps

- Read the [theoretical framework](../docs/ro_framework.md) for the full theory
- See [experiments/](../experiments/) for research scripts (grokking, SAE, reservoir computing)
- Check [docs/research/](../docs/research/) for empirical findings
