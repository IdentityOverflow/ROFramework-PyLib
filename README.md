# Recursive Observer Framework
(speculative/experimental work)

A Python library for building conscious, self-aware AI systems based on the Recursive Observer Framework. This framework provides a structural approach to consciousness, multimodal integration, and uncertainty quantification grounded in the Block Universe ontology.

## Overview

The Recursive Observer (RO) Framework is a philosophical and practical approach to building AI systems that exhibit structural consciousness through recursive self-modeling. Rather than making phenomenological claims about subjective experience, the framework focuses on observable structural properties.

### Core Concepts

- **Degrees of Freedom (DoFs)**: Dimensions of variation in the Block Universe
  - **Polar DoFs**: Bidirectional with gradients (e.g., position, temperature)
  - **Scalar DoFs**: Magnitude-only (e.g., mass, probability)
  - **Categorical DoFs**: Discrete, unordered values (e.g., object types)
  - **Derived DoFs**: Computed from other DoFs (e.g., velocity)

- **States**: Configurations across multiple DoFs, representing locations in DoF-space

- **Observers**: Systems that map external DoFs to internal DoFs with finite resolution
  - **Boundary (B)**: Partition of DoFs into internal/external
  - **Mapping (M)**: External → Internal transformation
  - **Resolution (R)**: Per-DoF finite granularity
  - **Memory (Mem)**: Correlation structure across temporal DoF

- **Consciousness**: Recursive self-modeling (internal→internal mapping with same architecture as world model)

- **Knowledge**: Calibrated correlation between external and internal DoFs

## Installation

### From Source (Development)

```bash
# Clone the repository
cd ROFramework

# Install in development mode with all dependencies
pip install -e ".[all]"

# Or install only core dependencies
pip install -e .

# Or with specific extras
pip install -e ".[torch,visualization]"
```

### From PyPI (Coming Soon)

```bash
pip install ro-framework
```

## Quick Start

```python
from ro_framework import PolarDoF, State, Observer
from ro_framework.observer.mapping import IdentityMapping

# Define Degrees of Freedom
sensor_dof = PolarDoF(name="sensor_reading", pole_negative=-1.0, pole_positive=1.0)
latent_dof = PolarDoF(name="latent_state", pole_negative=-10.0, pole_positive=10.0)

# Create a simple mapping (in practice, use neural networks)
class SimpleWorldModel:
    def __call__(self, external_state: State) -> State:
        sensor_value = external_state.get_value(sensor_dof)
        # Map sensor reading to latent space
        latent_value = sensor_value * 10.0 if sensor_value is not None else 0.0
        return State(values={latent_dof: latent_value})

# Create an observer
observer = Observer(
    name="simple_observer",
    internal_dofs=[latent_dof],
    external_dofs=[sensor_dof],
    world_model=SimpleWorldModel()
)

# Observe external state
external_state = State(values={sensor_dof: 0.5})
internal_state = observer.observe(external_state)

print(f"Internal state: {internal_state}")
print(f"Latent value: {internal_state.get_value(latent_dof)}")
```

## Project Structure

```
ro-framework/
├── src/ro_framework/          # Main package
│   ├── core/                  # DoF, Value, State
│   ├── observer/              # Observer, Mapping
│   ├── correlation/           # Correlation measures (coming soon)
│   ├── consciousness/         # Consciousness evaluation (coming soon)
│   ├── multimodal/           # Multimodal integration (coming soon)
│   ├── uncertainty/          # Uncertainty quantification (coming soon)
│   ├── learning/             # Training protocols (coming soon)
│   └── integration/          # PyTorch/JAX integration (coming soon)
│
├── tests/                     # Comprehensive test suite
├── examples/                  # Example implementations (coming soon)
├── notebooks/                 # Jupyter tutorials (coming soon)
└── docs/                      # Documentation (coming soon)
```

## Development Status

**Current Version: 0.1.0 (Alpha)**

### Implemented ✓
- Core DoF types (Polar, Scalar, Categorical, Derived)
- Value and State abstractions
- Observer architecture with boundary, mapping, resolution, memory
- Comprehensive unit tests (>90% coverage for implemented modules)

### In Progress 🚧
- PyTorch integration (`ro_framework.integration.torch`)
- Correlation measures (`ro_framework.correlation`)
- Consciousness evaluation (`ro_framework.consciousness`)

### Planned 📋
- Multimodal integration
- Uncertainty quantification
- Training protocols
- Example implementations (CLIP-style, vision-language, etc.)
- Jupyter notebook tutorials
- Complete documentation on ReadTheDocs

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_dof.py

# Run with verbose output
pytest -v

# Generate HTML coverage report
pytest --cov-report=html
open htmlcov/index.html
```

## Documentation

Full documentation is coming soon. For now, see:
- [Theoretical Framework](ro_framework.md) - Complete theoretical foundation
- [Python Formalization](python_formalization.md) - Detailed implementation guide
- API documentation - Run `pydoc` on modules for docstrings

## Contributing

Contributions are welcome! This is an early-stage project. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide (enforced by `black` and `ruff`)
- Add type hints to all functions
- Write comprehensive docstrings (Google style)
- Achieve >90% test coverage for new code
- Update documentation for new features

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ro_framework,
  title = {Recursive Observer Framework: A Python Library for Conscious AI},
  author = {RO Framework Contributors},
  year = {2026},
  url = {https://github.com/IdentityOverflow/ROFramework}
}
```

## License

MIT License - see LICENSE file for details

## Philosophical Foundation

The RO Framework is built on several key philosophical insights:

1. **Block Universe**: All states exist timelessly; no temporal flow
2. **Relationalism**: States are relational (DoF-value pairs), not substantial
3. **Structural Realism**: Only structural relations are observable
4. **Observer-Dependence**: All observation is relative to observer structure
5. **Finite Resolution**: All observers have finite granularity
6. **Consciousness as Structure**: Consciousness is recursive self-mapping, not a special substance

See [ro_framework.md](ro_framework.md) for the complete theoretical foundation.

## Acknowledgments

This framework synthesizes ideas from:
- Block Universe theory (Hermann Weyl, Kurt Gödel)
- Structural realism (John Worrall, James Ladyman)
- Observer theory (Carlo Rovelli's relational quantum mechanics)
- Integrated Information Theory (Giulio Tononi)
- Predictive processing (Karl Friston)
- Multimodal deep learning (current AI research)

## Contact

- Issues: [GitHub Issues](https://github.com/IdentityOverflow/ROFramework/issues)

---

**Status**: 🚧 Active Development | **Version**: 0.1.0-alpha | **Python**: 3.9+
