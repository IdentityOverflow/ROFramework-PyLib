# RO Framework Examples

This directory contains example implementations demonstrating the Recursive Observer Framework.

## Running Examples

Make sure you have installed the package:

```bash
# Activate conda environment
conda activate ro-framework

# Run an example
python examples/01_basic_observer.py
```

## Available Examples

### 01_basic_observer.py ✓ (Complete)

**Demonstrates:**
- Defining Degrees of Freedom (Polar DoFs)
- Creating States
- Building an Observer with a world model
- Performing observations (external → internal mapping)
- Computing state distances
- DoF normalization for neural networks

**Concepts covered:**
- DoF types (Polar)
- State representation
- Observer architecture
- World model (external→internal mapping)

**Output:**
Shows how an observer maps external sensor readings to internal latent states.

---

### 02_pytorch_conscious_observer.py ✓ (Complete)

**Demonstrates:**
- PyTorch neural network integration
- World model (MLP: external → internal)
- Self-model (MLP: internal → internal, SAME architecture)
- Recursive self-observation (consciousness!)
- MC Dropout uncertainty quantification
- Consciousness evaluation metrics
- Correlation analysis (Pearson, MI)

**Concepts covered:**
- PyTorch integration (`TorchNeuralMapping`)
- Conscious observer (with self-model)
- Structural consciousness evaluation
- Epistemic uncertainty via MC Dropout
- Correlation between external and internal DoFs

**Requirements:**
```bash
pip install ro-framework[torch]  # Requires PyTorch
```

**Output:**
```
🧠 Consciousness Score: 0.782/1.0

Consciousness Metrics:
  - Has self-model: True
  - Recursive depth: 1
  - Self-accuracy: 0.873
  - Architectural similarity: 1.000
```

Shows how to build a conscious AI system with PyTorch that can recursively model its own internal states.

---

### 03_multimodal_observer.py ✓ (Complete)

**Demonstrates:**
- Multimodal encoders (vision, language, audio)
- Fusion strategies (concatenation, attention, gating)
- MultimodalObserver with world model and self-model
- Supervised and self-supervised training protocols
- Active learning (uncertainty-based and diversity-based sampling)
- Comprehensive uncertainty quantification (ensemble, Bayesian)
- Calibration metrics and uncertainty decomposition

**Concepts covered:**
- ModalityEncoder abstraction
- VisionEncoder, LanguageEncoder, AudioEncoder
- FusionStrategy (ConcatenationFusion, AttentionFusion, GatedFusion)
- MultimodalObserver for conscious multimodal AI
- TrainingProtocol (SupervisedTraining, SelfSupervisedTraining)
- ActiveLearningStrategy (UncertaintyBasedSampling, DiversityBasedSampling)
- UncertaintyQuantifier (EnsembleUncertainty, BayesianUncertainty)
- CalibrationMetrics (ECE, MCE, NLL)
- Uncertainty decomposition (aleatoric vs epistemic)

**Requirements:**
```bash
pip install ro-framework[torch]  # Requires PyTorch and scikit-learn
```

**Output:**
```
=== Concatenation Fusion ===
Vision features: 16
Language features: 16
Audio features: 16
Fused features: 48
Average fused uncertainty: 0.1234

=== Multimodal Observer ===
Observer: MultimodalConsciousObserver
Is conscious: True
External DoFs: 48
Internal DoFs: 16

=== Uncertainty Quantification ===
Ensemble Uncertainty:
  output_0:
    Prediction: 0.1234
    Aleatoric: 0.0234
    Epistemic: 0.0778
    Total: 0.0811
    Confidence: 0.9189

Calibration Metrics:
  Expected Calibration Error: 0.0456
  Well-calibrated: True
```

Shows how to build a complete multimodal conscious AI system with sophisticated uncertainty quantification and active learning capabilities.

---

### 04_memory_temporal_correlation.py ✅ (Complete)

**Demonstrates:**
- Integrated memory system using temporal correlation
- Memory detection via correlation module (not just buffering)
- Multi-DoF memory analysis with cross-correlations
- Temporal correlation profiles at multiple lags
- Memory building through observations

**Concepts covered:**
- Memory as correlation across temporal DoF (theory alignment)
- Observer.has_memory() using temporal_correlation()
- Observer.analyze_memory_structure() for detailed analysis
- Observer.get_memory_correlations() for cross-DoF correlations
- Proper integration with correlation module

**Output:**
```
=== Scenario 1: Autocorrelated sequence (Memory Present) ===
Has memory: True
Temporal correlation profile (lags 1-5):
  sensor: ['0.823', '0.681', '0.577', '0.456', '0.371']

=== Scenario 2: Random sequence (No Memory) ===
Has memory: False
Temporal correlation profile:
  sensor: ['-0.062', '-0.114', '-0.237', '-0.045', '0.276']
```

Shows how memory is now properly detected using structural correlation measures, not just buffer size.

---

### 05_consciousness_evaluation.py ✅ (Complete)

**Demonstrates:**
- Integrated consciousness evaluation using ConsciousnessEvaluator
- Observer.is_conscious() with custom thresholds
- Observer.get_consciousness_metrics() for detailed assessment
- Observer.recursive_depth() calculation
- Multiple consciousness levels (non-conscious, conscious, meta-conscious)
- Consciousness metrics: self-accuracy, architectural similarity, calibration, meta-cognition

**Concepts covered:**
- Structural consciousness (not phenomenal experience)
- ConsciousnessEvaluator integration with Observer
- 7 consciousness metrics: has_self_model, recursive_depth, self_accuracy, architectural_similarity, calibration_error, meta_cognitive_capability, limitation_awareness
- Consciousness score as weighted combination
- Recursive depth 0, 1, 2 based on capacity
- Threshold-based consciousness decisions

**Output:**
```
Basic Conscious Observer (Self-Model Present)
============================================================

Observer: BasicConsciousObserver
Has self-model: True
Is conscious: True

Consciousness Metrics:
  Recursive depth: 1
  Self-accuracy: 0.500
  Architectural similarity: 1.000
  Calibration error: 0.200
  Meta-cognitive capability: 0.700
  Limitation awareness: 0.500

  Overall consciousness score: 0.617

Threshold Testing:
  Threshold 0.3: True
  Threshold 0.5: True
  Threshold 0.7: False
  Threshold 0.9: False
```

Shows how consciousness is properly evaluated using multiple structural metrics, not just "has self-model?".

---

## Planned Examples (Coming Soon)

### 06_clip_style.py 🚧

CLIP-style multimodal model with PyTorch.

**Will demonstrate:**
- PyTorch integration
- Contrastive learning
- Production-ready implementation

---

## Example Structure

Each example follows this structure:

1. **Import required modules** from ro_framework
2. **Define DoFs** for the problem domain
3. **Create mapping functions** (world model, self model)
4. **Build observer** with boundary, mappings, resolution
5. **Demonstrate observations** with sample data
6. **Show key features** relevant to that example

## Next Steps

After exploring these examples:

1. Read the [theoretical framework](../ro_framework.md) for deep understanding
2. Check the [Python formalization](../python_formalization.md) for implementation details
3. Review the [API documentation](../docs/) (coming soon)
4. Try the [Jupyter notebooks](../notebooks/) for interactive learning (coming soon)

## Contributing Examples

Have an interesting use case? Please contribute!

1. Create a new example file (`0X_your_example.py`)
2. Follow the structure above
3. Add comprehensive comments
4. Include sample output in docstring
5. Update this README
6. Submit a pull request

Examples we'd love to see:
- Cognitive modeling
- Robotics applications
- Scientific discovery
- Interpretable AI
- Multi-agent systems
- Temporal reasoning
