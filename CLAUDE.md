# ROFramework-PyLib — Development Guide

## What This Is
A Python library implementing the Recursive Observer Framework — a structural approach to observers, knowledge, and consciousness in AI. The library provides:
1. **DoF-typed data model** (Polar/Scalar/Categorical/Derived) with normalization, distance, vector conversion
2. **Observer abstraction** wrapping any model with boundary, mapping, resolution, memory
3. **Knowledge assessment** K(d_ext) = (ρ, ε, σ, C) — correlation, bias, noise, calibration
4. **Structural consciousness metrics** — recursive self-modeling depth and quality evaluation
5. **Correlation analysis** — observer-relative Pearson, MI, temporal, causal detection

## Environment
- Conda environment: `ro-framework`
- Python 3.9+
- Run tests: `conda activate ro-framework && pytest tests/ -v`

## Architecture
```
src/ro_framework/
├── core/              # DoF, Value, State (solid, don't touch)
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
└── integration/       # PyTorch bridge, model wrappers
    ├── torch.py
    └── wrappers.py
```

## Key Files Reference
- Theory: `ro_framework.md` (1519 lines, the full theoretical framework)
- Core types: `src/ro_framework/core/dof.py` (DoF hierarchy)
- Observer: `src/ro_framework/observer/observer.py` (O = B, M, R, Mem)
- Knowledge: `src/ro_framework/knowledge/assessment.py` (K = ρ, ε, σ, C)
- Consciousness: `src/ro_framework/consciousness/evaluation.py`
- Tests: `tests/unit/`

## Refactoring Plan — v0.2.0

### Phase 0: Workflow Tracking
- [x] Create this CLAUDE.md file

### Phase 1: Prune Dead Code
- [x] Delete `src/ro_framework/multimodal/` (5 files, ~2600 lines of scaffolding)
- [x] Delete 4 multimodal test files
- [x] Delete `examples/03_multimodal_observer.py`
- [x] Delete 11 outdated markdown files (STATUS, PHASE*, IMPLEMENTATION_SUMMARY, etc.)
- [x] Update `pyproject.toml` (remove pandas, torchvision, torchaudio)
- [x] Verify remaining tests pass

### Phase 2: Knowledge Module + Observer Rewrite
- [x] Create `knowledge/__init__.py` and `knowledge/assessment.py` (KnowledgeAssessment, compute_knowledge)
- [x] Add ObservationPair and ObservationLog to observer module (replaces memory_buffer entirely)
- [x] Rewrite `observer.py`: observe() stores pairs, memory methods use ObservationLog
- [x] Implement `assess_knowledge()` and `know()` (was stub returning False)
- [x] Fix `recursive_depth()` (structural chain, not dimension heuristic)
- [x] Fix `estimate_uncertainty()` (quadrature addition, not additive)
- [x] Fix `__repr__()` (remove is_conscious() call)
- [x] Write `tests/unit/test_knowledge.py`
- [x] Rewrite `tests/unit/test_observer.py` (memory_buffer → observation_log)
- [x] Rewrite `tests/unit/test_memory_integration.py`
- [x] All tests pass (173/173)

### Phase 3: Fix Consciousness Evaluation
- [x] `_evaluate_calibration()` — real ECE via binned uncertainty vs error comparison
- [x] `_evaluate_metacognition()` — behavioral: self-observation accuracy + depth + stability
- [x] `_evaluate_limitation_awareness()` — easy/hard input split, uncertainty ratio
- [x] `_evaluate_architectural_similarity()` — type match + dim ratio + shared attrs
- [x] Rewrite `tests/unit/test_consciousness.py` (36 tests, behavioral verification)
- [x] Rewrite `tests/unit/test_consciousness_integration.py` (13 tests, already updated)
- [x] All tests pass (190/190)

### Phase 4: Integration Improvements
- [x] Fix `TorchObserver.compute_saliency()` — real gradient-based attribution via input_tensor.grad
- [x] Fix `TorchObserver.observe_batch()` — single batched forward pass + observation log
- [x] Create `integration/wrappers.py` (wrap_callable, wrap_torch_model, create_dofs_for_vector)
- [x] Write `tests/unit/test_wrappers.py` (13 tests)
- [x] Update `tests/unit/test_torch_integration.py` (+3 tests: batch logging, batch≡sequential, saliency)
- [x] All tests pass (206/206)

### Phase 5: Documentation and Polish
- [x] Rewrite `README.md`
- [x] Update examples 01, 02, 04, 05
- [x] New `examples/03_knowledge_assessment.py`
- [x] Update `__init__.py` exports
- [x] Bump version to 0.2.0
- [x] All tests + examples pass (206/206)

### Phase 6: Short-term Roadmap (Bug Fixes and Usability)

- [x] Fix systematic error computation — sign-flip for negative correlation in `compute_knowledge()`
- [x] Add input/output DoF validation on `observe()` — raises `ValueError` on mismatch
- [x] Numpy-native batch path — `_CallableMapping.batch_call()` + `Observer.observe_batch()`
- [x] Observer serialization — `to_dict()`/`from_dict()` on DoF, State, ObservationLog, Observer; `save()`/`load()` via JSON
- [x] All tests pass (229/229)

### Phase 7: Tightening and Solidification

- [x] Fix CategoricalDoF indexing in `compute_uncertainty()` and `compute_saliency()` — use `dof.vector_dim` for multi-dimensional DoFs
- [x] Add `DoF.vector_dim` property (base returns 1, CategoricalDoF returns `len(categories)`)
- [x] Remove unused `typing-extensions` dependency from `pyproject.toml`
- [x] Document distributed representation / SAE challenge in README Known Limitations
- [x] Add `examples/06_serialization.py` (save/load workflow)
- [x] Add `examples/07_wrappers.py` (wrap_callable, wrap_torch_model, batch, knowledge)
- [x] Make knowledge type thresholds configurable — `KnowledgeAssessment.THRESHOLDS` class variable
- [x] Make consciousness score weights configurable — `ConsciousnessMetrics.DEFAULT_WEIGHTS` + optional `weights` param
- [x] Make `Observer.know()` calibration threshold configurable — `min_calibration` parameter
- [x] All tests pass (231/231)

### Phase 8: Training-Time Knowledge Dynamics & Resonance Detection

Motivated by: "On the Mechanism and Dynamics of Modular Addition" (He et al., 2025)
which shows learning is a resonance selection process — neurons compete, winners are
determined by initial spectral alignment, and grokking is a phase transition from
memorization to generalized (resonant) representation. Our K(d_ext) tuple can track
this process structurally.

#### 8a: Knowledge Trajectory Tracking (solid)

A `KnowledgeTracker` that records K(d_ext) = (ρ, ε, σ, C) over training epochs,
enabling temporal analysis of how knowledge forms, not just what it is at the end.

- [x] `knowledge/tracker.py` — `KnowledgeTracker` class
  - Wraps an Observer, records K(d_ext) at configurable intervals (every N steps/epochs)
  - Stores time series: `List[TrajectoryPoint]` per external DoF (TrajectoryPoint = epoch + KnowledgeAssessment)
  - Provides `trajectory(dof)`, `latest(dof)`, `step(epoch)`
- [x] Phase transition detection
  - `detect_grokking(dof)` — find the epoch where knowledge_type jumps from weak/false → strong
  - `detect_resonance(dof)` — find epochs where ρ is rising but σ is still high (feature locking in, pre-grokking)
  - `detect_forgetting(dof)` — find epochs where ρ drops (distribution shift, catastrophic forgetting)
- [x] Training loop integration
  - `tracker.step(epoch)` — call after each epoch, automatically runs assess_knowledge on all external DoFs
  - Works with any training loop (PyTorch, numpy, external)
  - Serializable (save/load trajectory with observer)
- [x] Tests + example: train an MLP on modular addition, show K trajectory through memorization → grokking
  - `examples/08_knowledge_tracker.py` — full training loop with grokking, GPU support
  - Smoke test in `tests/unit/test_tracker.py::TestTrackerTorchSmoke`
  - Uses sum-class averaging to remove within-pair noise (89% of variance); per-neuron R reaches 0.97
  - Auto-discovers model's dominant Fourier frequencies via DFT of sum-averaged activations
  - Grokking detected; feature-level knowledge ("strong") precedes behavioral generalization
  - Key finding: sum-averaging is task-specific (requires knowing grouping variable); see `memory/phase8a_grokking.md`
- [x] All tests pass (248/248)

#### 8b: Online Feature Discovery (experimental)

Hypothesis: instead of training a full SAE post-hoc, we can discover emerging features
during training by tracking which directions in activation space are being amplified.
A direction with growing, stable variance is a candidate monosemantic feature — a
resonance that's locking in.

- [x] `integration/activation_tracker.py` — `ActivationTracker` class
  - PyTorch forward hook that collects activations at a specified layer
  - Welford's online algorithm for mean and covariance (memory-efficient, O(D²))
  - PCA via `np.linalg.eigh` to find top-K principal directions
- [x] `discover_dofs(min_stability, min_stable_epochs, min_variance_fraction)` — extract candidate DoFs
  - Stable directions (|cos_sim| > threshold for N consecutive epochs) become PolarDoFs
  - Each discovered DoF has a projection vector (the eigenvector) for extracting values
  - `_ProjectionMapping` class for projecting activations onto discovered directions
- [x] Stability tracking — which directions persist vs. are transient
  - Greedy cosine similarity matching across epochs (handles rank swaps, sign ambiguity)
  - Direction identity histories track same direction across epochs
  - `detect_eigenvalue_spike()` for detecting feature lock-in
  - Optional readout alignment with fc2 weight matrix for task-relevance scoring
- [x] Tests: 26 tests in `tests/unit/test_activation_tracker.py`
  - Welford statistics, PCA lifecycle, direction matching, discovery, serialization
  - Torch smoke tests: hook collection, known-rank recovery, readout alignment
- [x] Example: `examples/09_activation_tracker.py` — grokking with honest comparison to Phase 8a
  - Stability clearly marks grokking transition (0.4 → 0.999 during generalization)
  - Eigenvalue spikes at memorization onset (epoch 250)
  - Top-variance PCA directions ≠ task-relevant Fourier features (readout alignment drops)
  - 89% of activation variance is within-sum-class noise; PCA captures embedding modes
  - Confirms: temporal dynamics (stability, spikes) are task-agnostic grokking detectors
- [x] All tests pass (274/274)

#### 8c: Knowledge-Guided Training (negative result)

Hypothesis: the K(d_ext) tuple can be used as a training signal, not just a metric.
If grokking is driven by the competition between loss minimization and regularization
(weight decay), then selectively increasing regularization pressure on features stuck
in the memorization phase should accelerate grokking.

- [x] `integration/training.py` — `KnowledgeRegularizer` class (combines KnowledgeLoss + AdaptiveRegularizer)
  - Reads K from KnowledgeTracker, classifies features as memorized/generalized/uncertain
  - Adjusts global weight decay: memorized → increase wd, generalized → decrease wd
  - Bias penalty: additive loss term for features with high |ε| (false knowledge)
  - `FeatureRegularization` dataclass for per-feature state
- [x] Tests: 15 tests in `tests/unit/test_training.py`
  - FeatureRegularization creation, KnowledgeRegularizer multipliers, integration with tracker, edge cases
- [x] Validation experiment: `examples/11_knowledge_guided_training.py`
  - **Result: K-guided training was 81% slower** (grokking at epoch 7250 vs baseline 4000)
  - Root cause: feature-behavioral lag — K reaches "strong" at epoch 500 while test acc is 0%
  - Regularizer misinterprets early feature formation as generalization → reduces wd → slows grokking
  - The "memorized" state (high ρ, low C) never occurs on clean sum-averaged data
  - Valid negative result: K(d_ext) is a feature-level metric, not a model-level one
- [x] All tests pass (291/291)

#### Phase 8 architecture

```text
src/ro_framework/
├── knowledge/
│   ├── assessment.py          # existing K(d_ext) computation
│   └── tracker.py             # NEW: KnowledgeTracker, trajectory, phase detection
├── integration/
│   ├── torch.py               # existing TorchObserver
│   ├── wrappers.py            # existing wrap_callable, wrap_torch_model
│   ├── activation_tracker.py  # ActivationTracker, discover_dofs (experimental)
│   ├── training.py            # KnowledgeRegularizer (experimental, negative result)
│   └── sae.py                 # NEW: SAEObserver, create_multilayer_sae_observers
```

### Phase 9: Interpretability Dashboard (Direction A)

Make the library work on real models. SAE integration is the bridge from toy to real.

- [x] SAE integration — `integration/sae.py`: `SAEObserver`, `create_multilayer_sae_observers`
  - Loads pre-trained SAEs (SAELens/TransformerLens), SAE features → ScalarDoFs
  - Bypasses world_model: constructs (label, SAE features) observation pairs directly
  - Supports mean/last/max aggregation across token sequence
  - Optional top-K feature filtering for large SAEs
- [x] GPT-2 proof of concept — `examples/12_sae_knowledge.py`
  - GPT-2 small + `gpt2-small-res-jb` SAE, 60 labeled texts (sentiment, code)
  - Code detection: strong knowledge (ρ=0.91-0.97) at all layers
  - Sentiment: weak knowledge (ρ=0.16-0.23) — honest result, needs more data
- [x] Feature-level knowledge profiles — `top_features_for()` returns sorted K tuples per SAE feature
- [x] Multi-layer comparison — knowledge across layers 0, 4, 8, 11 showing hierarchical decomposition
- [ ] SAE training tools — train SAEs on arbitrary model activations
- [x] Tests: 28 tests in `tests/unit/test_sae.py` (mocked model/SAE, no GPU needed)
- [x] All tests pass (319/319)

### Phase 10: Self-Aware Training (Direction B)

Use the framework as introspection machinery inside a training loop. Depends on Phase 9.

- [ ] Training-time feature introspection — periodic K(d_ext) on SAE features during fine-tuning
- [ ] Self-model integration — model maintains structural map of what it knows as it learns
- [ ] Multi-observer comparison — knowledge profiles across checkpoints, fine-tune variants, ensembles
- [ ] Multimodal bridge — assess alignment between visual and linguistic feature spaces

## Current Status
**v0.2.1-dev** — Phase 9 in progress (319/319 tests passing). SAE integration working on GPT-2.
