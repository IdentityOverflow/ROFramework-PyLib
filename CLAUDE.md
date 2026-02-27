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

- [ ] `knowledge/tracker.py` — `KnowledgeTracker` class
  - Wraps an Observer, records K(d_ext) at configurable intervals (every N steps/epochs)
  - Stores time series: `List[Tuple[int, KnowledgeAssessment]]` per external DoF
  - Provides `trajectory(dof) -> DataFrame-like` of (step, ρ, ε, σ, C, type) over time
- [ ] Phase transition detection
  - `detect_grokking(dof)` — find the epoch where knowledge_type jumps from weak/false → strong
  - `detect_resonance(dof)` — find epochs where ρ is rising but σ is still high (feature locking in, pre-grokking)
  - `detect_forgetting(dof)` — find epochs where ρ drops (distribution shift, catastrophic forgetting)
- [ ] Training loop integration
  - `tracker.step(epoch)` — call after each epoch, automatically runs assess_knowledge on all external DoFs
  - Works with any training loop (PyTorch, numpy, external)
  - Serializable (save/load trajectory with observer)
- [ ] Tests + example: train an MLP on modular addition, show K trajectory through memorization → grokking
- [ ] All tests pass

#### 8b: Online Feature Discovery (experimental)

Hypothesis: instead of training a full SAE post-hoc, we can discover emerging features
during training by tracking which directions in activation space are being amplified.
A direction with growing, stable variance is a candidate monosemantic feature — a
resonance that's locking in.

- [ ] `integration/activation_tracker.py` — `ActivationTracker` class
  - PyTorch forward hook that collects activations at a specified layer
  - Maintains running mean and covariance (incremental, memory-efficient)
  - Incremental PCA / SVD to find top-K principal directions
- [ ] `discover_dofs(threshold)` — extract candidate DoFs from tracked activations
  - Directions with eigenvalue growth above threshold become PolarDoFs
  - Each discovered DoF has a projection vector (the eigenvector) for extracting values
  - Returns DoFs + a projection mapping that composes with the model as a world_model
- [ ] Stability tracking — which directions persist vs. are transient
  - Compare principal directions across epochs (subspace angle / cosine similarity)
  - A direction that's been stable for N epochs is a "locked in" feature
  - Transient directions are noise or memorization artifacts
- [ ] Tests: train MLP, verify discovered directions correlate with known Fourier features
- [ ] All tests pass

#### 8c: Knowledge-Guided Training (experimental)

Hypothesis: the K(d_ext) tuple can be used as a training signal, not just a metric.
If grokking is driven by the competition between loss minimization and regularization
(weight decay), then selectively increasing regularization pressure on features stuck
in the memorization phase should accelerate grokking.

- [ ] `KnowledgeLoss` — differentiable loss terms derived from knowledge assessment
  - Calibration loss: penalize low C (observer doesn't know what it doesn't know)
  - Correlation reward: encourage high ρ for target DoFs
  - Bias penalty: penalize high |ε| (systematic distortion)
- [ ] `AdaptiveRegularizer` — adjusts weight decay per-feature based on K dynamics
  - Features with high ρ but low C (memorized, not generalized) get increased regularization
  - Features with high ρ and high C (genuinely learned) get reduced regularization
  - Hypothesis: this steers the competitive dynamics to favor resonant solutions
- [ ] Validation experiment: modular addition with and without knowledge-guided training
  - Measure: epochs to grokking, final knowledge quality, feature emergence speed
  - Compare: standard training vs. K-guided regularization
- [ ] Tests + example
- [ ] All tests pass

#### Phase 8 architecture

```text
src/ro_framework/
├── knowledge/
│   ├── assessment.py          # existing K(d_ext) computation
│   └── tracker.py             # NEW: KnowledgeTracker, trajectory, phase detection
├── integration/
│   ├── torch.py               # existing TorchObserver
│   ├── wrappers.py            # existing wrap_callable, wrap_torch_model
│   ├── activation_tracker.py  # NEW: ActivationTracker, discover_dofs (experimental)
│   └── training.py            # NEW: KnowledgeLoss, AdaptiveRegularizer (experimental)
```

#### Execution order

1. **8a first** — KnowledgeTracker is pure engineering on top of existing assess_knowledge().
   No new dependencies, no hypotheses. Proves the temporal analysis works.
2. **8b second** — ActivationTracker is experimental but grounded in well-known PCA/SVD.
   Validates whether online feature discovery produces meaningful DoFs.
3. **8c last** — KnowledgeLoss and AdaptiveRegularizer are the most speculative.
   Depends on 8a working well. May need iteration on the loss formulation.

## Current Status
**v0.2.1-dev** — Phase 7 complete (231/231 tests passing)
