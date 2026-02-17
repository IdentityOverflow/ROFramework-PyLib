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

## Current Status
**v0.2.0 complete** — All 5 phases done (206/206 tests passing, all examples run)
