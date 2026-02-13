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
- [ ] Delete `src/ro_framework/multimodal/` (5 files, ~2600 lines of scaffolding)
- [ ] Delete 4 multimodal test files
- [ ] Delete `examples/03_multimodal_observer.py`
- [ ] Delete 11 outdated markdown files (STATUS, PHASE*, IMPLEMENTATION_SUMMARY, etc.)
- [ ] Update `pyproject.toml` (remove pandas, torchvision, torchaudio)
- [ ] Verify remaining tests pass

### Phase 2: Knowledge Module + Observer Rewrite
- [ ] Create `knowledge/__init__.py` and `knowledge/assessment.py` (KnowledgeAssessment, compute_knowledge)
- [ ] Add ObservationPair and ObservationLog to observer module (replaces memory_buffer entirely)
- [ ] Rewrite `observer.py`: observe() stores pairs, memory methods use ObservationLog
- [ ] Implement `assess_knowledge()` and `know()` (was stub returning False)
- [ ] Fix `recursive_depth()` (structural chain, not dimension heuristic)
- [ ] Fix `estimate_uncertainty()` (quadrature addition, not additive)
- [ ] Fix `__repr__()` (remove is_conscious() call)
- [ ] Write `tests/unit/test_knowledge.py`
- [ ] Rewrite `tests/unit/test_observer.py` (memory_buffer → observation_log)
- [ ] Rewrite `tests/unit/test_memory_integration.py`
- [ ] All tests pass

### Phase 3: Fix Consciousness Evaluation
- [ ] `_evaluate_calibration()` — real ECE, not hardcoded 0.2
- [ ] `_evaluate_metacognition()` — behavioral test, not hasattr checks
- [ ] `_evaluate_limitation_awareness()` — easy/hard input split
- [ ] `_evaluate_architectural_similarity()` — structural comparison
- [ ] Rewrite `tests/unit/test_consciousness.py`
- [ ] Rewrite `tests/unit/test_consciousness_integration.py`
- [ ] All tests pass

### Phase 4: Integration Improvements
- [ ] Fix `TorchObserver.compute_saliency()` (real gradients, not zeros)
- [ ] Fix `TorchObserver.observe_batch()` (actual tensor batching)
- [ ] Create `integration/wrappers.py` (wrap_callable, wrap_torch_model, create_dofs_for_vector)
- [ ] Write `tests/unit/test_wrappers.py`
- [ ] Update `tests/unit/test_torch_integration.py`
- [ ] All tests pass

### Phase 5: Documentation and Polish
- [ ] Rewrite `README.md`
- [ ] Update examples 01, 02, 04, 05
- [ ] New `examples/03_knowledge_assessment.py`
- [ ] Update `__init__.py` exports
- [ ] Bump version to 0.2.0
- [ ] All tests + examples pass

## Current Status
**Active Phase**: 1 — Pruning dead code
