# Seed Architecture Experiments

Validation experiments for the Seed: a self-organizing neural architecture where five local rules applied by oscillatory nodes produce criticality, frequency bands, cross-scale coupling, and adaptive growth as emergent consequences.

See [docs/seed_architecture.md](../../docs/seed_architecture.md) for the full specification.

## Architecture Summary

Each node is an oscillatory unit with:
- Activation: `tanh(Σ w_ij * neighbor_j + external_drive + A*sin(phase) + noise)`
- Branching ratio σ tracking (EMA): how many neighbors fire when this node fires
- Hebbian weight adjustment: `Δw = lr * act_i * act_j * (1 - σ)` — the single self-regulation rule

The **sparse regime** is key: threshold (0.5) >> drive (0.2) + noise (0.05), so nodes need coupling input to fire. This makes σ a meaningful causal measure — activations propagate through the coupling structure, not from common drive.

## Experiments

### 01 — Criticality Validation

**Question**: Does Rule 2a (Hebbian adjustment governed by σ) produce self-organized criticality?

**Method**: 64 nodes, ring-lattice, 10k steps. Two conditions:
- Independent random drives: each node gets sparse random pulses (5% probability per step)
- Correlated signal drives: nodes near signal frequencies get common drive

**Results** (64 nodes, 10k steps):

| Metric | Independent | Correlated |
|--------|------------|------------|
| σ (target=1.0) | 1.09 | 1.00 |
| Active fraction | ~15% | ~20% |
| Cascade power law | PASS | PASS |
| Avalanche power law | PASS | PASS |

**Key finding**: σ converges toward 1.0 under both conditions. The critical fix was **σ decay for inactive nodes** — when a node isn't firing, its σ decays toward 0 (rather than staying frozen at a stale value). This breaks the chicken-and-egg problem where inactive nodes had artificially high σ, preventing weight growth.

### 02 — Computational Capacity

**Question**: Is the critical network computationally useful? Can it serve as a reservoir?

**Tests**:
1. **Echo state property**: Same network, different input signals → diverging internal states (confirmed)
2. **Fading memory**: Inject a pulse, measure how long the trace persists (10-20 steps at 64 nodes)
3. **Multi-class frequency discrimination**: Log-spaced frequencies, one-vs-rest ridge regression readout

**Results** (multi-class classification, test accuracy):

| Classes | Chance | 64 nodes | 256 nodes |
|---------|--------|----------|-----------|
| 2       | 50%    | 70%      | 75%       |
| 4       | 25%    | 68%      | 75%       |
| 6       | 17%    | 72%      | 52%       |
| 8       | 13%    | 71%      | 38%       |

**Key findings**:
- 64 nodes generalizes consistently at 70%+ across 2-8 classes (5.5x above chance at 8 classes)
- 256 nodes overfits at high class count (train=100%, test drops) — a readout problem, not a reservoir problem. More readout dimensions than training samples.
- The self-organized reservoir discriminates between frequencies with no trained internal weights

### 03 — Self-Scaling

**Question**: Do Rules 4 (recruit) and 5 (release) produce adaptive network sizing?

**Method**: 4 phases with 16 initial nodes:
1. Single frequency (0.1 Hz) — warmup
2. Add second frequency (0.1 + 0.4 Hz) — should trigger growth
3. Silence — should NOT trigger growth
4. Back to single frequency — test adaptation

**Results**:

| Phase | Nodes | σ | Behavior |
|-------|-------|---|----------|
| 1: Single freq | 16 → 16 | 1.15 | Stable, near-critical |
| 2: Two freqs | 16 → 25 | 1.25 → 1.2 | Grew when supercritical, then stabilized |
| 3: Silence | 25 → 25 | → 0 | No growth (activity gate) |
| 4: Single freq | 25 → 26 | ~1.0 | Stable, one recruit from brief supercritical spike |

**Key findings**:
- Recruitment works: grows when supercritical (overloaded), stops when σ ≤ 1.2
- Activity gating prevents runaway growth during silence
- Release doesn't fire — extra nodes get repurposed via frequency entrainment rather than pruned (all coupled nodes maintain some activity through propagation)
- Frequency entrainment confirmed: after Phase 2, freq range narrows as nodes converge toward the active signal

**Implementation notes**:
- Recruitment trigger: `mean(σ) > 1.2` AND `>20% nodes active` (activity gate prevents growth during silence)
- Release trigger: node active <1% of recent steps while network is active (never triggers because coupling keeps all nodes somewhat alive)
- Mutual information (MI) based triggers (original design) didn't work — MI between raw external input and node activations is a poor proxy for "signal present but underrepresented"

## Key Design Decisions

### Sparse regime
The most important design choice. With threshold >> drive + noise, nodes are mostly silent and need coupling input to fire. This makes the branching ratio meaningful and Rule 2a sufficient for self-regulation.

### σ decay for inactive nodes
When a node isn't firing, its σ decays: `σ *= (1 - α)`. Without this, inactive nodes keep stale σ values from early transients, preventing the subcritical weight growth that would eventually activate them. This single fix resolved all convergence issues.

### σ-based recruitment (not mutual information based)

The original spec used mutual information between external input and internal activations to trigger recruitment. In practice, MI estimation between raw inputs and node activations was unreliable — the sensor transforms inputs into drives, so measuring MI on raw inputs measures the wrong thing. The supercritical σ trigger is simpler and more direct: if the network is overloaded (σ > 1), it needs more capacity.

### Release as adaptive reuse

Nodes recruited for one signal don't go silent when that signal stops — coupling keeps them somewhat active, and frequency entrainment drifts them toward whatever signal IS present. This is arguably more biologically realistic than pruning: neurons get reassigned, not discarded.

### Bandwidth-limited entrainment

The original entrainment rule drifted each node toward the coupling-weighted mean of ALL neighbor frequencies. This caused all nodes to converge toward a single global mean — no bands formed. The fix: weight entrainment by resonance proximity (~1 octave bandwidth in log-frequency space). A node is only pulled by neighbors close to its own frequency. This creates local basins of attraction, allowing distinct bands to form.

### 04 — Frequency Band Formation & Cross-Scale Coupling

**Question**: Does the Seed produce distinct frequency bands when exposed to multi-timescale input? Do slow nodes model fast-node behavior?

**Method**: 64 nodes, 20k steps. Drive signal: slow envelope (0.05 Hz) amplitude-modulating a fast carrier (0.3 Hz). This presents two distinct timescales.

**Results** (20k steps, bandwidth-limited entrainment):

Frequency distribution after entrainment:

| Region | Freq range | Nodes | Signal |
| --- | --- | --- | --- |
| Low band | 0.03-0.08 Hz | 39 | Near slow (0.05 Hz) |
| Gap | 0.08-0.17 Hz | 0 | Empty |
| High band | 0.17-0.51 Hz | 25 | Near fast (0.30 Hz) |

Cross-scale coupling metrics:

| Metric | Value | Interpretation |
| --- | --- | --- |
| Phase-amplitude coupling (MI) | 0.064 | Moderate — slow phase modulates fast amplitude |
| Slow-fast variance correlation | -0.17 | Anticorrelated (correct for AM signal) |
| Direct band correlation | 0.039 | Near zero — bands operate independently |

**Key findings**:

- Frequency bands form spontaneously around environmental timescales (bimodal clustering with clear gap)
- Bandwidth-limited entrainment is essential — without it, all nodes collapse to a global mean
- Phase-amplitude coupling confirmed: slow-band phase modulates fast-band amplitude (PAC = 0.064)
- The 39:25 node ratio favoring the slow band makes sense: the slow envelope is always present while the fast carrier is amplitude-modulated (intermittent)
- Initial frequency distribution was uniform in [0.01, 1.0] — the clustering is entirely emergent

## Running

```bash
# Criticality validation (fast, ~10s)
python experiments/seed/01_criticality_validation.py --steps 10000 --nodes 64

# Computational capacity (includes multi-class frequency discrimination)
python experiments/seed/02_computational_capacity.py --steps 5000 --nodes 64

# Self-scaling (4 phases)
python experiments/seed/03_self_scaling.py --nodes 16

# Frequency bands and cross-scale coupling (~45s)
python experiments/seed/04_frequency_bands.py --steps 20000 --nodes 64
```

## Status

Core mechanism validated (criticality, reservoir capacity, adaptive growth). Frequency band formation and cross-scale coupling confirmed with bandwidth-limited entrainment. Next steps: coupling to the embodied environment as the sensor/actuator interface.
