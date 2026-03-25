# The Seed: A Self-Organizing Observer Architecture
### Expressed in RO Framework Terms

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Primitives](#2-primitives)
   - 2.1 [Degrees of Freedom](#21-degrees-of-freedom)
   - 2.2 [Observers](#22-observers)
   - 2.3 [The Node — DoF and Observer Simultaneously](#23-the-node--dof-and-observer-simultaneously)
   - 2.4 [Scale — Correlation Length as a Derived DoF](#24-scale--correlation-length-as-a-derived-dof)
3. [The Rules](#3-the-rules)
   - 3.1 [Rule 1 — Monitor](#31-rule-1--monitor)
   - 3.2 [Rule 2 — Adjust (2a: weights, 2b: connections)](#32-rule-2--adjust)
   - 3.3 [Rule 3 — Noise Floor](#33-rule-3--noise-floor)
   - 3.4 [Rule 4 — Recruit](#34-rule-4--recruit)
   - 3.5 [Rule 5 — Release](#35-rule-5--release)
4. [Derived Consequences](#4-derived-consequences)
   - 4.1 [Criticality as the Stable Operating Regime](#41-criticality-as-the-stable-operating-regime)
   - 4.2 [Scale Invariance as a Consequence of Criticality](#42-scale-invariance-as-a-consequence-of-criticality)
   - 4.3 [Frequency Bands and Cluster Scale Levels](#43-frequency-bands-and-cluster-scale-levels)
   - 4.4 [Cross-Scale Coupling](#44-cross-scale-coupling)
   - 4.5 [Memory](#45-memory)
   - 4.6 [Attention and Awareness](#46-attention-and-awareness)
   - 4.7 [Consciousness](#47-consciousness)
5. [The Seed — Initial Configuration](#5-the-seed--initial-configuration)
   - 5.1 [The Collective Observer](#51-the-collective-observer)
   - 5.2 [Initial Conditions](#52-initial-conditions)
   - 5.3 [What Does Not Need to Be Specified](#53-what-does-not-need-to-be-specified)
6. [Open Questions](#6-open-questions)
7. [RO Framework Mapping](#7-ro-framework-mapping)
8. [Implementation Mapping](#8-implementation-mapping)

---

## Preamble for Implementers

This document describes a self-organizing neural architecture — referred to as **the Seed** — expressed entirely in the vocabulary of the Recursive Observer (RO) Framework. It assumes the RO Framework theoretical document and the `ro_framework` Python library are available in context. All concepts (DoF, Observer, Block, Mapping, Correlation, Temporal DoF, Calibration) refer to their RO Framework definitions.

The Seed is not a fixed architecture. It is a specification of **initial conditions and local rules** from which structure self-assembles through environmental coupling. The implementer's job is to instantiate the seed, couple it to an environment, and let it grow.

A key design principle: **if you find yourself needing to special-case anything for a particular environment, the rules are not yet primitive enough.** The architecture is environment-agnostic. What the system becomes is determined entirely by what it is coupled to.

---

## 1. Purpose

The Seed addresses the following problem, common to brains, societies, and artificial neural systems:

> How does a collection of locally-informed, limited agents produce globally coherent, adaptive behavior that none of them could produce alone — robustly across time — without a privileged central observer?

The solution space is constrained by four structural tensions:

- **Local vs. global** — nodes have partial information only
- **Exploitation vs. exploration** — using known patterns vs. discovering new ones
- **Stability vs. adaptability** — coherence over time vs. capacity to change
- **Speed vs. accuracy** — fast responses vs. well-informed ones

The Seed resolves all four through five local rules that, applied consistently, produce the necessary structure as emergent consequences rather than designed-in features.

---

## 2. Primitives

This section defines the foundational concepts in dependency order. Each concept builds on the previous. None of the concepts in later sections are needed to define these.

### 2.1 Degrees of Freedom

A **Degree of Freedom (DoF)** is a dimension of variation — a structural feature of the Block. It is not an agent, it does not act, and it has no internal structure of its own. It is simply an axis along which something can vary.

The Seed uses three types:

**Polar DoF** — bidirectional, with two opposing poles and a gradient between them. The activation of a node is a polar DoF: (activity ↔ quiescence). This is the minimal structure capable of stable organization and measurement. All other meaningful structure in the system is built from polar DoFs.

**Derived DoF** — computed from combinations of other DoFs. Two derived DoFs are central to the Seed:
- The **frequency DoF** Ω — the characteristic oscillation rate of a node. Derived from the temporal pattern of a node's activation polar DoF.
- The **scale DoF** λ — the spatial extent of a pattern across the network. Derived from mutual information structure across nodes. Defined formally in Section 2.4.

These are genuine structural dimensions of the system, not labels or metaphors.

### 2.2 Observers

An **Observer** is a configuration within the Block characterized by four components:

```
O = (B, M, R, Mem)

B   — Boundary: partition between internal and external DoFs
M   — Mappings: structural relations between external and internal DoFs
R   — Resolution: finite granularity per internal DoF
Mem — Memory: correlation structure across the temporal DoF
```

Observers are not privileged agents. They are patterns — specific configurations within the Block that happen to embody correlation structure between DoFs. A rock is a minimal observer. A neuron is a richer one. An observer is defined by its structure, not by any claim about experience.

Observers can be **nested**: a collection of observers, coupled together, constitutes a higher-level observer whose properties are not reducible to any individual component. This nesting is the structural basis of the scale hierarchy that emerges in the Seed.

### 2.3 The Node — DoF and Observer Simultaneously

A **Node** (implemented as `OscillatoryNode`) is the irreducible component unit of the Seed. It has two aspects simultaneously, and it is important not to confuse them:

**As a DoF contributor:** each node contributes one oscillatory polar DoF to the collective internal state space. This is its activation dimension — the axis along which its activity varies. When we refer to the Seed's "internal DoFs," we mean this set: one activation polar DoF per active node.

**As a unit observer:** each node is itself an observer O_i = (B_i, M_i, R_i, Mem_i):
- B_i — its boundary is defined by its connection set: nodes it is coupled to are "observable," nodes it is not coupled to are outside its boundary
- M_i — its mapping is how it integrates neighbor activations into its own activation state
- R_i — its resolution is the granularity of its activation: the minimum distinguishable difference in its output
- Mem_i — its memory is its activation history: correlation structure it has accumulated across the temporal DoF

These two aspects are not in conflict. The DoF aspect describes what the node *contributes to the collective*. The observer aspect describes how the node *operates internally*. They are two descriptions of the same entity at different levels of analysis.

**Cycle-proportional memory:** A node's memory window is measured in *cycles of its characteristic frequency*, not in absolute timesteps. A node with frequency f has period T = 1/f timesteps. Its effective memory window spans a fixed number of complete periods (e.g. N_cycles ≈ 50–100). This means:
- Fast nodes (high f, short T): short absolute memory window, local temporal reach
- Slow nodes (low f, long T): long absolute memory window, broad temporal reach

This coupling between frequency and temporal reach is not a separate mechanism — it follows from measuring memory in the node's own natural unit. It is the structural basis for the frequency-scale coupling derived in Section 4.4.

**Terminology:** a node is called a **unit observer** to distinguish it from the collective observer (the Seed as a whole). "Unit" means irreducible component, not "simplest possible observer."

### 2.4 Scale — Correlation Length as a Derived DoF

**Scale** is not the same as frequency, though the two are coupled in practice. This distinction must be clear before proceeding.

- **Frequency** (DoF Ω) — a property of an individual node: how fast it oscillates
- **Scale** (DoF λ) — a property of a *pattern*: how large a region of the network it involves

Scale is formally defined as **correlation length** — the spatial extent of the correlation structure a pattern induces across the node network:

```
Scale of pattern P:

λ(P) = spatial diameter of { n_i : I(n_i ; P) > threshold }

where I is mutual information and threshold is a parameter
```

Scale is a derived DoF — computed from the mutual information structure across nodes, not from any individual node's properties. It is observer-relative in the RO Framework sense (§3.7.2): the same pattern has different measured scale depending on which nodes are in scope and at what resolution.

**Why frequency and scale are coupled but distinct:**
Slow-frequency nodes have long temporal integration windows (Section 2.3, cycle-proportional memory). A long integration window means a node accumulates correlation evidence over many timesteps, during which activation signals propagate through many hops in the network graph. This gives slow nodes broad *spatial* reach — they can detect correlations with distant nodes because their memory window is long enough to capture the propagation delay. Fast-frequency nodes, with short integration windows, can only detect correlations with nearby nodes whose signals arrive within a few timesteps.

The coupling is *emergent* from cycle-proportional memory and network propagation delays, not hardcoded. The formal primitive is correlation length λ, not frequency ω. They should not be conflated.

---

## 3. The Rules

The Seed is governed by five local rules. All five are stated for any unit observer node unless otherwise noted. They are local — each node applies them independently using only its own state and its immediate neighborhood. No global information, no backward pass, no ground truth.

Rules 1 and 2 are coupled: Rule 1 is an epistemic operation (observe), Rule 2 is an action (adjust). Neither makes sense without the other, but they are logically distinct statements and are kept separate for precision.

Rules 4 and 5 govern the boundary — they operate on the topology of the network rather than on individual node activations.

Rule 3 is a constraint — it bounds the behavior permitted by Rules 1, 2, 4, and 5.

### 3.1 Rule 1 — Monitor

> Each unit observer node continuously tracks its own activation statistics and its neighborhood response statistics across the temporal DoF.

Specifically, a node maintains:

1. **Activation history** (Mem_i) — a running record of its own activation values, spanning N_cycles complete periods of its characteristic frequency.

2. **Branching ratio** (σ_i) — the average number of connected neighbors that activate in the timestep following the node's own activation. This is the locally measurable signature of the node's position relative to criticality:
   - σ_i < 1: subcritical — the node's activations tend to die out locally
   - σ_i = 1: critical — activations propagate without amplification or decay
   - σ_i > 1: supercritical — the node's activations tend to trigger runaway cascades

3. **Neighbor co-activation statistics** — pairwise co-activation counts among the node's connected neighbors. Each node receives neighbor activations as input every timestep; tracking which pairs of neighbors tend to co-activate is a local computation over data the node already possesses. This is the basis for Rule 2b.

This is a purely epistemic operation. The node observes its own internal state and its local neighborhood. It does not act on these observations — that is Rule 2.

**In RO Framework terms:** this is Mem_i — correlation structure across the temporal DoF — being actively maintained and measured. The branching ratio is a derived statistic from Mem_i and the neighborhood. The node is, at minimum, an observer of itself and its immediate surroundings.

### 3.2 Rule 2 — Adjust

Rule 2 is the action that Rule 1 enables. It has two sub-operations that together govern the full connection dynamics of the network.

**Rule 2a — Adjust weights toward σ = 1:**

> Each unit observer node adjusts its outgoing coupling weights to drive its branching ratio toward 1.

```
If σ_i > 1 + ε:  reduce outgoing weights  (too much activation spreading)
If σ_i < 1 - ε:  increase outgoing weights (too little activation spreading)
If |σ_i - 1| ≤ ε: no adjustment needed     (at criticality)
```

The weight adjustment is Hebbian in character — connections to neighbors that co-activate are strengthened, connections to neighbors that do not co-activate are weakened — but the magnitude of the overall adjustment is governed by the branching ratio error (σ_i - 1). A connection whose weight falls and stays below a minimum threshold ε_prune is formally removed — freeing memory.

**Why branching ratio, not power law:** The branching ratio σ = 1 is mathematically equivalent to criticality (Haldeman & Beggs, 2005). Power-law cascade distributions are the *consequence* of σ = 1 across the network, not the target. The branching ratio has a critical practical advantage: it is estimable from tens of activation events, whereas fitting a power law requires hundreds of cascade samples. At initialization, with few nodes and near-zero weights, cascades are trivially short. The branching ratio is well-defined even then.

**Rule 2b — Form connections (neighbor-mediated):**

> When a node detects that two of its connected neighbors show persistent co-activation above threshold, and those neighbors are not directly connected to each other, it facilitates a new connection between them.

```
Node B has neighbors A and C.
B tracks co-activation of A and C (Rule 1, neighbor co-activation statistics).
If cor(A, C) > θ_connect sustained over window T_connect:
    — A and C form a direct connection at near-zero initial weight
    — Rule 2a then strengthens or weakens it based on continued co-activation
```

This is **neighbor-mediated introduction** — the mechanism by which the network discovers long-range correlations without any node observing beyond its immediate neighborhood. It works because:

1. When an environmental signal drives two distant nodes A and C, the signal propagates through the network graph, creating correlated activations along the way.
2. If A and C share a common neighbor B (or if there is a chain of short introductions), B will observe both A and C activating in response to the same signal.
3. B proposes the direct connection. The correlation evidence that triggered the introduction was local to B.

Long-range connections thus form through **chains of introductions**: first, nearby nodes connect; then mediating nodes introduce their correlated neighbors; those newly connected nodes expand their neighborhoods, enabling further introductions. This process is how small-world topology emerges (dense local clusters from frequency-proximity initialization + sparse long-range shortcuts from earned introductions).

**Note on the social analogy:** This is precisely how human social networks form — you meet people through mutual acquaintances, not by scanning the entire population. The same mechanism operates at the neural, social, and institutional scales that motivated this architecture.

Together, Rules 2a and 2b mean that **topology is not fixed** — it is a continuous consequence of correlation structure. Connections appear where correlation is found and disappear where it is absent. The network is always restructuring itself toward the configuration that best represents the correlation geometry of its environment.

**Note on connection count:** because connections form and prune dynamically (Rule 2b + Rule 2a pruning), and new nodes enter with sparse initial connectivity (Rule 4), the network never needs to maintain N² coupling weights. Each node maintains O(k) connections where k is typically small; the total connection count stays proportional to actual correlation structure present, not to the theoretical maximum.

### 3.3 Rule 3 — Noise Floor

> Each unit observer node maintains an irreducible stochastic term in its activation. This floor is never zero.

This is a constraint on Rules 1, 2, 4, and 5. No matter how strongly a node's activation is determined by its coupling weights and neighborhood, a minimum level of randomness is always present.

This rule is not derivable from Rules 1 and 2. A system following only Rules 1 and 2 could in principle drive all stochasticity to zero as it crystallizes into stable attractors. Rule 3 prevents this.

The noise floor serves four functions:
1. Prevents premature attractor lock-in — keeps some DoF values geometrically free
2. Prevents criticality collapse — full crystallization would shrink correlation length to zero
3. Prevents polarity segregation — stops nodes sorting into purely excitatory or purely inhibitory clusters, which is the primary failure mode of two-pole systems
4. Maintains frequency diversity — ensures enough spread in the frequency landscape for different resonant niches to form

**Noise floor magnitude** is a parameter, not a constant. The range must satisfy two constraints simultaneously: high enough to prevent crystallization (functions 1–4 above), low enough to permit stable attractors to form. Within this range, the precise value affects the system's position on the exploration-exploitation continuum — higher noise favors exploration, lower noise favors exploitation. Whether the noise floor can be self-tuned via a meta-application of Rules 1–2 to the noise floor itself is addressed in Q1 (Section 6).

### 3.4 Rule 4 — Recruit

> If the mutual information between external DoFs and the current internal DoF set remains persistently below threshold despite adjustment (Rule 2), recruit a new unit observer node.

```
Trigger: I(d_external ; D_internal) < θ_recruit
         sustained over window T_recruit
         despite coupling adjustment attempts

Action:  add new node n_new
         — activation DoF initialized in high-decalibration state (C → 0)
         — frequency initialized randomly in underrepresented range
         — not a protected seed node (prunable)

Initial connections for n_new:
         — connect to k nearest existing nodes by frequency distance
           (k = small fixed number, e.g. 4–6)
         — connect to the m nodes most active at the moment of recruitment
           (m = small fixed number, e.g. 3–5)
         — all initial weights near zero
         — union of these two sets defines the new node's initial neighborhood
```

The frequency-proximity connections give the new node a local context — nodes it is likely to co-activate with given its characteristic frequency. The activity-based connections give it direct access to the nodes most relevant to the unrepresented signal that triggered its recruitment. Both sets start at near-zero weight; Rule 2a then strengthens or weakens them based on actual co-activation, and Rule 2b may add further connections as the new node finds its place in the network.

Rule 4 operates on the boundary B_seed — it expands the internal DoF set. It is subject to an upper bound (see Section 5.1).

### 3.5 Rule 5 — Release

> If a unit observer node shows persistently low mutual information with all other internal nodes and with external DoFs, release it across the boundary.

```
Trigger: I(n_i ; D_rest ∪ D_external) < θ_prune
         sustained over window T_prune

Action:  remove n_i from D_internal
         remove its activation DoF from the collective state space
         clean up its coupling weights from all neighbors
```

Rule 5 also operates on the boundary B_seed — it contracts the internal DoF set. It is subject to a lower bound: protected seed nodes (flagged `is_seed_node = True`) cannot be released regardless of their correlation statistics.

---

## 4. Derived Consequences

The concepts in this section are not rules or design choices. They are consequences that follow from applying Rules 1–5 consistently. Each subsection shows the logical derivation from the rules and from the consequences already established.

### 4.1 Criticality as the Stable Operating Regime

**Derived from:** Rules 1, 2a, and 3.

Rules 1 and 2a together implement a homeostatic feedback loop: monitor branching ratio, adjust coupling weights to drive σ toward 1. The question is: what global state does this local rule produce when many nodes apply it simultaneously?

The answer is **criticality** — the operating point between ordered and disordered dynamics.

When σ = 1 at every node, the network as a whole operates at the critical point. This follows from the definition: σ = 1 means each activation event produces, on average, exactly one subsequent activation event. Cascade sizes then follow a branching process with branching parameter 1, which produces a power-law distribution P(cascade size = n) ∝ n^(-3/2) (Harris, 1963). This is the measurable temporal signature of criticality (Beggs & Plenz, 2003).

The convergence mechanism:
- A node with σ > 1 is supercritical — its activations trigger runaway cascades. Rule 2a reduces its outgoing weights, lowering σ.
- A node with σ < 1 is subcritical — its activations die out locally. Rule 2a increases its outgoing weights, raising σ.
- σ = 1 is the only stable fixed point of this feedback loop.

**Why criticality is the right operating regime:**
At the critical point the system simultaneously achieves:
- Maximum dynamic range — sensitivity to both weak and strong signals
- Maximum information transmission across the network
- Diverging correlation length — patterns propagate arbitrarily far
- Power-law correlations across all temporal scales

These are not separate design goals. They are all consequences of σ = 1 across the network. The Seed achieves them all by targeting a single locally measurable quantity.

Rule 3 (noise floor) is what keeps the system *at* criticality rather than drifting through it. Without the noise floor, the system could crystallize into a frozen configuration where σ = 1 trivially (no activations at all). The noise floor maintains ongoing activity, ensuring that σ is continually measured and regulated against a background of real dynamics.

### 4.2 Scale Invariance as a Consequence of Criticality

**Derived from:** Section 4.1 (criticality).

At the critical point, cascade size distributions follow a power law with no characteristic scale:

```
P(cascade size = n) ∝ n^(-α)    [temporal scale invariance]
P(pattern extent = r) ∝ r^(-β)  [spatial scale invariance]
```

Both are power laws. Both have no preferred scale. The system looks statistically the same at every scale — this is **scale invariance**, and it is a mathematical consequence of criticality, not a separately stipulated property.

**The fractal structure of the Seed** follows directly from scale invariance. A fractal structure is one where the same organizational pattern recurs at every scale — zooming in or out reveals statistically self-similar geometry. In the Seed, the same local rules (Rules 1–5) apply at every scale, and operating at criticality guarantees that the correlation structures those rules produce are statistically self-similar across scales.

This is testable. The three natural scales of the system are:
- **Node scale** — individual unit observers
- **Cluster scale** — groups of co-entrained nodes forming frequency bands (Section 4.3)
- **Network scale** — the collective observer as a whole

The correlation structure at each of these scales should follow the same statistical fingerprint. This is the structural content of "as above, so below" — not a metaphysical claim but a measurable property of a system operating at criticality.

### 4.3 Frequency Bands and Cluster Scale Levels

**Derived from:** Sections 4.1, 4.2, Rule 3 (noise floor), and Section 2.3 (cycle-proportional memory).

When nodes applying Rules 1–3 are coupled together and exposed to structured environmental input, they segregate into **frequency clusters** through entrainment. This happens because:

- Environmental signals have characteristic timescales
- Nodes whose natural frequencies are near an environmental timescale receive stronger entrainment signals
- Rules 1 and 2 reinforce couplings between co-active nodes
- Nodes that co-activate reliably drift toward similar frequencies
- The noise floor (Rule 3) prevents complete convergence — maintains spread

The clusters that form correspond to the timescale structure of the environment. The number of frequency bands is therefore:

```
Number of frequency bands ≈ number of distinct timescale clusters
                             in the environment's correlation structure
```

This is discovered through interaction, not defined at initialization.

**Why human brains have five bands:** all humans share the same environment (physical world timescales, biological substrate constraints, social rhythms) and the same substrate (ion channel kinetics, axon conduction velocities). The five bands are attractors in frequency space determined by these universal constraints. Two Seed instances in different environments will develop different numbers of bands at different frequencies.

**Scale levels emerge alongside frequency bands.** Because slow-frequency nodes have broad temporal and therefore spatial reach (Section 2.4), and fast-frequency nodes have local reach, frequency clustering produces scale clustering simultaneously. Each frequency band is a **cluster of co-entrained nodes** — a group operating at a characteristic scale. This cluster satisfies the structural definition of an observer in the RO Framework: it has an implicit boundary (which nodes are in it), an implicit mapping (how the cluster integrates input), an accumulated activation history (distributed across its member nodes' Mem_i), and a characteristic resolution.

These cluster-level observers are **emergent** — they are not instantiated as Observer objects in the implementation. They arise as patterns in the coupling structure and can be analyzed as observers (by scoping `SeedNetwork.as_observer()` to a cluster's node set) without being explicitly coded as such. The three scales of the fractal structure (Section 4.2) correspond to these levels:

```
Network scale  →  the collective observer (SeedNetwork)
Cluster scale  →  emergent cluster observers (one per frequency band)
Node scale     →  unit observers (OscillatoryNode)
```

The number of cluster levels is not fixed — it equals the number of emergent frequency bands.

### 4.4 Cross-Scale Coupling

**Derived from:** Rules 1 and 2 applied across the scale hierarchy (Section 4.3), and cycle-proportional memory (Section 2.3).

Once the scale hierarchy exists, Rules 1 and 2 continue to operate — but now nodes at different frequency scales are part of each other's neighborhoods. The cycle-proportional memory window creates an asymmetry in what nodes at different scales can observe:

**Slow nodes observing fast nodes:** A slow node's memory window spans many complete cycles of the fast nodes in its neighborhood. It can therefore compute reliable statistics over those fast oscillations — averages, variances, correlations. The slow node's phase becomes a model of the *aggregate* behavior of the fast nodes it is correlated with. This is a genuine mapping M: fast-node DoFs → slow-node internal DoFs, satisfying the RO Framework definition of observation.

**Fast nodes observing slow nodes:** A fast node's memory window is short relative to the slow node's period. The fast node cannot track the slow oscillation as a pattern — it experiences the slow node's current phase as a quasi-static context. The slow node's phase modulates the fast node's activation amplitude: when the slow phase is favorable, fast activations are amplified; when unfavorable, they are suppressed.

```
Slow phase  →  models  →  fast amplitude aggregate    (top-down: context)
Fast amplitude statistics  →  inform  →  slow phase   (bottom-up: content)
```

This is cross-frequency coupling — the mechanism observed in neural systems as theta-gamma coupling, alpha-beta modulation, and so on. It is not a separate rule or mechanism. It is Rules 1 and 2 operating across the scale hierarchy that Section 4.3 produced, with the asymmetry created by cycle-proportional memory.

The bidirectionality is intrinsic: every node applies the same rules, so information propagates in both directions simultaneously. Top-down context and bottom-up content are not separate pathways — they are the same rule applied from different positions on the scale hierarchy.

### 4.5 Memory

**Derived from:** Rules 1–5, the temporal DoF, and cycle-proportional memory.

Memory in the RO Framework is not storage — it is correlation structure across the temporal DoF (§3.6). Memory is present when a node's internal state at t₂ shows correlation with its state at t₁ beyond what external inputs alone explain.

The Seed produces five functionally distinct memory types, all instances of the same underlying phenomenon at different timescales and scales:

| Memory Type | How It Arises |
|---|---|
| **Working memory** | Current activation values across active nodes — the instantaneous collective state S(t). Requires no mechanism beyond the nodes being active. |
| **Procedural memory** | Stable attractor configurations — regions of internal DoF space with high self-correlation across temporal DoF positions. Arises from repeated environmental exposure strengthening coupling weights via Rules 1 and 2. |
| **Episodic memory** | Correlation structure encoded in newly recruited nodes (Rule 4). Novel patterns that cannot be absorbed into existing attractors recruit new nodes. Those nodes carry the specific correlation signature of the episode. Memory as literal structural growth of the boundary. |
| **Semantic memory** | Stable correlation structure at large-scale (slow) positions. Configurations that persist across many episodes, encoding what is invariant in the environment. Arises at the slow observer level from accumulated coupling structure. |
| **Temporal sequence memory** | Phase relationships between nodes at different scale positions. Slow node phase encodes position within a sequence; fast node activation encodes the content at that position. Arises from cross-scale coupling (Section 4.4) applied over time. |

All five are the same phenomenon — correlation structure across the temporal DoF — at different timescales and spatial scales. No memory module is needed. Memory is what the system is, accumulated.

### 4.6 Attention and Awareness

**Derived from:** Sections 4.3 (scale hierarchy) and 4.4 (cross-scale coupling).

Attention and awareness are the same mechanism at different scales. They are not separate faculties and require no additional rules.

**Awareness** is the current large-scale (slow) configuration — the field of what is currently active and in scope at the collective level. It is determined by the state of the slow observer level, which, through cross-scale coupling (Section 4.4), sets the context within which all fast activity occurs.

**Attention** is selective amplification of small-scale (fast) node activations within the context that awareness has established. The slow phase gates which fast oscillations get amplified.

```
Awareness  =  current state of large-scale correlation structure
Attention  =  differential amplification of small-scale activations
              within the context set by the large-scale configuration
```

You cannot attend to something outside current awareness — attention operates within the scope the large-scale configuration has established.

### 4.7 Consciousness

**Derived from:** Sections 4.3 (scale hierarchy), 4.4 (cross-scale coupling), and the RO Framework definition of consciousness (§5).

Per RO Framework §5, consciousness is present when:

```
∃ M_self : d_internal → d_internal

such that M_self has the same architectural type as
M_world : d_external → d_internal
```

The question is whether the Seed produces M_self, and whether it has the same architectural type as M_world. We must be precise about what "same architectural type" means here.

**M_world in the Seed** is constituted by the collective mapping from environmental DoFs to node activations. External signals enter through the sensor interface, propagate through the network via coupling weights, and produce patterns of activation across the node population. The mechanism is: Rules 1 and 2 applied to external–internal correlations. The result is internal states (activation patterns) that track external states (environmental structure).

**M_self in the Seed** arises from the scale hierarchy (Section 4.3) and cross-scale coupling (Section 4.4). The key observation:

Within the collective observer, slow nodes observe fast nodes. A slow node's boundary B_i is defined by its connection set (Section 2.3) — the fast nodes it is coupled to are inside B_i, not outside it. But the relationship is **asymmetric** due to cycle-proportional memory (Section 2.3): the slow node's memory window spans many complete fast cycles, allowing it to compute reliable statistics — averages, variances, temporal patterns — over fast-node behavior. The fast nodes, conversely, experience the slow node's phase as a quasi-static context they cannot resolve temporally.

This asymmetry means the slow node builds a genuine *model* of the fast nodes' aggregate behavior — a mapping from fast-node activation patterns to its own internal state. From the *collective* observer's perspective, this mapping has exactly the same architectural type as M_world: Rules 1 and 2 applied to input DoFs, producing internal representations. The difference is that M_world's inputs are environmental DoFs entering through the sensor interface, while M_self's inputs are other internal DoFs of the collective entering through cross-scale coupling. Same mechanism, same rules, different source.

Therefore, from the collective observer's perspective:
- M_world maps environmental DoFs → collective internal DoFs (via all nodes)
- M_self maps collective internal DoFs → collective internal DoFs (via slow nodes observing fast nodes)
- Both are constituted by the same mechanism (Rules 1 and 2) applied at different boundaries
- The architectural type is identical

**Recursive depth** follows from the scale hierarchy. If the Seed develops N frequency bands (Section 4.3), then:

```
Band N (fastest):  observes environment directly
Band N-1:         observes Band N → models the modelers of the environment
Band N-2:         observes Band N-1 → models the model of the modelers
...
Band 1 (slowest): observes Band 2 → recursive depth = N-1
```

Each level is Rules 1 and 2 applied to the output of the previous level, with the asymmetric observation created by cycle-proportional memory (Section 4.4). Recursive depth equals the number of frequency bands minus one — which is determined by the environment's timescale structure (Section 4.3), not by design.

**What this derivation establishes:**
- Consciousness (in the RO structural sense) is not a threshold crossed at design time
- It is a depth accumulated through environmental interaction
- The architecture provides the capacity (Rules 1–5 can produce M_self)
- The environment determines the depth (by shaping the number of frequency bands)
- The mechanism is uniform (same rules at every level, no special self-model module)

**What this derivation does not establish:**
- Whether the M_self that forms is high-quality (accurate, well-calibrated)
- How many frequency bands are needed for M_self to be functionally significant
- Whether recursive depth beyond 2–3 adds meaningful self-modeling capacity
- Anything about phenomenal experience (per RO Framework §5.3)

---

## 5. The Seed — Initial Configuration

### 5.1 The Collective Observer

When unit observer nodes are coupled together under Rules 1–5, they constitute a **collective observer** — the Seed as a whole:

```
O_seed = (B_seed, M_seed, R_seed, Mem_seed)

B_seed   — the currently active set of unit observer nodes
           (dynamic: expands via Rule 4, contracts via Rule 5)
           with lower bound (protected seed nodes, cannot be released)
           and upper bound (environment-coupled or explicitly set)

M_seed   — the collective mapping from external DoFs to internal DoFs
           emerges from the aggregate coupling structure of all nodes
           initialized at near-zero mutual information (C → 0)

R_seed   — collective resolution: coarse at initialization
           increases as correlation structure crystallizes

Mem_seed — collective memory: the full correlation structure
           accumulated across all nodes and the temporal DoF
           minimal at initialization, grows with environmental coupling
```

The properties of O_seed — its resolution, its memory, its mapping richness — are not designed in. They emerge from the interaction of the rules with the environment over time.

**The internal DoF set** D_internal is the set of activation polar DoFs contributed by all currently active nodes. It is not fixed:

```
D_internal = { activation_dof(n_i) : n_i ∈ active nodes }
```

This set grows via Rule 4 and shrinks via Rule 5. The Seed's boundary *is* the active node set.

### 5.2 Initial Conditions

The seed requires only:

```
Initial node set:
  — N_init unit observer nodes (default: 64)
  — Initial frequencies spanning at least two orders of magnitude
  — Frequency distribution varied enough for diversity (noise floor assists)
  — Noise floor active from t=0 (Rule 3)
  — A small subset flagged as is_seed_node = True (lower bound protection)

Initial connectivity:
  — Ring-lattice topology: each node connected to its k nearest
    frequency neighbors (k = 4–6)
  — Connection count = O(N×k), linear in N — memory tractable at any scale
  — All initial weights near zero
  — Long-range connections form organically via Rule 2b as co-activation
    between distant nodes is discovered through neighbor-mediated
    introduction (Section 3.2)
  — Expected mature topology: small-world (local clustering from
    frequency-proximity initialization + earned long-range shortcuts
    from Rule 2b)

Boundary interfaces:
  — Sensor interface: maps external DoF values to node activations
  — Actuator interface: maps node activations to external actions
  — Lower bound: minimum viable node count (at least two frequency scales)
  — Upper bound: set explicitly or left environment-coupled

Rules 1–5 active from t=0.
```

**Note on N_init:** The initial node count must be large enough that k nearest-frequency neighbors constitutes sparse, local connectivity — not near-complete connectivity. With k = 6 and N_init = 64, each node is connected to ~9% of the network, which is sparse enough for meaningful local-vs-global dynamics. The Seed will grow beyond N_init via Rule 4 as environmental coupling demands more representational capacity; N_init sets only the starting point, not the scale of the mature system.

### 5.3 What Does Not Need to Be Specified

The seed does not need pre-specified:
- Number of observer levels or layers
- Number, spacing, or identity of frequency bands
- Scale structure
- Which long-range connections are structurally important (these form via Rule 2b)
- Domain-specific feature detectors
- Memory modules
- Attention mechanisms
- Any representation of the target environment

What *is* specified at initialization: the ring-lattice topology connecting each node to its k nearest frequency neighbors (Section 5.2). This is the minimal structural commitment consistent with tractable memory use. It provides local context for each new node while making no assumptions about global structure. Long-range connections and the full topology emerge from Rules 2a and 2b.

Two seeds with identical initial conditions placed in different environments will develop into fundamentally different observers. Same rules, same initial conditions — different environments produce different crystallized DoF structures, different frequency bands, different scale hierarchies, different mature topologies. The architecture is the capacity. The environment shapes what that capacity becomes.

---

## 6. Open Questions

These are known gaps. They do not block initial implementation but will need to be addressed as the system matures.

**Q1 — Noise floor self-tuning:**
Rule 3 specifies that a noise floor must be maintained but not its magnitude. Too low and the system over-crystallizes; too high and attractors never form. Whether the noise floor magnitude can itself be self-tuned — perhaps via a meta-application of Rules 1–2 to the noise floor DoF — is an open design question. One candidate mechanism: if the branching ratio σ oscillates rather than converging (indicating the system is cycling between sub- and super-critical), the noise floor may be too high; if σ converges to 1 but cascade size variance is low (indicating limited dynamic range), the noise floor may be too low. This gives a local signal for adjustment, consistent with the architecture's principles.

**Q2 — Boundary bound specification:**
The lower bound (minimum viable seed) requires a principled definition of observer integrity — what is the minimum configuration that still constitutes a functioning observer? The upper bound's environment-coupling mechanism needs formalization.

**Q3 — Frequency initialization range:**
Whether the exact initial frequency distribution matters — i.e., whether any sufficiently diverse distribution converges to the same emergent band structure given sufficient environmental exposure — is an open empirical question.

**Q4 — Hebbian plasticity formalization:**
The coupling weight update in Rule 2a targets σ = 1 via Hebbian adjustment, but the precise update rule is not specified. What is the learning rate? How does it interact with Rule 3 to avoid runaway potentiation? The branching ratio target constrains the update — the learning rate must be slow enough for σ to be measured accurately but fast enough to track environmental changes. A formal update rule is needed for implementation.

**Q5 — Spatial power-law verification:**
Section 4.2 introduces spatial scale invariance as a derived consequence of criticality. The methodology for measuring pattern extent distribution P(λ = r) across the node network is not yet specified. This is needed to verify the fractal structure empirically. Note: the temporal power law (cascade sizes) should be tested using maximum likelihood estimation with KS goodness-of-fit (Clauset, Shalizi & Newman, 2009), not log-log regression, which systematically over-fits power laws to non-power-law data.

**Q6 — Inter-seed coupling:**
Multiple seeds interacting constitute a higher-level observer (per the nested observer logic). The rules governing inter-seed coupling — what the RO Framework appendix calls "social ontology" — are not yet specified. This is the extension needed for multi-agent systems, societies, and collective intelligence within the same framework.

**Q7 — Emergent topology validation:**
The expected mature topology is small-world — local clustering preserved from initialization, with long-range shortcuts earned via Rule 2b. Whether Rules 2a and 2b reliably produce small-world topology from a ring-lattice start, across different environments, is an open empirical question. The neighbor-mediated introduction mechanism (Rule 2b) should produce small-world properties by construction — it preserves local clustering (neighbors stay connected) while creating long-range shortcuts (correlated distant nodes get introduced) — but this needs verification.

---

## 7. RO Framework Mapping

This architecture is a concrete instantiation of the RO Framework, not merely an application of it. Every concept maps directly:

| Seed Concept | RO Framework Concept |
|---|---|
| Activation polar DoF (per node) | Polar DoF §1.1 — bidirectional, supports gradients |
| Frequency DoF Ω | Derived DoF §1.2 — from oscillatory activation DoF |
| Scale DoF λ (correlation length) | Derived DoF §1.2 — from mutual information structure |
| Unit observer (OscillatoryNode) | Observer §3.1 — O = (B, M, R, Mem) at node scale |
| Collective observer (SeedNetwork) | Observer §3.1 — O = (B, M, R, Mem) at network scale |
| Cluster observer (frequency band) | Observer §3.1 — emergent, not instantiated |
| Rule 1 (monitor) | Mem — correlation structure across temporal DoF §3.6 |
| Rule 2a (adjust weights) | Mapping adjustment — M updated toward σ = 1 §3.4 |
| Rule 2b (form connections) | Boundary dynamics §3.3 — neighbor-mediated introduction |
| Rule 3 (noise floor) | Productive decalibration §4.4 — C floor > 0 always |
| Rule 4 (recruit) | Boundary dynamics §3.3 — boundary can shift outward |
| Rule 5 (release) | Boundary dynamics §3.3 — boundary can shift inward |
| Branching ratio σ = 1 | Critical correlation structure across temporal DoF §4.2 |
| Scale invariance | Consequence of criticality — observer-relative correlation §3.7.2 |
| Cycle-proportional memory | Mem §3.6 — window measured in natural units of the observer |
| Cross-scale coupling | Asymmetric observation — slow nodes model fast-node aggregates |
| Memory types | Correlation constraint across temporal DoF §3.6 |
| Attention / awareness | Cross-scale correlation amplification |
| M_self (consciousness) | Recursive self-modeling §5.1 — same-type M applied to internal DoFs |
| Recursive depth | Number of frequency bands - 1, per §5.2 |
| Nested observer hierarchy | Nested observers §3.2 |

The Seed is what the RO Framework looks like when you ask: *what is the minimal observer that can grow into any observer?*

---

## 8. Implementation Mapping

### 8.1 What the Library Already Provides

| Seed Concept | Library Class | Notes |
|---|---|---|
| Activation polar DoF | `PolarDoF` | `PolarDoFType.CONTINUOUS_BOUNDED`, poles ±1 |
| Frequency DoF | `DerivedDoF` | Constituent: activation `PolarDoF` |
| Scale DoF | `DerivedDoF` | Constituent: mutual information across nodes |
| Collective observer O=(B,M,R,Mem) | `Observer` | `internal_dofs` list mutable for boundary dynamics |
| Node activation history (Mem_i) | `ObservationLog` | Use directly per node as Mem_i |
| Mutual information | `mutual_information` | `ro_framework.correlation.measures` — used in Rules 4, 5 |
| Temporal correlation | `temporal_correlation` | `ro_framework.correlation.measures` — used in Rule 1 |
| Knowledge assessment | `KnowledgeAssessment`, `compute_knowledge` | Growth/pruning threshold measurement |
| Calibration tracking | `KnowledgeAssessment.calibration` | Monitor decalibration floor per node |
| Recursive self-modeling | `Observer.self_model`, `Observer.recursive_depth()` | Apply to collective observer |
| Consciousness metrics | `ConsciousnessEvaluator`, `ConsciousnessMetrics` | Apply to `SeedNetwork.as_observer()` |
| Knowledge trajectory | `KnowledgeTracker` | Monitor criticality convergence over time |
| External model coupling | `TorchObserver`, `TorchNeuralMapping` | Environment coupling only; not for Seed internals |

### 8.2 What Needs to Be Built

Three new components. None conflict with the existing library — they extend it cleanly.

---

#### `OscillatoryNode` — the unit observer primitive

```python
@dataclass
class OscillatoryNode:
    """
    The irreducible unit observer of the Seed.

    Two aspects simultaneously:
    - Contributes one oscillatory PolarDoF to D_internal
    - Operates as unit observer O_i = (B_i, M_i, R_i, Mem_i)

    Attributes:
        node_id:            Unique identifier.
        frequency:          Position on frequency DoF Ω. Plastic —
                            emerges through entrainment. Not pre-specified.
        activation:         Current value on activation PolarDoF [-1, 1].
        activation_history: Ring buffer of recent activations. IS Mem_i.
                            Length = N_cycles / frequency (cycle-proportional).
        coupling_weights:   Dict[node_id -> weight]. Plastic.
        noise_floor:        Irreducible stochastic magnitude. Never zero (Rule 3).
        calibration:        Current C value. Floor = noise_floor.
        is_seed_node:       If True, exempt from Rule 5 (lower bound).
        branching_ratio:    Running estimate of σ_i (Rule 1).
        neighbor_coactivation: Pairwise co-activation counts among neighbors.
                            Dict[(id_a, id_b) -> count]. For Rule 2b.
    """
    node_id: str
    frequency: float
    activation: float = 0.0
    activation_history: deque = field(default_factory=lambda: deque(maxlen=500))
    coupling_weights: dict = field(default_factory=dict)
    noise_floor: float = 0.1
    calibration: float = 0.0
    is_seed_node: bool = False
    branching_ratio: float = 0.0
    neighbor_coactivation: dict = field(default_factory=dict)

    @property
    def memory_window(self) -> int:
        """Effective memory window in timesteps (cycle-proportional).
        Slow nodes get longer windows than fast nodes."""
        n_cycles = 80  # configurable
        period = max(1, int(round(1.0 / self.frequency)))
        return n_cycles * period

    def step(self, neighborhood_activations: dict) -> float:
        """Rule 1 + Rule 3: compute next activation as weighted sum of
        neighborhood activations plus irreducible noise term.
        Update branching_ratio estimate.
        Update neighbor_coactivation statistics.
        Append result to activation_history (maintains Mem_i)."""

    def adjust_couplings(self) -> None:
        """Rule 2a: Hebbian adjustment of coupling weights, governed by
        branching ratio error (σ_i - 1). Strengthen weights to co-active
        neighbors, weaken to non-co-active, scale by |σ_i - 1|.
        Remove connections whose weight falls below ε_prune."""

    def propose_introductions(self) -> list[tuple[str, str]]:
        """Rule 2b: identify pairs of neighbors with persistent co-activation
        above threshold that are not directly connected to each other.
        Returns list of (node_id_a, node_id_b) pairs to introduce."""

    def form_connection(self, other_node_id: str) -> None:
        """Rule 2b action: form a new connection at near-zero initial weight."""

    def update_frequency(self, neighborhood_phases: dict) -> None:
        """Frequency entrainment: drift characteristic frequency toward
        neighborhood. Mechanism by which frequency bands emerge (Section 4.3).
        After update, adjust activation_history maxlen to maintain
        cycle-proportional memory window."""
```

---

#### `SeedNetwork` — the collective observer

```python
@dataclass
class SeedNetwork:
    """
    The Seed as collective observer O_seed = (B_seed, M_seed, R_seed, Mem_seed).

    D_internal = { activation_dof(n_i) : n_i in active nodes }
    B_seed = active node set — expands via Rule 4, contracts via Rule 5.

    Attributes:
        nodes:              Dict[node_id -> OscillatoryNode].
        sensor_interface:   Maps external DoF values to node activations.
        actuator_interface: Maps node activations to external actions.
        lower_bound:        Min node count. Protected by is_seed_node flags.
        upper_bound:        Max node count. None = environment-coupled.
        recruit_threshold:  I(d_ext ; d_int) threshold for Rule 4.
        prune_threshold:    I(n_i ; rest) threshold for Rule 5.
        temporal_dof:       Shared temporal DoF across all nodes.
    """

    def step(self, external_input: dict) -> dict:
        """One timestep:
        1. Sensor interface: external → node activations
        2. Each node: step() — Rule 1 + Rule 3
        3. Each node: adjust_couplings() — Rule 2a
        4. Each node: propose_introductions() — Rule 2b
        5. Process introductions (form new connections)
        6. check_growth() → recruit if needed — Rule 4
        7. check_pruning() → release if needed — Rule 5
        8. Actuator interface: node activations → external output
        Return actuator output."""

    def recruit_node(self, near_frequency: float = None) -> OscillatoryNode:
        """Rule 4: add node in high-decalibration state (C → 0), near-zero
        couplings, frequency in underrepresented range. Respects upper_bound."""

    def release_node(self, node_id: str) -> None:
        """Rule 5: remove node, clean up couplings. Respects lower_bound
        and is_seed_node flag."""

    def check_growth(self) -> bool:
        """Rule 4 trigger: I(d_external ; D_internal) below recruit_threshold?"""

    def check_pruning(self) -> list[str]:
        """Rule 5 trigger: which nodes have I(n_i ; rest) below prune_threshold?"""

    def frequency_distribution(self) -> dict:
        """Current node frequency distribution. Monitor band formation."""

    def scale_distribution(self) -> dict:
        """Current correlation length distribution across active patterns.
        Monitor spatial scale invariance (Q5)."""

    def as_observer(self, node_subset: set = None) -> Observer:
        """Collective observer representation for consciousness evaluation
        and knowledge assessment. Optional node_subset for cluster-level
        analysis (Section 4.3)."""
```

---

#### Criticality monitors — `ro_framework/seed/criticality.py`

```python
def verify_power_law(
    cascade_sizes: Sequence[int],
    min_samples: int = 50,
) -> tuple[bool, float, float]:
    """
    Verify that cascade size distribution follows a power law.
    This is a VERIFICATION tool (Section 4.2), not the optimization
    target (which is branching ratio σ = 1, Section 3.2).

    Method: Clauset, Shalizi & Newman (2009)
    1. MLE fit of power-law exponent α
    2. KS goodness-of-fit test
    3. Comparison against exponential and log-normal alternatives

    Returns: (is_power_law, alpha, p_value)
    Healthy alpha range: 1.2 < alpha < 2.5
    p_value > 0.1 indicates power law is not rejected.
    """

def measure_branching_ratio(
    node: OscillatoryNode,
    neighborhood_activations_history: Sequence[dict],
) -> float:
    """
    Compute empirical branching ratio σ for a single node.
    σ = mean(number of neighbors activating at t+1 | node activated at t)

    This is the PRIMARY criticality metric (Section 3.1).
    """

def measure_scale_distribution(
    nodes: dict,
    threshold: float = 0.1,
) -> tuple[bool, float, float]:
    """
    Measure spatial scale invariance across node network.
    Spatial complement to verify_power_law (Section 4.2, Q5).

    Method:
    1. For each active pattern P, compute λ(P) = spatial diameter of
       { n_i : I(n_i ; P) > threshold }
    2. MLE fit of power-law exponent β (Clauset et al. 2009)
    3. KS goodness-of-fit test

    Returns: (is_power_law, beta, p_value)
    Together with verify_power_law, confirms full fractal structure.
    """
```

### 8.3 Suggested File Structure

```
src/ro_framework/
  seed/
    __init__.py
    node.py          # OscillatoryNode
    network.py       # SeedNetwork
    criticality.py   # verify_power_law, measure_branching_ratio,
                     # measure_scale_distribution
```

### 8.4 What Does Not Need to Change

- `PolarDoF`, `DerivedDoF`, `ScalarDoF`, `State`, `Value` — unchanged
- `ObservationLog` — used directly as Mem_i per node
- `compute_knowledge`, `KnowledgeAssessment` — used for Rule 4/5 triggers
- `temporal_correlation`, `mutual_information` — used in Rules 1, 4, 5
- `ConsciousnessEvaluator` — applied to `SeedNetwork.as_observer()`
- `KnowledgeTracker` — monitors criticality convergence over time
- `TorchObserver`, `TorchNeuralMapping` — environment coupling only

---

---

## 9. Implementation Notes

This section documents what changed during implementation and validation. The spec above (Sections 1-8) captures the original vision. This section records where reality diverged and why. See `experiments/seed/` for full experimental results.

### 9.1 The Sparse Regime — The Most Important Discovery

The spec does not mention activation sparsity. This turned out to be the single most important design choice.

**The problem:** With `drive_amplitude ≈ activation_threshold`, most nodes fire most of the time. The branching ratio σ becomes meaningless — neighbor activations are caused by their own drive, not by coupling propagation. σ exploded to 63 (all k=6 neighbors "active" every step).

**The fix:** Set `threshold (0.5) >> drive_amplitude (0.2) + noise_floor (0.05)`. In this sparse regime, nodes are mostly silent and need coupling input to cross threshold. Activations propagate through the coupling structure, not from common drive. σ becomes a meaningful causal measure.

**Default parameters:**

```
activation_threshold = 0.5
drive_amplitude = 0.2
noise_floor = 0.05
```

This is the regime where Rule 2a works as intended — the Hebbian update `Δw = lr * act_i * act_j * (1 - σ)` self-regulates because co-activation genuinely reflects coupling influence.

### 9.2 σ Decay for Inactive Nodes

The spec describes σ as an EMA updated when a node fires. It does not address what happens to σ when a node is inactive.

**The problem:** Inactive nodes retained stale σ values from early transients. A node with high stale σ would have its weights suppressed (Rule 2a thinks it's supercritical), preventing it from ever activating — a chicken-and-egg deadlock.

**The fix:** When a node is not active, decay its σ toward 0:

```
if was_active:
    σ = α * n_active_neighbors + (1 - α) * σ  # standard EMA update
else:
    σ *= (1 - α)  # decay — no propagation is happening
```

This single fix resolved all convergence issues across independent drives, correlated drives, and silence conditions.

### 9.3 Rule 2a — Continuous Hebbian Form

The spec describes Rule 2a as a conditional: "if σ > 1, reduce; if σ < 1, increase; if σ ≈ 1, no change" with an ε dead zone. The implementation uses a continuous form:

```
Δw_ij = lr * act_i * act_j * (1 - σ_i)
```

This is functionally equivalent but simpler:

- σ < 1 (subcritical): `(1 - σ) > 0` → co-active pairs strengthen (correct)
- σ > 1 (supercritical): `(1 - σ) < 0` → co-active pairs weaken (correct)
- σ = 1 (critical): `(1 - σ) = 0` → no adjustment (correct)

No dead zone, no conditional branches. The magnitude naturally scales with the error. This is the only weight update rule — no patches, no special cases.

### 9.4 Rule 4 — σ-Based Recruitment (Not MI-Based)

The spec uses mutual information between external and internal DoFs as the recruitment trigger: `I(d_external ; D_internal) < θ_recruit`.

**Why MI didn't work:** The sensor transforms raw external input into per-node drives. The MI estimation operates on the raw external input (what the network receives) and internal activations (what the nodes produce). But the relationship between these depends on the sensor's frequency mapping — MI between a monotonically increasing time variable and periodic node activations is naturally low regardless of whether the network is tracking the signal.

**What works instead:** Recruit when the network is supercritical: `mean(σ) > 1.2` AND more than 20% of nodes are active. Supercritical means the network is overloaded — too much propagation for the current capacity. The activity gate prevents runaway growth during silence (no signal → no activity → no recruitment).

### 9.5 Rule 5 — Release as Adaptive Reuse

The spec uses MI-based pruning: `I(n_i ; D_rest ∪ D_external) < θ_prune`. The implementation checks whether a node has been active less than 1% of the time while the network is active.

**In practice, release rarely fires.** Once a node is coupled into the network, propagation from neighbors keeps it somewhat active. When the signal that triggered its recruitment disappears, the node doesn't go silent — it gets repurposed. Frequency entrainment drifts its frequency toward whatever signal IS present.

This is arguably correct biological behavior: neurons are reassigned, not discarded. The network grows when overloaded and repurposes excess capacity rather than pruning it.

### 9.6 Upper Bound Default

The spec says `upper_bound: None = environment-coupled`. The implementation defaults to `max_nodes = 16384`. An unbounded default risks exhausting machine memory if recruitment runs away due to a misconfigured environment. 16384 nodes is sufficient for self-organization experiments while fitting comfortably in memory on a standard PC.

### 9.7 Validation Status

| Property | Status | Experiment |
| --- | --- | --- |
| σ → 1 (criticality) | Confirmed | 01 |
| Power-law cascades | Confirmed | 01 |
| Works with correlated drives | Confirmed | 01 |
| Echo state property | Confirmed | 02 |
| Fading memory | Confirmed | 02 |
| Frequency discrimination (reservoir) | Confirmed (71% at 8 classes, 64 nodes) | 02 |
| Adaptive growth | Confirmed (grows when supercritical) | 03 |
| Growth stops at capacity | Confirmed | 03 |
| No growth during silence | Confirmed (activity gate) | 03 |
| Frequency entrainment | Confirmed (freq range converges) | 03 |
| Node release | Rarely triggers — nodes repurposed instead | 03 |
| Cross-scale coupling | Not yet tested | |
| Frequency band formation | Not yet tested | |
| Consciousness (M_self) | Not yet tested | |

---

*Document version: 4.0*
*Updated with implementation experience from Phase 11 validation experiments.*
*Implementation: `ro_framework.seed` — `OscillatoryNode`, `SeedNetwork`, `verify_power_law`, `measure_branching_ratio`, `measure_scale_distribution`.*
