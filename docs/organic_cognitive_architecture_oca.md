# Organic Cognitive Architecture (OCA)

*A reservoir-based, continuously learning cognitive architecture inspired by biological brains but grounded in control theory and reinforcement learning.*

---

## 1. Design Goals

The Organic Cognitive Architecture (OCA) is designed to move away from the "statistical encyclopedia" paradigm of modern large language models and toward a **homeostatic, adaptive cognitive organism**.

Core goals:
- Continuous, lifelong learning without catastrophic forgetting
- Limited, fallible knowledge (no full-internet memorization)
- Energy-efficient computation
- Temporal coherence and internal state
- Individuality and history-dependent behavior

The system is intended to behave more like an **artificial animal** than a chatbot.

---

## 2. High-Level Architecture

The system consists of:

- A set of **interconnected reservoirs** (liquid neural regions)
- **Trainable bottleneck pipes** between regions
- A **long-term crystallized memory store** (Vector Database)
- A **global reward prediction error (RPE) signal** controlling learning
- Explicit **information flow regulation** (gating)
- A **circadian cycle** (online interaction vs offline consolidation)

The architecture emphasizes *dynamics first, structure second*.

---

## 3. Core Computational Substrate: Reservoirs

### 3.1 Reservoir Computing Basics

Each region is implemented as a **Reservoir** (e.g., Echo State Network / Liquid State Machine):

- Fixed, randomly connected recurrent network
- Operates near criticality (edge of chaos)
- Rich temporal dynamics
- No backpropagation through recurrent weights

Reservoirs act as **high-dimensional nonlinear state generators**.

Only the *connections between reservoirs* are trained.

---

## 4. Brain-Inspired Functional Regions

> Biological names are **functional metaphors**, not claims of anatomical equivalence.

### 4.1 Sensory Reservoirs

**Function:**
- Encode raw sensory streams (text, vision, audio, proprioception)
- Convert input into temporal state trajectories

**Properties:**
- Fast dynamics
- High bandwidth
- No long-term memory

---

### 4.2 Context / Association Reservoir (Hippocampal-like)

**Function:**
- Bind multi-modal inputs
- Maintain short-to-medium-term context
- Interface with long-term memory

**Key Mechanism:**
- Detects when internal state trajectories are about to decay
- Triggers memory write / refresh

---

### 4.3 Long-Term Memory (Crystallized Memory)

**Implementation:**
- External Vector Database (embedding-based)

**Stored Items:**
- Reservoir state snapshots
- Compressed trajectories
- Episodic summaries

**Access Pattern:**
- Read: similarity-based retrieval
- Write: gated by salience and surprise

This memory is **persistent**, unlike reservoirs.

---

### 4.4 Logic / Simulation Reservoir (Prefrontal-like)

**Function:**
- Forward-simulate internal dynamics
- Predict near-future states
- Detect potential "bad outcomes"

Rather than symbolic logic, reasoning emerges via **temporal projection**:

> "If I continue like this, does the trajectory enter a low-value region?"

---

### 4.5 Salience / Value Reservoir (Amygdala-like)

**Function:**
- Assign affective value to situations
- Filter information flow
- Decide what is worth remembering

**Outputs:**
- Salience signal (importance)
- Value estimate for Critic

Low-salience inputs may be ignored entirely.

---

### 4.6 Action / Expression Reservoir (Motor / Language Cortex)

**Function:**
- Generate actions, text, or control signals
- Compete between possible outputs

**Constraint:**
- Subject to gating and inhibition

---

### 4.7 Gating & Inhibition System (Basal Ganglia-like)

**Function:**
- Regulate information flow between regions
- Suppress impulsive or low-value actions

Implements:
- "Shut up and think" behavior
- Action selection under conflict

---

## 5. Learning Architecture: Dopamine as Plasticity Gate

### 5.1 Global Learning Signal

Learning is controlled by a **single scalar broadcast signal**:

> **Reward Prediction Error (RPE)**

This signal determines *whether* learning occurs, not *what* is learned.

---

### 5.2 Actor–Critic Decomposition

Learning is split into two roles:

#### 5.2.1 Critic (Value System)

**Role:**
- Predict expected value of the current global state

**Output:**
- Scalar value: `V(s)`

**Properties:**
- Slow learning
- Conservative updates
- Low plasticity

This forms the **baseline expectation**.

---

#### 5.2.2 Actor (Behavior System)

**Role:**
- Generate actions
- Control gating
- Write to memory

**Learning Rule:**
- Actor updates occur *only when RPE ≠ 0*

Stable, expected outcomes do not modify behavior.

---

### 5.3 Reward Prediction Error (RPE)

The learning signal is computed as:

```
RPE = R − V(s) − cost(action)
```

Where:
- `R` = observed outcome
- `V(s)` = predicted value
- `cost(action)` = small penalty to prevent spurious actions

**Interpretation:**
- RPE > 0 → outcome better than expected → reinforce
- RPE < 0 → worse than expected → suppress
- RPE ≈ 0 → no learning (habit preserved)

This prevents superstitious learning and noise accumulation.

---

## 6. Plasticity Targets

Only **bottleneck pipes** are trainable:

- Between reservoirs
- Between reservoirs and memory interface
- Between reservoirs and action heads

Reservoir internal weights remain fixed.

Plasticity is:
- Local
- Sparse
- Gated by RPE

---

## 7. Circadian Cycle (Sleep)

### 7.1 Wake Phase

- Online interaction
- Minimal learning
- Buffer experiences
- Fast reaction

---

### 7.2 Sleep Phase

- Replay buffered trajectories
- Recompute RPE offline
- Consolidate only high-surprise episodes
- Clean memory
- Adjust Critic slowly

This reduces interference and catastrophic forgetting.

---

## 8. Stability Mechanisms

To prevent pathological dynamics:

- Action costs
- Entropy regulation
- Novelty pressure
- Memory decay
- Gating thresholds

The system is kept near, but not beyond, criticality.

---

## 9. Expected Behavioral Properties

The OCA agent will:
- Learn continuously
- Forget unimportant details
- Develop habits
- Exhibit mood-like states
- Show individual divergence
- Occasionally ignore input

Failures are expected to be **human-like**, not catastrophic.

---

## 10. Positioning

This architecture is:
- Not a replacement for transformers
- Not optimized for benchmarks
- Not globally knowledgeable

It is optimized for:
- Persistence
- Adaptation
- Embodiment
- Long-term interaction

> **The goal is not intelligence-as-knowledge, but intelligence-as-survival.**

---

## 11. Summary

The Organic Cognitive Architecture integrates:
- Reservoir computing
- Actor–Critic reinforcement learning
- Externalized memory
- Homeostatic control

into a single coherent cognitive organism.

It is a blueprint for systems that can **grow, adapt, and age**, rather than merely answer questions.
