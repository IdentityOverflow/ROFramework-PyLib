# RC-1 Findings: Modular Addition with a Fixed Reservoir

Empirical results from the first successful reservoir-computing experiment for the Recursive Observer Framework. Uses the same modular addition task as Phase 8, but with a fixed random reservoir and a trainable readout head.

## Setup

Final working configuration:

- Task: `(a, b) -> (a + b) mod 97`
- Split: 75% train / 25% test over all `(a, b)` pairs
- Reservoir: 729 fixed neurons
- Dynamics: one-hot `a` then one-hot `b`, with 5 recurrent settle steps per symbol
- Spectral radius: `0.99`
- Input scaling: `0.1`
- Readout head: `Linear(729→729) → ReLU → Linear(729→97)`
- Optimizer: AdamW, `lr=5e-3`, `weight_decay=1.2`
- Knowledge tracking target: sum-averaged readout hidden features

## Main Result

**The fixed reservoir generalizes modular addition, but only with a nonlinear readout head.**

With a linear readout, the system memorized but did not generalize: ridge regression and SGD both stayed near `0.0%–0.3%` test accuracy despite substantial train accuracy. With the nonlinear readout head, test accuracy rose steadily:

- Epoch 250: `2%`
- Epoch 500: `21%`
- Epoch 1000: `70%`
- Epoch 2000: `90%`
- Epoch 3000: `95%`
- Epoch 4000: `97%`
- Epoch 6750: `99%`

Train accuracy hit `100%` by epoch 250, then test accuracy continued climbing for thousands of epochs. So the experiment still shows a grokking-like delayed generalization dynamic even though the reservoir itself never changes.

## Reservoir Knowledge Before Training

The raw fixed reservoir contained only moderate latent knowledge of the discovered Fourier features:

- Single-feature assessments were mostly `uncertain`, with `ρ ≈ 0.36–0.44`
- Multi-feature assessments improved to `ρ ≈ 0.65–0.74`

So the reservoir did encode useful task structure, but not in a linearly accessible form strong enough for direct classification.

## Readout Knowledge During Training

Once the nonlinear readout head began training, `K(d_ext)` on the readout hidden features rapidly became strong:

- At epoch 0, most tracked frequencies were already classified as `strong`
- By epoch 250, all tracked frequencies had `ρ > 0.90`
- By the end of training, all tracked `sin`/`cos` features reached `ρ = 0.999`

This means the trained head learned to extract an almost perfectly Fourier-aligned internal basis from the fixed reservoir.

## Spectral Analysis

The reservoir itself did **not** show a strongly peaked Fourier spectrum before training:

- Top raw frequencies by DFT power were `k=15`, `k=24`, and `k=2`
- Power spectrum CV was `0.13`, indicating a nearly uniform spectrum rather than a sharp Fourier preference

So the successful result does not come from the reservoir being "pre-solved" in an obvious spectral sense. The latent signal is weakly distributed, and the readout head learns how to use it.

The trained readout also did not simply amplify the top pre-training reservoir frequencies. In logit space, the top frequencies shifted to `k=36`, `k=46`, `k=7`, and `k=2`, with only some overlap with the original reservoir spectrum. This suggests the learned head composes reservoir modes rather than merely selecting the strongest raw ones.

## Interpretation

This result narrows the claim.

What is true:

- A fixed random reservoir can support modular-addition generalization.
- The reservoir contains latent Fourier-structured information before training.
- The readout can learn to extract that structure while the reservoir remains frozen.

What is not true:

- A linear readout is sufficient for this task at `p=97`.
- The reservoir is born with obviously strong task-aligned Fourier modes.
- RC fully eliminates feature-behavior lag. It eliminates hidden-layer training, but once the readout becomes multi-layer, a smaller version of the same issue reappears: latent structure exists before behavior fully reflects it.

## Implications for the RO Framework

This is still a good RO-style result:

- Boundary is clean: fixed reservoir inside, one-hot symbols outside
- Mapping is structurally explicit: random recurrent dynamics plus trained readout
- Memory is real: the recurrent state carries partial computation from `a` to `b`
- Knowledge remains measurable through `K(d_ext)`

The main adjustment is conceptual: for RC on nontrivial symbolic tasks, "the readout is the knowledge" is still correct, but that readout may need nonlinear composition. A strict linear probe was too weak to demonstrate the architecture's actual capacity.

## Next Questions

1. Which ingredient mattered most: recurrent settle steps, fixed bias, or nonlinear readout depth?
2. Does `K(d_ext)` on readout hidden features predict later behavioral generalization better than raw accuracy?
3. Can K-guided training operate on readout hidden units without recreating Phase 8c's failure mode?
4. How much of the result depends on the specific train/test split and `p=97`?
