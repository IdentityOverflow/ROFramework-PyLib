"""
experiments/embodied/reservoir.py — Single fixed-weight ESN reservoir (PyTorch).

Architecture
------------
A single reservoir replaces the old 5-reservoir hierarchy.  The reservoir is a
fixed random recurrent network; only the readout layer (W_out in brain.py) is
trained.  All weights are initialised deterministically from a seed and never
updated.

Leaky integrator dynamics (one step):
    noise  = N(0, noise_scale)
    pre    = tanh(W_in @ x  +  W_res @ h  +  bias  +  noise)
    h_new  = (1 − α) · h  +  α · pre

GPU note
--------
This is GPU-first: all tensors live on the target device as float32.
W_res spectral-radius rescaling uses numpy (eigvals needs float64), then moves
to device.  At N=4096 the W_res matmul is well within a 4090's L2 cache.

Size presets (W_res memory footprint):
    RES_TINY   =   512  →   1 MB    fast unit tests / CPU
    RES_SMALL  =  1024  →   4 MB    quick experiments
    RES_MEDIUM =  2187  →  18 MB    3^7, matches old brain_gpu.py sub-reservoirs
    RES_LARGE  =  4096  →  64 MB    default for RTX 4090
    RES_XL     =  8192  → 256 MB    4090 comfortable
    RES_XXL    = 16384  →   1 GB    4090 has 24 GB VRAM, fine for experiments

Phase 2 extension points
------------------------
The constructor accepts input_slices, carrier_freqs, and carrier_amps.  These
are stored but unused in Phase 1.  Phase 2 will use them to inject per-slice
sinusoidal carrier noise, scaffolding frequency-specific resonance for each
sensory modality (vision, tactile, internal state).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class SingleReservoir:
    """
    A single fixed-weight ESN reservoir backed by PyTorch tensors.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector (e.g. 263 for full obs, or 266 with
        action feedback).
    size : int
        Number of reservoir neurons.  See module-level size presets.
    spectral_radius : float
        Target spectral radius of W_res.  Values close to 1.0 give long echo
        length; lower values give faster forgetting.  Default 0.99 (edge of
        chaos).  Note: spinning attractors are controlled by bias_scale=0 and
        noise, NOT by spectral_radius.
    noise_scale : float
        Std of per-step Gaussian noise added to the pre-activation.  Keeps
        dead neurons active and prevents attractor lock-in.
    alpha : float
        Leaky integrator rate ∈ (0, 1].  1.0 = standard ESN (no leaking);
        lower values slow the reservoir's response time.
    bias_scale : float
        Std of fixed random bias vector.  MUST be 0.0 in practice: any nonzero
        bias propagates through W_out and locks the motor output into a
        persistent turn direction (spinning attractor).
    seed : int
        Deterministic initialisation seed.  Reservoir weights are fully
        reproducible from the seed — they don't need to be saved.
    device : torch.device or None
        Target device.  None defaults to CPU.
    input_slices : list of slice or None
        [Phase 2] Partition of the input vector by sensory modality, e.g.
        [slice(0,242), slice(242,260), slice(260,263)] for vision/tactile/state.
        When provided alongside carrier_freqs, each slice's corresponding
        columns of W_in will be modulated at the matching carrier frequency.
    carrier_freqs : list of float or None
        [Phase 2] Carrier frequency per modality in cycles/step.  At 60 fps,
        0.2 cycles/step ≈ 12 Hz (fast, for vision); 0.067 ≈ 4 Hz (medium,
        for tactile); 0.017 ≈ 1 Hz (slow, for internal state).
    carrier_amps : list of float or None
        [Phase 2] Carrier amplitude per modality (additive sinusoidal term on
        top of white noise).
    """

    def __init__(
        self,
        input_dim: int,
        size: int,
        spectral_radius: float = 0.99,
        noise_scale: float = 0.9,
        alpha: float = 1.0,
        bias_scale: float = 0.0,
        seed: int = 42,
        device: Optional[torch.device] = None,
        # ── Phase 2 stubs — accepted but unused ───────────────────────────────
        input_slices:  Optional[List[slice]] = None,
        carrier_freqs: Optional[List[float]] = None,
        carrier_amps:  Optional[List[float]] = None,
    ) -> None:
        self._size        = size
        self._noise_scale = noise_scale
        self._alpha       = alpha
        self._dev         = device or torch.device("cpu")

        # ── Phase 2 stubs (stored for future use) ─────────────────────────────
        self._input_slices  = input_slices
        self._carrier_freqs = carrier_freqs
        self._carrier_amps  = carrier_amps
        self._step_count    = 0   # needed for carrier phase in Phase 2

        # ── Initialise weights deterministically with numpy ────────────────────
        rng = np.random.default_rng(seed)

        w_in_np  = rng.standard_normal((size, input_dim)).astype(np.float32)
        bias_np  = (rng.standard_normal(size) * bias_scale).astype(np.float32)
        w_res_np = rng.standard_normal((size, size)).astype(np.float32)

        # Rescale W_res to target spectral radius (eigvals needs float64)
        eigvals = np.linalg.eigvals(w_res_np.astype(np.float64))
        cur_sr  = float(np.max(np.abs(eigvals)))
        if cur_sr > 0:
            w_res_np *= float(spectral_radius) / cur_sr

        # Keep numpy copy of W_res for brain_viz spring layout
        self._w_res_np = w_res_np.copy()

        # ── Move to device as float32 ─────────────────────────────────────────
        self._W_in  = torch.from_numpy(w_in_np).to(self._dev)
        self._W_res = torch.from_numpy(w_res_np).to(self._dev)
        self._bias  = torch.from_numpy(bias_np).to(self._dev)
        self._h     = torch.zeros(size, dtype=torch.float32, device=self._dev)

    # ── Forward pass ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Advance one step.

        Parameters
        ----------
        x : torch.Tensor, shape (input_dim,), float32, on self._dev

        Returns
        -------
        h : torch.Tensor, shape (size,) — updated hidden state
        """
        noise = (
            torch.randn(self._size, dtype=torch.float32, device=self._dev)
            * self._noise_scale
        )
        pre     = torch.tanh(self._W_in @ x + self._W_res @ self._h + self._bias + noise)
        self._h = (1.0 - self._alpha) * self._h + self._alpha * pre
        self._step_count += 1
        return self._h

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero the hidden state (call on episode reset, not step counter)."""
        self._h.zero_()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Current hidden state as numpy float32 (for visualisation)."""
        return self._h.cpu().numpy()

    @property
    def recurrent_weights(self) -> np.ndarray:
        """Fixed W_res as numpy float32 (for brain_viz spring layout)."""
        return self._w_res_np

    @property
    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"SingleReservoir(size={self._size}, "
            f"noise={self._noise_scale}, alpha={self._alpha}, "
            f"device={self._dev})"
        )
