"""Self-encoding: the twist machinery (ro_framework.md §5.4).

The twist requires the self-model to represent its own representing — its
inputs must include a representation of its own mapping and of the
observer's resolution on d_meta. The encoding used here is BEHAVIORAL:
a mapping is represented by its responses on a fixed probe battery, i.e.
by its relational profile through a finite-resolution channel, not by its
parameters (which most MappingFunctions do not expose). The battery's size
and granularity IS the self-representation's resolution — R(d_meta) enters
by construction, and its values ride along as declared inputs.

The fixed-point regress (the encoding depends on the mapping's behavior,
which depends on its inputs, which include the encoding) is handled the
way this framework handles everything: unrolled across the temporal DoF.
The encoding computed at t is consumed at t+1. During encode() itself,
the mapping is probed with its self-description DoFs blanked to 0.0 —
probed "in the dark about itself" — so the encoding captures behavior
modulo self-description and the regress never enters the encoder.

Recognition (TwistAssessment) mirrors ClosureAssessment's discipline:
structure alone is not enough. Receiving your own description is not the
twist — USING it is. The `consumes` clause demands (a) sensitivity: output
moves beyond d_meta resolution under perturbation of the encoding inputs,
and (b) discrimination: output distinguishes the true self-encoding from a
permuted foil. Permutation preserves summary statistics, so a system that
consumes only a mean of its self-description — a statistic, not a
description — is refused.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ro_framework.core.dof import DoF, PolarDoF, ScalarDoF
from ro_framework.core.state import State
from ro_framework.observer.mapping import MappingFunction


@dataclass(frozen=True)
class TwistAssessment:
    """twisted(O) recognition result (ro_framework.md §5.4).

    Criteria:
        structural: the wiring exists — a self-encoder is attached and ALL
            of its DoFs (response block + resolution block) are declared
            inputs of the self-model.
        sensitivity: mean |Δ d_meta| under Gaussian perturbation of the
            encoding inputs, compared against d_meta resolution.
        discrimination: mean |Δ d_meta| between the true self-encoding and
            a permuted foil — position-sensitivity of the consumption.
        consumes: sensitivity AND discrimination both exceed resolution.
        twisted: structural AND consumes. Binary in kind; the graded
            magnitudes are reported alongside.
    """

    structural: bool
    sensitivity: float
    discrimination: float
    resolution_scale: float
    consumes: bool
    twisted: bool


class BehavioralEncoder:
    """Encodes a mapping by its responses on a fixed probe battery.

    Attributes:
        probe_states: The battery — input states for the mapping under
            encoding (values on its non-self-description input DoFs).
        response_dofs: The mapping's output DoFs to read responses from
            (normally the observer's d_meta).
        d_enc: Generated ScalarDoFs carrying the response values —
            one per (probe, response DoF).
        d_res: Generated ScalarDoFs carrying the observer's resolution
            on each response DoF (R(d_meta) as declared input).
    """

    def __init__(self, probe_states: List[State], response_dofs: List[DoF],
                 name_prefix: str = "selfenc") -> None:
        if not probe_states:
            raise ValueError("BehavioralEncoder requires at least one probe state")
        if not response_dofs:
            raise ValueError("BehavioralEncoder requires response DoFs")
        self.probe_states = list(probe_states)
        self.response_dofs = list(response_dofs)
        # Response values are signed (positions, not magnitudes): PolarDoF.
        # Resolution values are magnitudes: ScalarDoF.
        self.d_enc: List[DoF] = [
            PolarDoF(name=f"{name_prefix}_p{i}_{dof.name}")
            for i in range(len(probe_states)) for dof in response_dofs
        ]
        self.d_res: List[DoF] = [
            ScalarDoF(name=f"{name_prefix}_res_{dof.name}")
            for dof in response_dofs
        ]

    @property
    def all_dofs(self) -> List[DoF]:
        return self.d_enc + self.d_res

    def encode(self, mapping: MappingFunction,
               resolution: Optional[Dict[DoF, float]] = None) -> State:
        """Encode a mapping as its probe-battery response profile.

        Probes are presented with the encoder's own DoFs blanked to 0.0
        (behavior modulo self-description; see module docstring).

        Returns:
            State with values on d_enc (responses) and d_res (resolution).
        """
        resolution = resolution or {}
        values: Dict[DoF, float] = {}
        k = 0
        for probe in self.probe_states:
            probe_input = probe
            for dof in self.all_dofs:
                if probe_input.get_value(dof) is None:
                    probe_input = probe_input.set_value(dof, 0.0)
            response = mapping(probe_input)
            for dof in self.response_dofs:
                v = response.get_value(dof)
                values[self.d_enc[k]] = (
                    float(v) if isinstance(v, (int, float)) else 0.0
                )
                k += 1
        for j, dof in enumerate(self.response_dofs):
            values[self.d_res[j]] = float(resolution.get(dof, 1e-6))
        return State(values=values)


def assess_twist(
    observer,
    n_perturb: int = 8,
    perturb_scale: float = 0.1,
    seed: int = 0,
) -> TwistAssessment:
    """Recognize twisted(O) for an Observer (see TwistAssessment).

    Pure recognition over the observer's current configuration: no state
    is mutated and nothing is logged.
    """
    encoder = observer.self_encoder
    model = observer.self_model

    def _refuse(structural: bool = False) -> TwistAssessment:
        return TwistAssessment(
            structural=structural, sensitivity=0.0, discrimination=0.0,
            resolution_scale=0.0, consumes=False, twisted=False,
        )

    if encoder is None or model is None or observer.internal_state is None:
        return _refuse()

    declared = set(getattr(model, "input_dofs", None) or [])
    structural = set(encoder.all_dofs).issubset(declared)

    d_meta = observer.d_meta
    if not d_meta:
        return _refuse(structural)

    resolution = {d: observer.get_resolution(d) for d in d_meta}
    enc_true = encoder.encode(model, resolution)

    def _apply(enc_state: State) -> np.ndarray:
        s = observer.internal_state
        for dof in encoder.all_dofs:
            v = enc_state.get_value(dof)
            s = s.set_value(dof, float(v) if v is not None else 0.0)
        out = model(s)
        return np.array([
            float(out.get_value(d)) if isinstance(out.get_value(d), (int, float))
            else 0.0
            for d in d_meta
        ])

    base = _apply(enc_true)
    res_scale = float(np.mean([observer.get_resolution(d) for d in d_meta]))
    res_scale = max(res_scale, 1e-9)

    rng = np.random.default_rng(seed)
    enc_vals = np.array([
        float(enc_true.get_value(d) or 0.0) for d in encoder.d_enc
    ])

    # (a) sensitivity: Gaussian perturbation of the response block
    deltas = []
    for _ in range(n_perturb):
        noisy = enc_true
        for dof, v in zip(encoder.d_enc, enc_vals):
            scale = perturb_scale * (abs(v) + 1.0)
            noisy = noisy.set_value(dof, float(v + rng.normal(0.0, scale)))
        deltas.append(float(np.mean(np.abs(_apply(noisy) - base))))
    sensitivity = float(np.mean(deltas)) if deltas else 0.0

    # (b) discrimination: permuted foil (preserves summary statistics —
    # consuming a mere statistic of the self-description does not pass)
    disc_deltas = []
    for _ in range(max(1, n_perturb // 2)):
        perm = rng.permutation(len(enc_vals))
        if np.all(perm == np.arange(len(enc_vals))):
            continue
        foil = enc_true
        for dof, v in zip(encoder.d_enc, enc_vals[perm]):
            foil = foil.set_value(dof, float(v))
        disc_deltas.append(float(np.mean(np.abs(_apply(foil) - base))))
    discrimination = float(np.mean(disc_deltas)) if disc_deltas else 0.0

    consumes = sensitivity > res_scale and discrimination > res_scale
    return TwistAssessment(
        structural=structural, sensitivity=sensitivity,
        discrimination=discrimination, resolution_scale=res_scale,
        consumes=consumes, twisted=bool(structural and consumes),
    )
