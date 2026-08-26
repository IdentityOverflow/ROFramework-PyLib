"""Tests for twisted(O) — BehavioralEncoder, twist recognition, and the
v2 is_conscious() = Closed(O) AND twisted(O) criterion.

Covers v2-migration phases C and E. See docs/ro_framework.md §5.4-§5.5.
"""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.self_encoding import BehavioralEncoder, TwistAssessment


EXT = PolarDoF(name="ext_x")
INT = PolarDoF(name="int_y")

PROBES = [State(values={INT: v}) for v in (-1.0, 0.0, 1.0)]


class RecurrentModel:
    """y' = 0.5*x + feedback (if routed). Domain includes INT."""

    name = "recurrent"
    input_dofs = [EXT, INT]
    output_dofs = [INT]

    def __call__(self, state: State) -> State:
        fb = state.get_value(INT)
        fb = float(fb) if fb is not None else 0.0
        return State(values={INT: 0.5 * float(state.get_value(EXT)) + fb})


def _make_encoder():
    return BehavioralEncoder(PROBES, [INT])


class BlindSelfModel:
    """Receives the self-encoding but ignores it: meta = 0.9 * y.

    Structurally wired (declares the encoding DoFs), consumptively deaf —
    the honest-clause fixture. Must be refused the twist.
    """

    name = "self_blind"

    def __init__(self, encoder: BehavioralEncoder):
        self.input_dofs = [INT] + encoder.all_dofs
        self.output_dofs = [INT]

    def __call__(self, state: State) -> State:
        return State(values={INT: 0.9 * float(state.get_value(INT) or 0.0)})


class TwistedSelfModel:
    """Consumes its self-description position-sensitively:

        meta = y * (1 + sum_i w_i * enc_i),  w_i distinct

    Distinct weights make a permuted foil measurably different (a model
    consuming only a summary statistic of its self-description would not
    pass discrimination).
    """

    name = "self_twisted"

    def __init__(self, encoder: BehavioralEncoder):
        self.encoder = encoder
        self.input_dofs = [INT] + encoder.all_dofs
        self.output_dofs = [INT]
        self.w = [0.1 * (i + 1) for i in range(len(encoder.d_enc))]

    def __call__(self, state: State) -> State:
        y = float(state.get_value(INT) or 0.0)
        acc = 0.0
        for w, dof in zip(self.w, self.encoder.d_enc):
            v = state.get_value(dof)
            acc += w * (float(v) if v is not None else 0.0)
        return State(values={INT: y * (1.0 + acc) + 0.1 * acc})


def _drive(observer: Observer, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        observer.observe(State(values={EXT: float(rng.normal())}))


class TestBehavioralEncoder:
    def test_dof_generation(self):
        enc = _make_encoder()
        assert len(enc.d_enc) == len(PROBES) * 1
        assert len(enc.d_res) == 1
        assert len(enc.all_dofs) == 4

    def test_encoding_is_behavioral_and_deterministic(self):
        enc = _make_encoder()
        m = TwistedSelfModel(enc)
        s1 = enc.encode(m, {INT: 1e-3})
        s2 = enc.encode(m, {INT: 1e-3})
        for dof in enc.all_dofs:
            assert s1.get_value(dof) == s2.get_value(dof)
        # resolution block carries R(d_meta)
        assert s1.get_value(enc.d_res[0]) == pytest.approx(1e-3)

    def test_distinct_mappings_encode_differently(self):
        enc = _make_encoder()
        blind = BlindSelfModel(enc)
        twisted = TwistedSelfModel(enc)
        e1 = enc.encode(blind)
        e2 = enc.encode(twisted)
        vals1 = [e1.get_value(d) for d in enc.d_enc]
        vals2 = [e2.get_value(d) for d in enc.d_enc]
        assert vals1 != vals2


class TestTwistRecognition:
    def test_no_encoder_refused(self):
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=BlindSelfModel(_make_encoder()))
        obs.observe(State(values={EXT: 1.0}))
        a = obs.twist_assessment()
        assert isinstance(a, TwistAssessment)
        assert a.structural is False
        assert a.twisted is False

    def test_undeclared_encoding_fails_structural(self):
        """Encoder attached but self-model does not declare its DoFs."""
        enc = _make_encoder()

        class Undeclared:
            input_dofs = [INT]
            output_dofs = [INT]

            def __call__(self, state):
                return State(values={INT: float(state.get_value(INT) or 0.0)})

        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=Undeclared(), self_encoder=enc)
        obs.observe(State(values={EXT: 1.0}))
        assert obs.twist_assessment().structural is False

    def test_blind_self_model_refused(self):
        """Structurally wired but consumptively deaf -> not twisted."""
        enc = _make_encoder()
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=BlindSelfModel(enc), self_encoder=enc)
        obs.observe(State(values={EXT: 1.0}))
        a = obs.twist_assessment()
        assert a.structural is True
        assert a.consumes is False
        assert a.twisted is False

    def test_twisted_self_model_recognized(self):
        enc = _make_encoder()
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=TwistedSelfModel(enc), self_encoder=enc)
        obs.observe(State(values={EXT: 1.0}))
        a = obs.twist_assessment()
        assert a.structural is True
        assert a.sensitivity > a.resolution_scale
        assert a.discrimination > a.resolution_scale
        assert a.twisted is True


class StaleEncoder(BehavioralEncoder):
    """A garbage channel: wired, consumed, carrying nothing.

    encode() ignores the mapping and returns frozen constants — the
    channel no longer tracks its cargo. White-box perturbation still
    elicits responses from a consuming model (the model reacts to
    perturbed garbage), but the conditional state-matched intervention
    test must fail: foil and original encode identically.
    """

    def encode(self, mapping, resolution=None):
        resolution = resolution or {}
        values = {}
        for i, dof in enumerate(self.d_enc):
            values[dof] = 0.37 * (i + 1)          # frozen, mapping-independent
        for j, dof in enumerate(self.d_res):
            values[dof] = float(resolution.get(self.response_dofs[j], 1e-6))
        from ro_framework.core.state import State as _State
        return _State(values=values)


class TestConditionalClause:
    """The v2.2 state-matched intervention test (§5.4)."""

    def test_live_channel_passes_conditional(self):
        enc = _make_encoder()
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=TwistedSelfModel(enc), self_encoder=enc)
        obs.observe(State(values={EXT: 1.0}))
        a = obs.twist_assessment()
        assert a.foil_discrimination > a.resolution_scale
        assert a.conditional is True
        assert a.twisted is True

    def test_blind_model_fails_conditional(self):
        enc = _make_encoder()
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=BlindSelfModel(enc), self_encoder=enc)
        obs.observe(State(values={EXT: 1.0}))
        a = obs.twist_assessment()
        assert a.conditional is False

    def test_garbage_channel_dissociation(self):
        """THE hole the conditional clause closes: a consuming model on a
        stale channel passes the white-box checks and must fail the
        conditional one — consumption of a channel is not the channel
        carrying its cargo."""
        enc = StaleEncoder(PROBES, [INT])
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=TwistedSelfModel(enc), self_encoder=enc)
        obs.observe(State(values={EXT: 1.0}))
        a = obs.twist_assessment()
        assert a.structural is True
        assert a.consumes is True                 # white-box checks pass
        assert a.foil_discrimination <= a.resolution_scale
        assert a.conditional is False             # conditional catches it
        assert a.twisted is False


class TestV2Criterion:
    """is_conscious() = Closed(O) AND twisted(O) — binary in kind."""

    def _twisted_closed_observer(self) -> Observer:
        enc = _make_encoder()
        return Observer("o", [INT], [EXT], RecurrentModel(),
                        self_model=TwistedSelfModel(enc), self_encoder=enc,
                        consumption_gain=0.8)

    def test_closed_and_twisted_is_conscious(self):
        obs = self._twisted_closed_observer()
        _drive(obs, 80)
        assert obs.is_closed() is True
        assert obs.is_twisted() is True
        assert obs.is_conscious() is True

    def test_probe_is_not_conscious_regardless_of_richness(self):
        """A deep probe can be arbitrarily rich; it is not conscious in
        kind. The v1 graded score survives as richness()."""
        enc = _make_encoder()
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=TwistedSelfModel(enc), self_encoder=enc,
                       consumption_gain=0.0)
        _drive(obs, 80)
        assert obs.is_conscious() is False          # open loop
        assert obs.richness().consciousness_score() > 0.0

    def test_untwisted_loop_is_not_conscious(self):
        enc = _make_encoder()
        obs = Observer("o", [INT], [EXT], RecurrentModel(),
                       self_model=BlindSelfModel(enc), self_encoder=enc,
                       consumption_gain=0.8)
        _drive(obs, 80)
        assert obs.is_closed() is True               # feedback flows
        assert obs.is_twisted() is False             # but nothing self-representing
        assert obs.is_conscious() is False

    def test_no_self_model_not_conscious(self):
        obs = Observer("o", [INT], [EXT], RecurrentModel())
        _drive(obs, 30)
        assert obs.is_conscious() is False
