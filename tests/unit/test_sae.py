"""Tests for SAE integration (SAEObserver).

All tests use mocked model and SAE — no GPU or model downloads required.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock torch and SAE libraries before importing sae module
import sys


@dataclass
class MockSAEConfig:
    d_sae: int = 100
    d_in: int = 768


class MockSAE:
    """Mock SAELens SAE that returns deterministic feature activations."""

    def __init__(self, d_sae: int = 100, d_in: int = 768):
        self.cfg = MockSAEConfig(d_sae=d_sae, d_in=d_in)
        self._projection = np.random.RandomState(42).randn(d_in, d_sae).astype(np.float32)

    def encode(self, activations):
        """Encode activations to sparse features using a fixed projection."""
        import torch

        if isinstance(activations, torch.Tensor):
            act_np = activations.cpu().numpy()
        else:
            act_np = np.array(activations)
        # Simple linear projection + ReLU (mimics SAE encoding)
        features = np.maximum(0, act_np @ self._projection)
        return torch.tensor(features, dtype=torch.float32)


class MockCache(dict):
    """Mock TransformerLens activation cache."""

    pass


class MockModel:
    """Mock TransformerLens HookedTransformer."""

    def __init__(self, d_model: int = 768, n_layers: int = 12, seed: int = 0):
        self.d_model = d_model
        self.n_layers = n_layers
        self._rng = np.random.RandomState(seed)

    def run_with_cache(self, text, stop_at_layer=None, **kwargs):
        """Return deterministic activations based on text hash."""
        # Use text hash as seed for reproducible activations
        text_seed = hash(text) % (2**31)
        rng = np.random.RandomState(text_seed)

        import torch

        # Generate activations: [1, seq_len, d_model]
        seq_len = len(text.split()) + 1  # rough token count
        activations = rng.randn(1, seq_len, self.d_model).astype(np.float32)

        cache = MockCache()
        # Populate cache for all possible hook points
        for layer in range(self.n_layers):
            for hook in ["hook_resid_pre", "hook_resid_post", "hook_resid_mid",
                         "hook_mlp_out", "hook_attn_out"]:
                key = f"blocks.{layer}.{hook}"
                cache[key] = torch.tensor(activations)

        logits = torch.zeros(1, seq_len, 50257)
        return logits, cache


# Patch imports so we can test without real torch/sae_lens/transformer_lens
# (torch is available in test env, but sae_lens/transformer_lens may not be)
@pytest.fixture(autouse=True)
def mock_sae_imports(monkeypatch):
    """Ensure sae module thinks SAE dependencies are available."""
    import ro_framework.integration.sae as sae_module

    monkeypatch.setattr(sae_module, "_SAE_AVAILABLE", True)


@pytest.fixture
def mock_model():
    return MockModel(d_model=768, n_layers=12, seed=0)


@pytest.fixture
def mock_sae():
    return MockSAE(d_sae=100, d_in=768)


@pytest.fixture
def label_dofs():
    from ro_framework.core.dof import PolarDoF, ScalarDoF

    return [
        PolarDoF(name="sentiment", description="Positive/negative sentiment",
                 pole_negative=-1.0, pole_positive=1.0),
        ScalarDoF(name="is_code", description="Code probability",
                  min_value=0.0, max_value=1.0),
    ]


@pytest.fixture
def sae_observer(mock_model, mock_sae, label_dofs):
    from ro_framework.integration.sae import SAEObserver

    return SAEObserver(
        model=mock_model,
        sae=mock_sae,
        hook_point="blocks.8.hook_resid_pre",
        label_dofs=label_dofs,
        name="test_sae_observer",
    )


# ─── TestSAEObserverInit ───────────────────────────────────────────────


class TestSAEObserverInit:
    def test_creates_with_auto_feature_dofs(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        obs = SAEObserver(
            model=mock_model,
            sae=mock_sae,
            hook_point="blocks.8.hook_resid_pre",
            label_dofs=label_dofs,
        )
        assert len(obs.feature_dofs) == 100  # matches mock SAE d_sae
        assert obs.feature_dofs[0].name == "sae_feat_0"
        assert obs.n_observations == 0

    def test_creates_with_custom_feature_dofs(self, mock_model, mock_sae, label_dofs):
        from ro_framework.core.dof import ScalarDoF
        from ro_framework.integration.sae import SAEObserver

        custom_dofs = [ScalarDoF(name=f"custom_{i}") for i in range(100)]
        obs = SAEObserver(
            model=mock_model,
            sae=mock_sae,
            hook_point="blocks.8.hook_resid_pre",
            label_dofs=label_dofs,
            feature_dofs=custom_dofs,
        )
        assert obs.feature_dofs[0].name == "custom_0"

    def test_invalid_aggregation_raises(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        with pytest.raises(ValueError, match="aggregation"):
            SAEObserver(
                model=mock_model,
                sae=mock_sae,
                hook_point="blocks.8.hook_resid_pre",
                label_dofs=label_dofs,
                aggregation="invalid",
            )

    def test_parses_layer_from_hook_point(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        obs = SAEObserver(
            model=mock_model,
            sae=mock_sae,
            hook_point="blocks.11.hook_resid_post",
            label_dofs=label_dofs,
        )
        assert obs._stop_at_layer == 12


# ─── TestObserveText ───────────────────────────────────────────────────


class TestObserveText:
    def test_records_observation(self, sae_observer, label_dofs):
        state = sae_observer.observe_text(
            "I love this movie",
            labels={label_dofs[0]: 0.9, label_dofs[1]: 0.0},
        )
        assert sae_observer.n_observations == 1
        assert state is not None

    def test_labels_recorded_in_external_state(self, sae_observer, label_dofs):
        sae_observer.observe_text(
            "I love this movie",
            labels={label_dofs[0]: 0.9, label_dofs[1]: 0.0},
        )
        pair = list(sae_observer.observer.observation_log)[0]
        assert pair.external_state.get_value(label_dofs[0]) == 0.9
        assert pair.external_state.get_value(label_dofs[1]) == 0.0

    def test_features_extracted(self, sae_observer, label_dofs):
        state = sae_observer.observe_text(
            "I love this movie",
            labels={label_dofs[0]: 0.9, label_dofs[1]: 0.0},
        )
        # Internal state should have values for all feature DoFs
        for dof in sae_observer.feature_dofs:
            val = state.get_value(dof)
            assert val is not None
            assert isinstance(val, float)

    def test_features_are_non_negative(self, sae_observer, label_dofs):
        """SAE features after ReLU should be non-negative."""
        state = sae_observer.observe_text(
            "I love this movie",
            labels={label_dofs[0]: 0.9, label_dofs[1]: 0.0},
        )
        for dof in sae_observer.feature_dofs:
            assert state.get_value(dof) >= 0.0

    def test_missing_labels_raises(self, sae_observer, label_dofs):
        with pytest.raises(ValueError, match="Missing labels"):
            sae_observer.observe_text(
                "I love this movie",
                labels={label_dofs[0]: 0.9},  # missing is_code
            )

    def test_different_texts_produce_different_features(self, sae_observer, label_dofs):
        s1 = sae_observer.observe_text(
            "I love this movie",
            labels={label_dofs[0]: 0.9, label_dofs[1]: 0.0},
        )
        s2 = sae_observer.observe_text(
            "def foo(): return 42",
            labels={label_dofs[0]: 0.0, label_dofs[1]: 1.0},
        )
        # At least some features should differ
        vals1 = [s1.get_value(d) for d in sae_observer.feature_dofs]
        vals2 = [s2.get_value(d) for d in sae_observer.feature_dofs]
        assert vals1 != vals2


# ─── TestAggregation ───────────────────────────────────────────────────


class TestAggregation:
    def test_mean_aggregation(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        obs = SAEObserver(
            model=mock_model, sae=mock_sae,
            hook_point="blocks.8.hook_resid_pre",
            label_dofs=label_dofs, aggregation="mean",
        )
        state = obs.observe_text("test", labels={label_dofs[0]: 0.0, label_dofs[1]: 0.0})
        assert state is not None

    def test_last_aggregation(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        obs = SAEObserver(
            model=mock_model, sae=mock_sae,
            hook_point="blocks.8.hook_resid_pre",
            label_dofs=label_dofs, aggregation="last",
        )
        state = obs.observe_text("test", labels={label_dofs[0]: 0.0, label_dofs[1]: 0.0})
        assert state is not None

    def test_max_aggregation(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        obs = SAEObserver(
            model=mock_model, sae=mock_sae,
            hook_point="blocks.8.hook_resid_pre",
            label_dofs=label_dofs, aggregation="max",
        )
        state = obs.observe_text("test", labels={label_dofs[0]: 0.0, label_dofs[1]: 0.0})
        assert state is not None


# ─── TestObserveTexts ──────────────────────────────────────────────────


class TestObserveTexts:
    def test_batch_processing(self, sae_observer, label_dofs):
        texts = ["I love this movie", "I hate this movie", "def foo(): pass"]
        labels = [
            {label_dofs[0]: 0.9, label_dofs[1]: 0.0},
            {label_dofs[0]: -0.9, label_dofs[1]: 0.0},
            {label_dofs[0]: 0.0, label_dofs[1]: 1.0},
        ]
        states = sae_observer.observe_texts(texts, labels)
        assert len(states) == 3
        assert sae_observer.n_observations == 3

    def test_length_mismatch_raises(self, sae_observer, label_dofs):
        with pytest.raises(ValueError, match="same length"):
            sae_observer.observe_texts(
                ["a", "b"],
                [{label_dofs[0]: 0.0, label_dofs[1]: 0.0}],
            )


# ─── TestAssessKnowledge ───────────────────────────────────────────────


class TestAssessKnowledge:
    def test_returns_none_with_insufficient_data(self, sae_observer, label_dofs):
        # Only 3 observations, need 10
        for i in range(3):
            sae_observer.observe_text(
                f"text {i}",
                labels={label_dofs[0]: float(i) / 3, label_dofs[1]: 0.0},
            )
        result = sae_observer.assess_knowledge(label_dofs[0])
        assert result is None

    def test_returns_assessment_with_enough_data(self, sae_observer, label_dofs):
        for i in range(20):
            sae_observer.observe_text(
                f"text number {i} with some content",
                labels={label_dofs[0]: float(i) / 20, label_dofs[1]: 0.0},
            )
        result = sae_observer.assess_knowledge(label_dofs[0])
        assert result is not None
        assert hasattr(result, "correlation")
        assert hasattr(result, "knowledge_type")

    def test_knowledge_type_is_valid(self, sae_observer, label_dofs):
        for i in range(20):
            sae_observer.observe_text(
                f"text number {i} with words",
                labels={label_dofs[0]: float(i) / 20, label_dofs[1]: 0.0},
            )
        result = sae_observer.assess_knowledge(label_dofs[0])
        if result is not None:
            assert result.knowledge_type in ("strong", "weak", "false", "uncertain")


# ─── TestTopFeaturesFor ────────────────────────────────────────────────


class TestTopFeaturesFor:
    def test_returns_sorted_list(self, sae_observer, label_dofs):
        for i in range(30):
            sae_observer.observe_text(
                f"text with content number {i} and more words",
                labels={label_dofs[0]: float(i) / 30, label_dofs[1]: 0.0},
            )
        results = sae_observer.top_features_for(label_dofs[0], n=5)
        assert len(results) <= 5
        # Should be sorted by |correlation| descending
        if len(results) >= 2:
            corrs = [abs(r.assessment.correlation) for r in results]
            assert corrs == sorted(corrs, reverse=True)

    def test_respects_n_parameter(self, sae_observer, label_dofs):
        for i in range(30):
            sae_observer.observe_text(
                f"text with content number {i} and more words",
                labels={label_dofs[0]: float(i) / 30, label_dofs[1]: 0.0},
            )
        results_3 = sae_observer.top_features_for(label_dofs[0], n=3)
        assert len(results_3) <= 3

    def test_returns_empty_with_insufficient_data(self, sae_observer, label_dofs):
        results = sae_observer.top_features_for(label_dofs[0], n=5)
        assert results == []


# ─── TestTopKFeatures ──────────────────────────────────────────────────


class TestTopKFeatures:
    def test_top_k_limits_tracked_features(self, mock_model, mock_sae, label_dofs):
        from ro_framework.integration.sae import SAEObserver

        obs = SAEObserver(
            model=mock_model, sae=mock_sae,
            hook_point="blocks.8.hook_resid_pre",
            label_dofs=label_dofs,
            top_k_features=20,
        )
        assert len(obs.feature_dofs) == 20


# ─── TestCreateFeatureDofs ─────────────────────────────────────────────


class TestCreateFeatureDofs:
    def test_creates_scalar_dofs(self):
        from ro_framework.integration.sae import create_feature_dofs

        dofs = create_feature_dofs(50, prefix="feat")
        assert len(dofs) == 50
        assert dofs[0].name == "feat_0"
        assert dofs[49].name == "feat_49"

    def test_dofs_are_scalar(self):
        from ro_framework.core.dof import ScalarDoF
        from ro_framework.integration.sae import create_feature_dofs

        dofs = create_feature_dofs(10)
        for dof in dofs:
            assert isinstance(dof, ScalarDoF)


# ─── TestEdgeCases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_word_text(self, sae_observer, label_dofs):
        state = sae_observer.observe_text(
            "hello",
            labels={label_dofs[0]: 0.0, label_dofs[1]: 0.0},
        )
        assert state is not None

    def test_observer_property(self, sae_observer):
        obs = sae_observer.observer
        assert obs is not None
        assert obs.name == "test_sae_observer"

    def test_layer_parsing(self):
        from ro_framework.integration.sae import SAEObserver

        assert SAEObserver._parse_layer("blocks.8.hook_resid_pre") == 8
        assert SAEObserver._parse_layer("blocks.11.hook_resid_post") == 11
        assert SAEObserver._parse_layer("blocks.0.hook_mlp_out") == 0

    def test_invalid_hook_point_raises(self):
        from ro_framework.integration.sae import SAEObserver

        with pytest.raises(ValueError, match="Could not parse"):
            SAEObserver._parse_layer("no_layer_here")
