"""Unit tests for RC-1 Echo State Network experiment."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "reservoir"))
from rc1_modular_addition import (
    EchoStateNetwork,
    make_dataset,
    compute_sum_averaged_reservoir,
    discover_top_frequencies,
    setup_observer_and_tracker,
    populate_observations,
    ridge_regression,
)


class TestESNConstruction:
    def test_buffer_not_parameter(self):
        """W_in and W_res should be buffers, not parameters."""
        esn = EchoStateNetwork(input_dim=10, reservoir_size=8, output_dim=5)
        param_names = [n for n, _ in esn.named_parameters()]
        assert "W_in" not in param_names
        assert "W_res" not in param_names
        assert "bias" not in param_names
        assert "readout_in.weight" in param_names
        assert "readout_in.bias" in param_names
        assert "readout_out.weight" in param_names
        assert "readout_out.bias" in param_names

    def test_spectral_radius(self):
        """W_res should have spectral radius close to requested value."""
        for sr in [0.5, 1.0, 1.5]:
            esn = EchoStateNetwork(
                input_dim=10, reservoir_size=32, output_dim=5, spectral_radius=sr
            )
            eigvals = np.linalg.eigvals(esn.W_res.numpy())
            actual_sr = np.max(np.abs(eigvals))
            assert abs(actual_sr - sr) < 0.01, f"Expected SR={sr}, got {actual_sr}"

    def test_feedforward_shape(self):
        """Feedforward mode produces correct output shapes."""
        esn = EchoStateNetwork(input_dim=10, reservoir_size=8, output_dim=5)
        x = torch.randn(3, 10)
        h = esn.reservoir_state(x)
        assert h.shape == (3, 8)
        logits = esn(x)
        assert logits.shape == (3, 5)

    def test_deterministic_seed(self):
        """Same seed produces identical reservoir weights."""
        esn1 = EchoStateNetwork(10, 8, 5, seed=42)
        esn2 = EchoStateNetwork(10, 8, 5, seed=42)
        assert torch.allclose(esn1.W_in, esn2.W_in)
        assert torch.allclose(esn1.W_res, esn2.W_res)

    def test_different_seed(self):
        """Different seeds produce different weights."""
        esn1 = EchoStateNetwork(10, 8, 5, seed=42)
        esn2 = EchoStateNetwork(10, 8, 5, seed=99)
        assert not torch.allclose(esn1.W_in, esn2.W_in)

    def test_reservoir_state_bounded(self):
        """tanh output should be in [-1, 1]."""
        esn = EchoStateNetwork(input_dim=10, reservoir_size=16, output_dim=5)
        x = torch.randn(20, 10) * 10  # large inputs
        h = esn.reservoir_state(x)
        assert h.min() >= -1.0
        assert h.max() <= 1.0


class TestDataset:
    def test_dataset_shapes(self):
        p = 7
        (train_x, train_y), (test_x, test_y) = make_dataset(p)
        assert train_x.shape[1] == 2 * p
        assert train_y.shape[0] == train_x.shape[0]
        assert test_x.shape[1] == 2 * p
        assert train_x.shape[0] + test_x.shape[0] == p * p

    def test_labels_in_range(self):
        p = 7
        (_, train_y), (_, test_y) = make_dataset(p)
        assert train_y.min() >= 0
        assert train_y.max() < p
        assert test_y.min() >= 0
        assert test_y.max() < p


class TestSumAveraging:
    def test_sum_averaged_shape(self):
        p = 7
        esn = EchoStateNetwork(input_dim=2 * p, reservoir_size=16, output_dim=p)
        h_avg = compute_sum_averaged_reservoir(esn, p)
        assert h_avg.shape == (p, 16)

    def test_discover_frequencies_count(self):
        p = 7
        esn = EchoStateNetwork(input_dim=2 * p, reservoir_size=16, output_dim=p)
        h_avg = compute_sum_averaged_reservoir(esn, p)
        top = discover_top_frequencies(h_avg, p, top_n=3)
        assert len(top) == 3
        assert all(isinstance(k, (int, np.integer)) and isinstance(f, float) for k, f in top)


class TestRidgeRegression:
    def test_ridge_solves_identity(self):
        """Ridge should perfectly solve H @ W = Y when H is full rank."""
        rng = np.random.default_rng(42)
        N, D, C = 50, 10, 3
        H = rng.standard_normal((N, D))
        W_true = rng.standard_normal((D, C))
        Y = H @ W_true
        Y_labels = Y.argmax(axis=1)

        H_test = rng.standard_normal((20, D))
        Y_test_logits = H_test @ W_true
        Y_test = torch.tensor(Y_test_logits.argmax(axis=1), dtype=torch.long)

        train_acc, test_acc, W_out = ridge_regression(H, Y, H_test, Y_test, reg_lambda=1e-8)
        assert train_acc > 0.95
        assert test_acc > 0.9


class TestObserverIntegration:
    def test_setup_and_populate(self):
        """Observer setup + populate + assess_knowledge works end-to-end."""
        p = 7
        reservoir_size = 16
        esn = EchoStateNetwork(input_dim=2 * p, reservoir_size=reservoir_size, output_dim=p)
        h_avg = compute_sum_averaged_reservoir(esn, p)

        freq_indices = [1, 2]
        observer, tracker, fdofs, ndofs = setup_observer_and_tracker(
            freq_indices, reservoir_size, max_features=1
        )
        populate_observations(observer, h_avg, p, fdofs, ndofs, freq_indices)

        assert len(observer.observation_log) == p

        # Assess knowledge should return something (may be weak, that's fine)
        result = tracker.step(0)
        assert len(result) == len(fdofs)
        for dof, assessment in result.items():
            assert assessment is not None
            assert assessment.knowledge_type in ("strong", "weak", "false", "uncertain")

    def test_multi_feature_assessment(self):
        """Multi-feature K should be >= single-feature K."""
        p = 7
        reservoir_size = 32
        esn = EchoStateNetwork(input_dim=2 * p, reservoir_size=reservoir_size, output_dim=p)
        h_avg = compute_sum_averaged_reservoir(esn, p)
        freq_indices = [1]

        # Single feature
        obs1, trk1, fd1, nd1 = setup_observer_and_tracker(freq_indices, reservoir_size, max_features=1)
        populate_observations(obs1, h_avg, p, fd1, nd1, freq_indices)
        trk1.step(0)
        rho_single = trk1.latest(fd1[0]).correlation

        # Multi feature (but not more than n//10)
        obs2, trk2, fd2, nd2 = setup_observer_and_tracker(freq_indices, reservoir_size, max_features=3)
        populate_observations(obs2, h_avg, p, fd2, nd2, freq_indices)
        trk2.step(0)
        rho_multi = trk2.latest(fd2[0]).correlation

        # Multi should be >= single (more features can only help)
        assert rho_multi >= rho_single - 0.01  # small tolerance for numerical noise
