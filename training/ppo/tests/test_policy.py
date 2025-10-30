"""
Unit tests for EPDParamPolicy.

These tests validate tensor shapes, log-probability consistency, and the
round-trip behaviour between Dirichlet means and recovered parameter tables.

Do not execute automatically in this environment; run manually via:
    python -m training.ppo.tests.test_policy
"""

import unittest

import numpy as np
import torch

from training.ppo.cold_start import DirichletInit
from training.ppo.policy import EPDParamPolicy, PolicyOutput


class PolicyShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)

        alpha_pos = np.full((5, 3), 10.0, dtype=np.float64)
        alpha_weight = np.full((5, 2), 12.0, dtype=np.float64)
        self.dirichlet_init = DirichletInit(
            alpha_pos=alpha_pos,
            alpha_weight=alpha_weight,
            mean_pos_segments=alpha_pos / alpha_pos.sum(axis=-1, keepdims=True),
            mean_weights=alpha_weight / alpha_weight.sum(axis=-1, keepdims=True),
            invalid_pos_rows=np.zeros(5, dtype=bool),
            invalid_weight_rows=np.zeros(5, dtype=bool),
            concentration=10.0,
        )

    def _make_policy(self) -> EPDParamPolicy:
        return EPDParamPolicy(
            num_steps=6,
            num_points=2,
            hidden_dim=32,
            num_layers=2,
            context_dim=4,
            dirichlet_init=self.dirichlet_init,
        )

    def test_forward_shapes(self) -> None:
        policy = self._make_policy()
        step_indices = torch.tensor([0, 1, 4], dtype=torch.long)
        context = torch.randn(3, policy.context_dim)

        output = policy(step_indices, context=context)
        self.assertEqual(output.alpha_pos.shape, (3, 3))
        self.assertEqual(output.alpha_weight.shape, (3, 2))
        self.assertTrue(torch.all(output.alpha_pos > 0))
        self.assertTrue(torch.all(output.alpha_weight > 0))

    def test_mean_roundtrip(self) -> None:
        policy = self._make_policy()
        step_indices = torch.arange(5, dtype=torch.long)
        output = policy(step_indices)
        mean_segments = policy._normalize(output.alpha_pos)
        _, mean_weights = policy.mean_table(output)
        np.testing.assert_allclose(
            mean_segments.detach().cpu().numpy(),
            self.dirichlet_init.mean_pos_segments,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            mean_weights.detach().cpu().numpy(),
            self.dirichlet_init.mean_weights,
            atol=1e-4,
        )


class PolicySamplingTest(unittest.TestCase):
    def test_log_prob_matches_distribution(self) -> None:
        torch.manual_seed(123)
        policy = EPDParamPolicy(num_steps=4, num_points=2)
        step_indices = torch.tensor([0, 1], dtype=torch.long)
        output = policy(step_indices)

        sample = policy.sample_table(output)
        dir_pos = torch.distributions.Dirichlet(output.alpha_pos)
        dir_weight = torch.distributions.Dirichlet(output.alpha_weight)
        manual_log_prob = dir_pos.log_prob(sample.segments) + dir_weight.log_prob(sample.weights)

        torch.testing.assert_close(sample.log_prob, manual_log_prob, atol=1e-6, rtol=1e-6)

    def test_autograd_flow(self) -> None:
        torch.manual_seed(321)
        policy = EPDParamPolicy(num_steps=3, num_points=2)
        step_indices = torch.tensor([0, 1], dtype=torch.long)
        output = policy(step_indices)
        sample = policy.sample_table(output)

        loss = sample.log_prob.mean()
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(any(torch.any(torch.isfinite(g)) for g in grads))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
