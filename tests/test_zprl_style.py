import unittest

import numpy as np
import torch

from mpcrl.zprl_style import BottleneckBasePolicy, fit_behavior_clone


class TestZPRLStyle(unittest.TestCase):
    def test_observation_normalization_and_refit(self):
        rng = np.random.default_rng(0)
        obs = rng.normal(size=(128, 5)).astype(np.float32)
        obs[:, 0] = obs[:, 0] * 100.0 + 500.0
        target = np.tanh(obs[:, :2] / np.array([500.0, 2.0], dtype=np.float32)).astype(np.float32)

        policy = BottleneckBasePolicy(5, 2, latent_dim=4, hidden=32)
        fit_behavior_clone(
            policy,
            obs,
            target,
            epochs=2,
            batch_size=32,
            device='cpu',
            show_progress=False,
        )

        x = torch.as_tensor(obs)
        xn = policy.normalize_obs(x)
        self.assertTrue(torch.isfinite(xn).all())
        self.assertLess(float(torch.abs(xn.mean(dim=0)).max()), 0.05)
        self.assertEqual(tuple(policy(x[:3]).shape), (3, 2))
        self.assertTrue(all(not p.requires_grad for p in policy.parameters()))

        # Refit must temporarily unfreeze the policy and freeze it again.
        fit_behavior_clone(
            policy,
            obs[:64],
            target[:64],
            epochs=1,
            batch_size=32,
            device='cpu',
            show_progress=False,
        )
        self.assertTrue(all(not p.requires_grad for p in policy.parameters()))


if __name__ == '__main__':
    unittest.main()
