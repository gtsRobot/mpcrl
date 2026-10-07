import unittest
import numpy as np
import torch
from mpcrl.planning_residual import expand_temporal_residual, safeguard_residual_plan


class DummyWorldModel:
    def rollout_return(self, obs, action_sequences, discount, uncertainty_penalty):
        x = torch.as_tensor(action_sequences, dtype=torch.float32)
        # Prefer plans close to target action 0.5.
        score = -((x - 0.5) ** 2).sum(dim=(1, 2))
        return score, torch.zeros_like(score)


class T(unittest.TestCase):
    def test_shape_and_first_action(self):
        r = np.array([1.0, -2.0], dtype=np.float32)
        x = expand_temporal_residual(r, 3, 2, decay=0.5)
        self.assertEqual(x.shape, (3, 2))
        np.testing.assert_allclose(x[0], r)
        np.testing.assert_allclose(x[1], 0.5 * r)
        np.testing.assert_allclose(x[2], 0.25 * r)

    def test_decay_is_bounded(self):
        x = expand_temporal_residual([1.0], 4, 1, decay=2.0)
        np.testing.assert_allclose(x[:, 0], np.ones(4))

    def test_safeguard_rejects_degrading_residual(self):
        base = np.zeros((3, 1), dtype=np.float32)
        delta = np.full((2, 1), -1.0, dtype=np.float32)
        out = safeguard_residual_plan(
            DummyWorldModel(), np.zeros(1), base, delta, [-1], [1], scales=(1.0, 0.5, 0.0)
        )
        self.assertEqual(out.scale, 0.0)
        self.assertEqual(out.predicted_gain, 0.0)
        np.testing.assert_allclose(out.corrected_plan, base)

    def test_safeguard_shrinks_when_partial_is_best(self):
        base = np.zeros((2, 1), dtype=np.float32)
        delta = np.ones((1, 1), dtype=np.float32)
        out = safeguard_residual_plan(
            DummyWorldModel(), np.zeros(1), base, delta, [-1], [1], scales=(1.0, 0.5, 0.0)
        )
        self.assertEqual(out.scale, 0.5)
        self.assertGreater(out.predicted_gain, 0.0)
        self.assertAlmostEqual(float(out.corrected_plan[0, 0]), 0.5)


if __name__ == '__main__':
    unittest.main()
