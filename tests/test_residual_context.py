import unittest

import numpy as np

from mpcrl.residual_context import gated_strength, residual_context


class T(unittest.TestCase):
    def test_context_contains_effective_gate(self):
        obs = np.array([1.0, 2.0], np.float32)
        chunk = np.array([[3.0, 4.0], [5.0, 6.0]], np.float32)
        c = residual_context(obs, chunk, 0.25)
        self.assertEqual(c.shape, (7,))
        self.assertAlmostEqual(float(c[-1]), 0.25, places=6)

    def test_gate_power(self):
        self.assertAlmostEqual(gated_strength(0.5, 0.8, 1.0), 0.4, places=6)
        self.assertAlmostEqual(gated_strength(0.5, 0.8, 2.0), 0.2, places=6)


if __name__ == '__main__':
    unittest.main()
