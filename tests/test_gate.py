import unittest
from mpcrl.gate import adaptive_uncertainty_gate, RunningUncertaintyGate, residual_ramp


class T(unittest.TestCase):
    def test_legacy_monotonic(self):
        self.assertLess(adaptive_uncertainty_gate(0.0), adaptive_uncertainty_gate(0.1))

    def test_normalized_gate_tracks_relative_uncertainty(self):
        gate = RunningUncertaintyGate(momentum=0.9, initial_std=0.05, z_threshold=0.5, z_temperature=0.75)
        for _ in range(30):
            gate(0.10)
        low = gate(0.08, update=False)
        high = gate(0.16, update=False)
        self.assertLess(low.value, high.value)
        self.assertGreaterEqual(low.value, 0.05)
        self.assertLessEqual(high.value, 0.95)

    def test_ramp(self):
        self.assertEqual(residual_ramp(9, 10, 5), 0.0)
        self.assertAlmostEqual(residual_ramp(10, 10, 5), 0.2)
        self.assertEqual(residual_ramp(14, 10, 5), 1.0)
        self.assertEqual(residual_ramp(100, 10, 5), 1.0)


if __name__ == '__main__':
    unittest.main()
