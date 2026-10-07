import unittest

import torch

from mpcrl.lewm import SIGReg


class TestSIGReg(unittest.TestCase):
    def test_finite_and_grad(self):
        torch.manual_seed(0)
        z = torch.randn(64, 8, requires_grad=True)
        loss = SIGReg(knots=9, num_proj=32)(z)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(z.grad)

    def test_penalises_anisotropy_and_collapse_on_raw_embeddings(self):
        """The whole point of the regulariser. The old version, fed BatchNormed
        embeddings, ranked anisotropic and uniform distributions as BETTER than a
        true N(0,I); on raw embeddings the statistic separates them cleanly."""
        torch.manual_seed(0)
        sig = SIGReg(knots=17, num_proj=256)
        n, d = 256, 64

        def score(z):
            return float(sum(sig(z) for _ in range(6)) / 6)

        ideal = score(torch.randn(n, d))
        aniso = score(torch.randn(n, d) @ torch.diag(torch.linspace(1.0, 0.001, d)))
        collapse = score(torch.randn(n, 1) * torch.ones(1, d) + 0.01 * torch.randn(n, d))

        self.assertGreater(aniso, ideal * 2, f'anisotropic {aniso} vs ideal {ideal}')
        self.assertGreater(collapse, ideal * 2, f'rank-1 {collapse} vs ideal {ideal}')

    def test_ideal_gaussian_scores_near_one(self):
        """Calibration anchor: with the Epps-Pulley N factor a true N(0,I) sits
        near 1.0 at any batch size, so a training log whose sigreg plateaus near
        1.0 has met the isotropy target."""
        torch.manual_seed(0)
        sig = SIGReg(knots=17, num_proj=256)
        for n in (128, 512, 2048):
            val = float(sum(sig(torch.randn(n, 64)) for _ in range(6)) / 6)
            self.assertGreater(val, 0.3, f'N={n}')
            self.assertLess(val, 3.0, f'N={n}')

    def test_scaled_form_is_batch_size_invariant(self):
        """The N factor is what makes lambda mean the same thing across batch
        sizes; the unscaled statistic decays as 1/N."""
        torch.manual_seed(0)
        scaled = SIGReg(knots=17, num_proj=512, scale_by_n=True)
        small = float(sum(scaled(torch.randn(128, 32)) for _ in range(6)) / 6)
        large = float(sum(scaled(torch.randn(2048, 32)) for _ in range(6)) / 6)
        self.assertLess(abs(small - large) / max(small, large), 0.5,
                        f'scaled form drifted with N: {small} vs {large}')

        plain = SIGReg(knots=17, num_proj=512, scale_by_n=False)
        small_p = float(sum(plain(torch.randn(128, 32)) for _ in range(6)) / 6)
        large_p = float(sum(plain(torch.randn(2048, 32)) for _ in range(6)) / 6)
        self.assertGreater(small_p / large_p, 4.0,
                           'unscaled form should decay roughly as 1/N')

    def test_scale_by_n_is_exactly_the_n_multiple(self):
        torch.manual_seed(0)
        plain = SIGReg(knots=17, num_proj=256, scale_by_n=False)
        scaled = SIGReg(knots=17, num_proj=256, scale_by_n=True)
        z = torch.randn(128, 32)
        torch.manual_seed(1)
        a = float(plain(z))
        torch.manual_seed(1)
        b = float(scaled(z))
        self.assertAlmostEqual(b / a, 128.0, delta=1.0)


if __name__ == '__main__':
    unittest.main()
