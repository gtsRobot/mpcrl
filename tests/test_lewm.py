import unittest

import numpy as np
import torch

from mpcrl.lewm import LeWorldModel, ViTEncoder, ARPredictor, RewardProbe


class TestLeWorldModel(unittest.TestCase):
    def _model(self, **kw):
        base = dict(action_dim=2, latent_dim=24, image_size=32, patch_size=8,
                    encoder_depth=2, encoder_heads=3, predictor_depth=2,
                    predictor_heads=3, history=2, sigreg_projections=16, sigreg_knots=5)
        base.update(kw)
        ad = base.pop('action_dim')
        return LeWorldModel(ad, **base)

    def _frames(self, n=8, size=32):
        return torch.randint(0, 256, (n, 3, size, size), dtype=torch.uint8)

    def test_loss_backward_vit_ar(self):
        torch.manual_seed(0)
        m = self._model()
        hist = torch.randint(0, 256, (8, 1, 3, 32, 32), dtype=torch.uint8)
        loss, info = m.loss(self._frames(), torch.randn(8, 2), self._frames(), history=hist)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn('sigreg', info)
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_loss_backward_cnn_mlp_ablation(self):
        torch.manual_seed(0)
        m = self._model(encoder='cnn', predictor='mlp')
        loss, info = m.loss(self._frames(), torch.randn(8, 2), self._frames())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_encoder_emits_patch_tokens(self):
        m = self._model()
        z = m.encode(self._frames(4))
        self.assertEqual(z.dim(), 3)                 # [B, tokens, D]
        self.assertEqual(z.shape[1], m.tokens)
        self.assertEqual(z.shape[1], (32 // 8) ** 2)

    def test_no_batchnorm_anywhere(self):
        """The terminal BatchNorm1d is what made SIGReg vacuous; it must not return."""
        m = self._model()
        offenders = [type(mod).__name__ for mod in m.modules()
                     if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
        self.assertEqual(offenders, [])

    def test_batch_size_one_works_in_train_mode(self):
        """BatchNorm1d raised 'Expected more than 1 value per channel' here, which
        is exactly what the planner does when encoding a single frame."""
        m = self._model()
        m.train()
        self.assertEqual(m.encode(self._frames(1)).shape[0], 1)

    def test_latent_is_not_rank_capped(self):
        """Linear(128, latent_dim) used to cap the latent at a 128-d subspace
        regardless of the configured latent_dim."""
        m = self._model(latent_dim=48, encoder_depth=1)
        with torch.no_grad():
            z = m.pooled(m.encode(self._frames(200)))
        x = (z - z.mean(0, keepdim=True)).double().numpy()
        sv = np.linalg.svd(x, compute_uv=False)
        self.assertGreater(int((sv > sv.max() * 1e-8).sum()), 40)

    def test_mismatched_heads_gives_an_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._model(latent_dim=16, encoder_heads=3)
        self.assertIn('divisible', str(ctx.exception))

    def test_reward_probe_accepts_token_latents(self):
        m = self._model()
        probe = RewardProbe(24, 2)
        out = probe(m.encode(self._frames(5)), torch.randn(5, 2))
        self.assertEqual(tuple(out.shape), (5, 1))


class TestARPredictorCausality(unittest.TestCase):
    def test_mask_is_block_causal(self):
        p = ARPredictor(latent_dim=24, action_dim=2, tokens=4, depth=1, heads=3, history=3)
        m = p.mask
        frame = torch.arange(m.shape[0]) // 4
        # newest frame may look back at the oldest
        self.assertFalse(m[frame == 2][:, frame == 0].any())
        # the oldest frame may not look forward
        self.assertTrue(m[frame == 0][:, frame == 2].all())

    def test_adaln_zero_starts_as_identity_in_the_action_path(self):
        torch.manual_seed(0)
        p = ARPredictor(latent_dim=24, action_dim=2, tokens=4, depth=2, heads=3, history=2)
        z = torch.randn(3, 4, 24)
        h = torch.randn(3, 1, 4, 24)
        a0 = p(z, torch.zeros(3, 2), history=h)
        a1 = p(z, torch.full((3, 2), 5.0), history=h)
        self.assertLess(float((a0 - a1).abs().max()), 1e-6)

    def test_output_shape_matches_one_frame_of_tokens(self):
        p = ARPredictor(latent_dim=24, action_dim=2, tokens=4, depth=1, heads=3, history=2)
        out = p(torch.randn(3, 4, 24), torch.randn(3, 2), history=torch.randn(3, 1, 4, 24))
        self.assertEqual(tuple(out.shape), (3, 4, 24))

    def test_history_of_one_needs_no_history_tensor(self):
        p = ARPredictor(latent_dim=24, action_dim=2, tokens=4, depth=1, heads=3, history=1)
        out = p(torch.randn(3, 4, 24), torch.randn(3, 2))
        self.assertEqual(tuple(out.shape), (3, 4, 24))


if __name__ == '__main__':
    unittest.main()
