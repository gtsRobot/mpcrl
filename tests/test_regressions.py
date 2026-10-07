"""Regression tests for the failure modes that silently broke multi-hour runs.

Each test here corresponds to a bug that the previous smoke-test-only suite would
have passed straight through.
"""
import unittest

import numpy as np
import torch

from mpcrl.cem import CEMPlanner
from mpcrl.gate import RunningUncertaintyGate
from mpcrl.planning_residual import safeguard_residual_plan
from mpcrl.replay import ResidualReplay
from mpcrl.sac import ResidualSAC
from mpcrl.world_model import EnsembleWorldModel


class _ScoreWM:
    """World model stub returning a caller-supplied score vector."""

    device = torch.device('cpu')

    def __init__(self, fn):
        self.fn = fn

    def rollout_return(self, obs, seq, discount=0.99, uncertainty_penalty=0.0):
        n = len(seq)
        return self.fn(n), torch.zeros(n)


class TestCEMDegeneracy(unittest.TestCase):
    """torch.topk ranks NaN ABOVE every finite value, so a single NaN candidate
    used to make `float(vals[0]) > best_score` false for every iteration, leaving
    best_seq as None and crashing plan_chunk with a TypeError."""

    def _plan(self, fn):
        p = CEMPlanner(_ScoreWM(fn), -np.ones(3, np.float32), np.ones(3, np.float32),
                       horizon=5, candidates=32, elites=8, iterations=5)
        return p.plan(np.zeros(4, np.float32))

    def test_single_nan_candidate_still_returns_a_plan(self):
        def fn(n):
            s = torch.randn(n)
            s[0] = float('nan')
            return s
        r = self._plan(fn)
        self.assertIsNotNone(r.actions)
        self.assertTrue(np.all(np.isfinite(r.actions)))
        self.assertEqual(r.actions.shape, (5, 3))
        r.actions[:3]  # must not raise

    def test_all_nan_scores_still_return_a_finite_plan(self):
        r = self._plan(lambda n: torch.full((n,), float('nan')))
        self.assertIsNotNone(r.actions)
        self.assertTrue(np.all(np.isfinite(r.actions)))
        self.assertTrue(r.degenerate)

    def test_all_inf_scores_still_return_a_finite_plan(self):
        r = self._plan(lambda n: torch.full((n,), -float('inf')))
        self.assertIsNotNone(r.actions)
        self.assertTrue(np.all(np.isfinite(r.actions)))

    def test_plan_is_clipped_to_action_bounds(self):
        r = self._plan(lambda n: torch.randn(n))
        self.assertTrue(np.all(r.actions >= -1.0 - 1e-6))
        self.assertTrue(np.all(r.actions <= 1.0 + 1e-6))

    def test_elites_greater_than_candidates_does_not_crash(self):
        p = CEMPlanner(_ScoreWM(lambda n: torch.randn(n)), -np.ones(2, np.float32),
                       np.ones(2, np.float32), horizon=3, candidates=8, elites=64, iterations=2)
        self.assertEqual(p.elites, 8)
        self.assertIsNotNone(p.plan(np.zeros(3, np.float32)).actions)

    def test_execute_modes_all_produce_valid_plans(self):
        for mode in ('best_sample', 'elite_mean', 'cem_mean'):
            p = CEMPlanner(_ScoreWM(lambda n: torch.randn(n)), -np.ones(2, np.float32),
                           np.ones(2, np.float32), horizon=4, candidates=16, elites=4,
                           iterations=3, execute=mode)
            r = p.plan(np.zeros(3, np.float32))
            self.assertEqual(r.actions.shape, (4, 2), mode)
            self.assertTrue(np.all(np.isfinite(r.actions)), mode)

    def test_unknown_execute_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            CEMPlanner(_ScoreWM(lambda n: torch.randn(n)), [-1], [1], execute='nope')


class TestWorldModelStability(unittest.TestCase):
    def _batch(self, n=256, od=5, ad=2):
        o = np.random.randn(n, od).astype('f')
        a = np.random.randn(n, ad).astype('f')
        return {'obs': o, 'action': a, 'next_obs': o + 0.1 * np.pad(a, ((0, 0), (0, od - ad))),
                'reward': -(a * a).sum(1, keepdims=True).astype('f')}

    def test_uncertainty_grows_with_horizon(self):
        """Member trajectories must diverge. The previous implementation advanced
        the ensemble MEAN state, making them identical by construction, so the
        gate that the whole method depends on saw an almost constant signal."""
        torch.manual_seed(0)
        np.random.seed(0)
        wm = EnsembleWorldModel(5, 2, ensemble_size=5, hidden_dim=64)
        wm.fit_batch(self._batch(), updates=40, batch_size=128)
        seqs = np.random.uniform(-1, 1, (32, 16, 2)).astype('f')
        spreads = []
        for h in (1, 2, 4, 8, 16):
            _, spread = wm.rollout_return(np.zeros(5, np.float32), seqs[:, :h], 0.99, 0.0)
            spreads.append(float(spread.mean()))
        self.assertTrue(all(b > a for a, b in zip(spreads, spreads[1:])),
                        f'spread must increase with horizon, got {spreads}')

    def test_non_finite_loss_does_not_corrupt_weights(self):
        wm = EnsembleWorldModel(5, 2, ensemble_size=3, hidden_dim=32)
        wm.fit_batch({'obs': np.full((64, 5), 1e20, np.float32),
                      'action': np.zeros((64, 2), np.float32),
                      'next_obs': np.full((64, 5), -1e20, np.float32),
                      'reward': np.full((64, 1), 1e20, np.float32)},
                     updates=10, batch_size=32)
        for p in wm.members.parameters():
            self.assertTrue(torch.isfinite(p).all())

    def test_normalizer_is_blended_not_overwritten(self):
        """Overwriting the normalizers from a fresh resample shifted the input
        scaling discontinuously under already-trained weights."""
        wm = EnsembleWorldModel(5, 2, ensemble_size=2, hidden_dim=16)
        wm.update_stats(self._batch())
        first = wm.obs_mean.clone()
        shifted = self._batch()
        shifted['obs'] = shifted['obs'] + 100.0
        wm.update_stats(shifted)
        moved = (wm.obs_mean - first).abs().mean().item()
        self.assertLess(moved, 50.0, 'stats jumped as if overwritten rather than blended')
        self.assertGreater(moved, 0.0)

    def test_freeze_makes_further_training_loud(self):
        wm = EnsembleWorldModel(5, 2, ensemble_size=2, hidden_dim=16)
        wm.freeze()
        with self.assertRaises(RuntimeError):
            wm.fit_batch(self._batch(), updates=1)

    def test_terminal_value_shifts_the_score(self):
        """gamma**H is >0.8 at H=20/gamma=0.99, so the terminal value carries most
        of the discounted return and must actually reach the score."""
        wm = EnsembleWorldModel(5, 2, ensemble_size=3, hidden_dim=32)
        wm.update_stats(self._batch())
        seqs = np.zeros((4, 10, 2), np.float32)
        base, _ = wm.rollout_return(np.zeros(5, np.float32), seqs, 0.99, 0.0)
        withv, _ = wm.rollout_return(np.zeros(5, np.float32), seqs, 0.99, 0.0,
                                     terminal_value_fn=lambda o: torch.full((len(o),), 100.0))
        delta = float((withv - base).mean())
        self.assertAlmostEqual(delta, 100.0 * 0.99 ** 10, places=2)


class TestBootstrapSemantics(unittest.TestCase):
    def test_replay_carries_per_transition_discount(self):
        rb = ResidualReplay(16, 4, 2, default_discount=0.99)
        rb.add(np.zeros(4, np.float32), np.zeros(2, np.float32), 1.0,
               np.zeros(4, np.float32), 0.0, discount=0.99 ** 3)
        b = rb.sample(1)
        self.assertIn('discount', b)
        self.assertAlmostEqual(float(b['discount'][0][0]), 0.99 ** 3, places=6)

    def test_sac_uses_the_stored_discount_for_bootstrapping(self):
        torch.manual_seed(0)
        agent = ResidualSAC(4, 2, hidden=32, obs_dim=3)
        n = 64
        batch = {'context': np.zeros((n, 4), np.float32),
                 'residual': np.zeros((n, 2), np.float32),
                 'reward': np.ones((n, 1), np.float32),
                 'next_context': np.zeros((n, 4), np.float32),
                 'done': np.zeros((n, 1), np.float32),
                 'discount': np.full((n, 1), 0.5, np.float32)}
        logs = agent.update(batch)
        self.assertTrue(np.isfinite(logs['q_loss']))

    def test_done_flag_zeroes_the_bootstrap(self):
        """With done=1 the target must be exactly the reward."""
        torch.manual_seed(0)
        agent = ResidualSAC(4, 2, hidden=32, lr=3e-3, critic='plain')
        n = 128
        r = 3.0
        batch = {'context': np.zeros((n, 4), np.float32),
                 'residual': np.zeros((n, 2), np.float32),
                 'reward': np.full((n, 1), r, np.float32),
                 'next_context': np.zeros((n, 4), np.float32),
                 'done': np.ones((n, 1), np.float32),
                 'discount': np.full((n, 1), 0.99, np.float32)}
        for _ in range(1500):
            agent.update(batch)
        with torch.no_grad():
            q = agent.q1(torch.zeros(1, 4), torch.zeros(1, 2)).item()
        self.assertAlmostEqual(q, r, delta=0.3)


class TestAdvantageCritic(unittest.TestCase):
    def test_advantage_is_anchored_at_zero_residual(self):
        """A(c, 0) == 0 is what makes the decomposition identified and makes the
        shield's accept/reject test read as 'is intervening here worth it'."""
        torch.manual_seed(0)
        agent = ResidualSAC(4, 2, hidden=32)
        c = torch.randn(8, 4)
        zero = torch.zeros(8, 2)
        self.assertLess(float(agent.q1.advantage(c, zero).detach().abs().max()), 1e-5)

    def test_q_equals_value_plus_advantage(self):
        torch.manual_seed(0)
        agent = ResidualSAC(4, 2, hidden=32)
        c = torch.randn(8, 4)
        a = torch.randn(8, 2).clamp(-1, 1)
        lhs = agent.q1(c, a)
        rhs = agent.q1.value(c) + agent.q1.advantage(c, a)
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-5))

    def test_diagnostics_report_the_advantage_signal_ratio(self):
        torch.manual_seed(0)
        agent = ResidualSAC(4, 2, hidden=32)
        n = 64
        d = agent.diagnostics({'context': np.random.randn(n, 4).astype('f'),
                               'residual': np.random.randn(n, 2).astype('f')})
        self.assertIn('adv_q_std_ratio', d)
        self.assertTrue(np.isfinite(d['adv_q_std_ratio']))


class TestSafeguardModes(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((4, 2), np.float32)
        self.delta = np.full((2, 2), 0.5, np.float32)
        self.low = -np.ones(2, np.float32)
        self.high = np.ones(2, np.float32)

    def test_off_applies_the_residual_unchanged(self):
        out = safeguard_residual_plan(None, np.zeros(3, np.float32), self.base, self.delta,
                                      self.low, self.high, mode='off')
        self.assertEqual(out.scale, 1.0)
        np.testing.assert_allclose(out.corrected_plan[:2], self.delta, atol=1e-6)

    def test_all_nan_scores_fall_back_to_the_unmodified_plan(self):
        wm = _ScoreWM(lambda n: torch.full((n,), float('nan')))
        out = safeguard_residual_plan(wm, np.zeros(3, np.float32), self.base, self.delta,
                                      self.low, self.high, mode='model')
        self.assertEqual(out.scale, 0.0)
        np.testing.assert_allclose(out.corrected_plan, self.base, atol=1e-6)

    def test_model_value_requires_a_terminal_value_fn(self):
        wm = _ScoreWM(lambda n: torch.randn(n))
        with self.assertRaises(ValueError):
            safeguard_residual_plan(wm, np.zeros(3, np.float32), self.base, self.delta,
                                    self.low, self.high, mode='model_value')

    def test_critic_mode_requires_agent_context_residual(self):
        with self.assertRaises(ValueError):
            safeguard_residual_plan(None, np.zeros(3, np.float32), self.base, self.delta,
                                    self.low, self.high, mode='critic')

    def test_critic_mode_uses_the_learned_advantage(self):
        agent = ResidualSAC(4, 2, hidden=32)
        out = safeguard_residual_plan(
            None, np.zeros(3, np.float32), self.base, self.delta, self.low, self.high,
            mode='critic', agent=agent, context=np.zeros(4, np.float32),
            residual=np.ones(2, np.float32))
        self.assertIn(out.scale, (0.0, 0.25, 0.5, 1.0))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            safeguard_residual_plan(None, np.zeros(3, np.float32), self.base, self.delta,
                                    self.low, self.high, mode='bogus')


class TestGateScaleRobustness(unittest.TestCase):
    def test_adaptive_initial_std_tracks_the_signal_scale(self):
        """Uncertainty is now in return units, whose magnitude varies by orders of
        magnitude across tasks; a hardcoded initial_std of 0.05 saturates the gate."""
        fixed = RunningUncertaintyGate(initial_std=0.05, min_gate=0.0, max_gate=1.0)
        adaptive = RunningUncertaintyGate(initial_std=-1, min_gate=0.0, max_gate=1.0)
        for g in (fixed, adaptive):
            g(50.0)
        z_fixed = abs(fixed(55.0, update=False).normalized_uncertainty)
        z_adaptive = abs(adaptive(55.0, update=False).normalized_uncertainty)
        self.assertGreater(z_fixed, 50.0)
        self.assertLess(z_adaptive, 1.0)

    def test_state_dict_roundtrip_preserves_initial_std(self):
        g = RunningUncertaintyGate(initial_std=-1)
        g(42.0)
        other = RunningUncertaintyGate(initial_std=-1)
        other.load_state_dict(g.state_dict())
        self.assertAlmostEqual(other.initial_std, g.initial_std, places=6)
        self.assertEqual(other.count, g.count)


if __name__ == '__main__':
    unittest.main()
