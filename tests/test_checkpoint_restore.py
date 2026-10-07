import copy
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_lewm_checkpoint import restore_models
from mpcrl.config import load_yaml
from mpcrl.lewm import LeWorldModel, RewardProbe
from mpcrl.sac import ResidualSAC


class TestCheckpointRestore(unittest.TestCase):
    def bundle(self, paired):
        cfg = load_yaml(Path(__file__).resolve().parents[1] / 'configs/packaging/lewm_smoke.yaml')
        model = LeWorldModel(
            2, latent_dim=24, hidden=32, sigreg_lambda=.001,
            sigreg_projections=8, sigreg_knots=5, image_size=32, patch_size=16,
            encoder_depth=1, predictor_depth=1, history=2 if paired else 1,
        )
        probe = RewardProbe(24, 2)
        bundle = {'config': cfg, 'lewm': copy.deepcopy(model.state_dict()),
                  'reward_probe': copy.deepcopy(probe.state_dict())}
        if paired:
            bundle['pair_summary'] = {'shared_world_model': True}
        return bundle, model

    def test_standalone_recovers_actual_history_not_saved_yaml_value(self):
        bundle, original = self.bundle(False)
        cfg, model, probe, agent = restore_models(bundle, 2, torch.device('cpu'))
        self.assertEqual(cfg['lewm']['history'], 2)
        self.assertEqual(model.history, 1)
        self.assertIsNone(agent)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in probe.parameters()))
        frame = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        torch.testing.assert_close(model.encode(frame), original.eval().encode(frame))

    def test_paired_restores_actor_and_frozen_world_model(self):
        bundle, _ = self.bundle(True)
        source_agent = ResidualSAC(26, 2, hidden=32)
        bundle['residual_sac'] = copy.deepcopy(source_agent.state_dict())
        _, model, _, agent = restore_models(bundle, 2, torch.device('cpu'))
        self.assertEqual(model.history, 2)
        context = np.linspace(-1, 1, 26, dtype=np.float32)
        np.testing.assert_allclose(agent.act(context, deterministic=True),
                                   source_agent.act(context, deterministic=True), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
