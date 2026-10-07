from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.train import evaluate_final
from mpcrl.config import load_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.gate import make_uncertainty_gate
from mpcrl.sac import ResidualSAC
from mpcrl.utils import configure_accelerator
from mpcrl.world_model import EnsembleWorldModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--episodes', type=int, default=5)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = load_yaml(run / 'config.yaml')

    # LeWM runs are produced by experiments.train_lewm / train_lewm_mpc_rl, which
    # write a completely different config schema (lewm/reward_probe/rl_residual
    # instead of world_model/residual/sac) and evaluate themselves at the end of
    # training. Detect that before subscripting keys that do not exist.
    if 'world_model' not in cfg or 'residual' not in cfg or 'sac' not in cfg:
        raise SystemExit(
            f'{run} does not look like a state-space run: config.yaml has no '
            "'world_model'/'residual'/'sac' block. LeWM runs are evaluated by their own "
            'training entry point and already contain eval.csv / pair_summary.json.'
        )

    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    seed = int(cfg['env']['seed'])

    probe_env, _, _ = make_mujoco_env(cfg['env']['id'], seed)
    od, ad = dims(probe_env)
    low, high = action_bounds(probe_env)
    probe_env.close()

    w = cfg['world_model']
    rc = cfg['residual']
    sc = cfg['sac']

    ck = torch.load(run / 'final.pt', map_location=device)
    method = ck.get('method')
    if method not in {'mpc_only', 'action_residual', 'planning_residual'}:
        raise SystemExit(
            f'Unsupported checkpoint method={method!r}. '
            'ZPRL-style runs are evaluated automatically at the end of training.'
        )

    wm = EnsembleWorldModel(
        od, ad, w['ensemble_size'], w['hidden_dim'], w['lr'],
        w['weight_decay'], device, precision
    )
    wm.load_state_dict(ck['world_model'])
    wm.eval()

    k = 1 if method == 'action_residual' else int(rc['chunk_len'])
    temporal_decay = float(rc.get('temporal_decay', 0.8))
    gate_power = float(rc.get('gate_power', 1.0)) if method == 'planning_residual' else 1.0

    agent = None
    normalized_gate = None
    if method != 'mpc_only':
        context_dim = od + k * ad + 1
        agent = ResidualSAC(
            context_dim,
            ad,
            sc['hidden_dim'],
            sc['lr'],
            sc['gamma'],
            sc['tau'],
            sc['init_alpha'],
            sc['target_entropy_scale'],
            rc.get('consistency_coef', 0.0) if method == 'planning_residual' else 0.0,
            device,
            precision,
        )
        try:
            agent.load_bundle(ck['agent'])
        except RuntimeError as exc:
            raise SystemExit(
                'Checkpoint architecture does not match the current gate-aware '
                'residual controller. This is probably a legacy checkpoint from '
                'before effective_gate was added to the SAC context.'
            ) from exc

        if method == 'planning_residual' and rc.get('adaptive_gate', True):
            normalized_gate = make_uncertainty_gate(rc)
            if normalized_gate is not None and 'gate_state' in ck:
                normalized_gate.load_state_dict(ck['gate_state'])

    rows = evaluate_final(
        cfg,
        method,
        wm,
        agent,
        normalized_gate,
        device,
        low,
        high,
        ad,
        k,
        temporal_decay,
        gate_power,
        int(args.episodes),
        seed,
        show_progress=True,
    )
    out = run / 'eval.csv'
    pd.DataFrame(rows).to_csv(out, index=False)

    vals = np.asarray([r['episode_return'] for r in rows], dtype=float)
    print(f'run={run}')
    print(f'method={method} train_seed={seed} episodes={len(vals)}')
    print(
        f'deterministic_eval_return={vals.mean():.3f} '
        f'std={vals.std(ddof=1) if len(vals) > 1 else 0.0:.3f}'
    )
    print('saved', out)


if __name__ == '__main__':
    main()
