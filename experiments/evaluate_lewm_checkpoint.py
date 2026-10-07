"""Evaluate existing LeWM checkpoints without collecting data or retraining.

This packaging entry point reuses the archived controller/evaluation semantics.
It does not repair the research-protocol issues listed in docs/PACKAGE_STATUS.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from experiments.train_lewm_mpc_rl import build_planner, evaluate_controller
from mpcrl.config import save_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.lewm import LeWorldModel, RewardProbe
from mpcrl.sac import ResidualSAC
from mpcrl.utils import configure_accelerator, set_seed


def restore_models(bundle, action_dim, device):
    """Restore the two checkpoint formats actually present in the source project."""
    required = {'config', 'lewm', 'reward_probe'}
    if not required.issubset(bundle):
        raise ValueError('Expected a LeWM checkpoint with config, lewm and reward_probe.')
    cfg = copy.deepcopy(bundle['config'])
    lc = cfg['lewm']
    paired = 'pair_summary' in bundle
    # train_lewm.py explicitly used history=1 even when YAML said history=2.
    history = int(lc.get('history', 2)) if paired else 1
    model = LeWorldModel(
        action_dim, latent_dim=lc['latent_dim'], hidden=lc.get('hidden_dim', 256),
        sigreg_lambda=lc['sigreg_lambda'], sigreg_projections=lc['sigreg_projections'],
        sigreg_knots=lc['sigreg_knots'], encoder=lc.get('encoder', 'vit'),
        predictor=lc.get('predictor', 'ar_transformer'),
        image_size=int(cfg['env']['render_size']), patch_size=int(lc.get('patch_size', 16)),
        encoder_depth=int(lc.get('encoder_depth', 6)), encoder_heads=int(lc.get('encoder_heads', 3)),
        predictor_depth=int(lc.get('predictor_depth', 4)), predictor_heads=int(lc.get('predictor_heads', 3)),
        history=history, sigreg_scale_by_n=bool(lc.get('sigreg_scale_by_n', paired)),
    ).to(device)
    model.load_state_dict(bundle['lewm'], strict=True)
    probe = RewardProbe(lc['latent_dim'], action_dim).to(device)
    probe.load_state_dict(bundle['reward_probe'], strict=True)
    model.eval().requires_grad_(False)
    probe.eval().requires_grad_(False)
    agent = None
    if 'residual_sac' in bundle:
        rc = cfg['rl_residual']
        agent = ResidualSAC(
            context_dim=int(lc['latent_dim']) + action_dim, residual_dim=action_dim,
            hidden=int(rc.get('hidden_dim', 512)),
            lr=float(rc.get('lr', 3e-4)), gamma=float(rc.get('gamma', .99)),
            tau=float(rc.get('tau', .005)), init_alpha=float(rc.get('init_alpha', .1)),
            target_entropy_scale=float(rc.get('target_entropy_scale', .7)),
            consistency_coef=float(rc.get('consistency_coef', .05)),
            critic=bundle['residual_sac'].get('critic_kind', 'dueling'),
            device=device, precision=cfg.get('hardware', {}).get('precision', 'fp32'),
        )
        agent.load_bundle(bundle['residual_sac'])
    return cfg, model, probe, agent


def finite_json(value):
    import math
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--method', choices=['auto', 'baseline', 'rl', 'both'], default='auto')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--inspect-only', action='store_true', help='Load/validate weights without running rollouts.')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint missing. Weights are excluded from this GitHub package; see docs/WORKFLOW.md.')
    if not args.inspect_only and args.outdir is None:
        parser.error('--outdir is required for evaluation')
    if args.outdir and args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error('--outdir must be new or empty; historical evaluations are not overwritten')
    # No unrestricted pickle fallback: source checkpoints load with weights_only=True.
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = bundle['config']
    hw = dict(cfg.get('hardware', {}))
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        if args.device == 'cuda':
            hw['require_cuda'] = True
        device = configure_accelerator(hw)
    seed = int(cfg['env']['seed'])
    set_seed(seed)
    shape_env, _, _ = make_mujoco_env(cfg['env']['id'], seed)
    try:
        _, action_dim = dims(shape_env)
    finally:
        shape_env.close()
    cfg, model, probe, agent = restore_models(bundle, action_dim, device)
    methods = (['baseline', 'rl'] if agent else ['baseline']) if args.method in {'auto', 'both'} else [args.method]
    if (args.method in {'rl', 'both'}) and agent is None:
        parser.error('This checkpoint contains no residual_sac weights; choose baseline/auto.')
    precision = cfg.get('hardware', {}).get('precision', 'fp32') if device.type == 'cuda' else 'fp32'
    metadata = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'env': cfg['env']['id'], 'train_seed': seed, 'actual_model_history': model.history,
        'paired_checkpoint': 'pair_summary' in bundle, 'contains_rl': agent is not None,
        'device': str(device), 'episodes_per_method': args.episodes,
        'evaluation_seed_rule': '700000 + train_seed * 1000 + episode',
        'protocol': 'Archived evaluate_controller semantics; see docs/PACKAGE_STATUS.md.',
    }
    if args.inspect_only:
        print(json.dumps(metadata, indent=2))
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, args.outdir / 'checkpoint_config.yaml')
    env, _, _ = make_mujoco_env(cfg['env']['id'], seed, render_mode='rgb_array')
    try:
        low, high = action_bounds(env)
        results = {}
        for method in methods:
            name = 'lewm_mpc' if method == 'baseline' else 'lewm_mpc_rl'
            folder = args.outdir / name
            folder.mkdir()
            planner = build_planner(cfg, model, probe, low, high, device, precision)
            results[name] = evaluate_controller(
                env, cfg, planner, model, device, low, high, seed, args.episodes,
                name, folder, bundle.get('pair_summary', {}).get('effective_rank', float('nan')),
                agent=agent if method == 'rl' else None,
            )
        metadata['results'] = results
    finally:
        env.close()
    (args.outdir / 'evaluation.json').write_text(
        json.dumps(finite_json(metadata), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Evaluation saved:', args.outdir)


if __name__ == '__main__':
    main()
