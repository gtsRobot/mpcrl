from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from mpcrl.cem import CEMPlanner
from mpcrl.config import load_yaml, save_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.gate import residual_ramp
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.replay import ResidualReplay, TransitionReplay
from mpcrl.sac import ResidualSAC
from mpcrl.utils import accelerator_summary, configure_accelerator, set_seed
from mpcrl.world_model import EnsembleWorldModel
from mpcrl.zprl_style import BottleneckBasePolicy, fit_behavior_clone


def _pbar(iterable, enabled=True, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=0.5, **kwargs)


def _save_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.random.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _normalized_obs(base, obs, device):
    with torch.no_grad():
        ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return base.normalize_obs(ot).squeeze(0).cpu().numpy()


def _latent(base, obs, device):
    with torch.no_grad():
        ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return base.encode(ot).squeeze(0).cpu().numpy()


def _context(base, obs, z, ramp, device):
    return np.concatenate([
        _normalized_obs(base, obs, device),
        z,
        np.asarray([ramp], dtype=np.float32),
    ]).astype(np.float32, copy=False)


def _base_action(base, obs, device, mid, span, low, high):
    with torch.no_grad():
        ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        an = base(ot).squeeze(0).cpu().numpy()
    return np.clip(mid + span * an, low, high)


def _dataset_mse(base, states, targets_norm, device):
    with torch.no_grad():
        pred = base(torch.as_tensor(states, dtype=torch.float32, device=device)).cpu().numpy()
    return float(np.mean((pred - targets_norm) ** 2))


def _dagger_refine(
    cfg,
    base,
    planner,
    env,
    states,
    targets,
    device,
    low,
    high,
    mid,
    span,
    seed,
    rounds,
    steps_per_round,
    refit_epochs,
    show_progress,
):
    """Small DAgger-style correction for BC covariate shift."""
    if rounds <= 0 or steps_per_round <= 0:
        return states, targets, np.nan

    batch_size = int(cfg['train']['batch_size'])
    bc_loss = np.nan
    for rd in range(rounds):
        extra_states = []
        extra_targets = []
        obs, _ = env.reset(seed=seed + 303 + rd)
        planner.reset()
        iterator = _pbar(
            range(steps_per_round),
            show_progress,
            total=steps_per_round,
            desc=f'ZPRL DAgger {rd + 1}/{rounds}',
            leave=False,
        )
        for _ in iterator:
            plan = planner.plan(obs)
            expert_action = np.clip(plan.actions[0], low, high)
            learner_action = _base_action(base, obs, device, mid, span, low, high)
            extra_states.append(obs.copy())
            extra_targets.append(expert_action.copy())

            no, _, te, tr, _ = env.step(learner_action)
            if te or tr:
                obs, _ = env.reset()
                planner.reset()
            else:
                obs = no

        states = np.concatenate([
            states,
            np.asarray(extra_states, dtype=np.float32),
        ], axis=0)
        targets = np.concatenate([
            targets,
            np.asarray(extra_targets, dtype=np.float32),
        ], axis=0)
        targets_norm = np.clip((targets - mid) / np.maximum(span, 1e-6), -1, 1)
        bc_loss = fit_behavior_clone(
            base,
            states,
            targets_norm,
            epochs=refit_epochs,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
            update_obs_stats=True,
        )
    return states, targets, bc_loss


def evaluate_base_policy(
    cfg,
    base,
    device,
    low,
    high,
    mid,
    span,
    train_seed,
    episodes,
    show_progress=True,
):
    """Evaluate the frozen behavior-cloned base before residual RL starts."""
    if episodes <= 0:
        return []

    rng_state = _save_rng_state()
    env, _, _ = make_mujoco_env(cfg['env']['id'], train_seed + 15000)
    rows = []
    try:
        ep_iter = _pbar(
            range(episodes),
            show_progress,
            total=episodes,
            desc=f"base eval {cfg['env']['id']} | seed={train_seed}",
            leave=False,
            unit='ep',
        )
        for ep in ep_iter:
            eval_seed = 150000 + train_seed * 1000 + ep
            obs, _ = env.reset(seed=eval_seed)
            done = False
            info = {}
            em = EpisodeMetrics()
            while not done:
                action = _base_action(base, obs, device, mid, span, low, high)
                no, r, te, tr, info = env.step(action)
                done = te or tr
                em.step(
                    r,
                    action,
                    resnorm=0.0,
                    effective_resnorm=0.0,
                    gate=0.0,
                    ramp=0.0,
                )
                obs = no

            row = em.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({
                'eval_episode': ep,
                'eval_seed': eval_seed,
                'train_seed': train_seed,
                'method': 'zprl_base',
                'env': cfg['env']['id'],
            })
            rows.append(row)
            if show_progress:
                ep_iter.set_postfix(ret=f"{row['episode_return']:.1f}", length=row['episode_length'])
    finally:
        env.close()
        _restore_rng_state(rng_state)
    return rows


def evaluate_zprl(
    cfg,
    base,
    agent,
    device,
    low,
    high,
    mid,
    span,
    latent_residual_scale,
    train_seed,
    episodes,
    show_progress=True,
):
    if episodes <= 0:
        return []

    rng_state = _save_rng_state()
    env, _, _ = make_mujoco_env(cfg['env']['id'], train_seed + 20000)
    rows = []
    try:
        ep_iter = _pbar(
            range(episodes),
            show_progress,
            total=episodes,
            desc=f"final eval {cfg['env']['id']} | zprl_style | seed={train_seed}",
            leave=False,
            unit='ep',
        )
        for ep in ep_iter:
            eval_seed = 200000 + train_seed * 1000 + ep
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)

            obs, _ = env.reset(seed=eval_seed)
            done = False
            info = {}
            em = EpisodeMetrics()

            while not done:
                z = _latent(base, obs, device)
                c = _context(base, obs, z, 1.0, device)
                dz = agent.act(c, deterministic=True)
                effective_dz = latent_residual_scale * dz
                z2 = torch.as_tensor(
                    z + effective_dz, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    an = base.decode(z2).squeeze(0).cpu().numpy()
                action = np.clip(mid + span * an, low, high)

                no, r, te, tr, info = env.step(action)
                done = te or tr
                em.step(
                    r,
                    action,
                    resnorm=float(np.linalg.norm(dz)),
                    effective_resnorm=float(np.linalg.norm(effective_dz)),
                    gate=1.0,
                    ramp=1.0,
                )
                obs = no

            row = em.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({
                'eval_episode': ep,
                'eval_seed': eval_seed,
                'train_seed': train_seed,
                'method': 'zprl_style',
                'env': cfg['env']['id'],
            })
            rows.append(row)
            if show_progress:
                ep_iter.set_postfix(ret=f"{row['episode_return']:.1f}", length=row['episode_length'])
    finally:
        env.close()
        _restore_rng_state(rng_state)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/halfcheetah.yaml')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--latent-dim', type=int)
    ap.add_argument('--bc-collect-steps', type=int)
    ap.add_argument('--bc-epochs', type=int)
    ap.add_argument('--eval-episodes', type=int)
    ap.add_argument('--no-progress', action='store_true')
    args = ap.parse_args()

    show_progress = not args.no_progress
    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg['env']['seed']
    cfg['env']['seed'] = seed
    set_seed(seed)

    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    env, obs, _ = make_mujoco_env(cfg['env']['id'], seed)
    od, ad = dims(env)
    low, high = action_bounds(env)
    mid = (high + low) / 2
    span = (high - low) / 2

    tc = cfg['train']
    zc = cfg.get('zprl_style', {})
    latent_dim = int(args.latent_dim or zc.get('latent_dim', 16))
    base_hidden_dim = int(zc.get('base_hidden_dim', 256))
    bc_collect_steps = int(args.bc_collect_steps or zc.get('bc_collect_steps', 6000))
    bc_epochs = int(args.bc_epochs or zc.get('bc_epochs', 80))
    latent_residual_scale = float(zc.get('residual_scale', 0.05))
    latent_residual_start = int(zc.get('residual_start_steps', 1))
    latent_ramp_steps = int(zc.get('residual_ramp_steps', 5000))
    dagger_rounds = int(zc.get('dagger_rounds', 0))
    dagger_steps = int(zc.get('dagger_steps_per_round', 0))
    dagger_epochs = int(zc.get('dagger_refit_epochs', 30))
    base_eval_episodes = int(zc.get('base_eval_episodes', 3))
    base_warn_threshold = zc.get('base_eval_warn_threshold')
    zprl_consistency = float(
        zc.get('consistency_coef', cfg['residual'].get('consistency_coef', 0.05))
    )

    buf = TransitionReplay(
        max(100000, int(args.steps) + bc_collect_steps + dagger_rounds * dagger_steps + 10000),
        od,
        ad,
    )
    warm = int(tc['warmup_steps'])
    for _ in _pbar(
        range(warm),
        show_progress,
        total=warm,
        desc=f"ZPRL warmup {cfg['env']['id']} seed={seed}",
        leave=False,
    ):
        a = env.action_space.sample()
        no, r, te, tr, _ = env.step(a)
        done = te or tr
        buf.add(obs, a, r, no, done)
        obs = no
        if done:
            obs, _ = env.reset()

    w = cfg['world_model']
    wm = EnsembleWorldModel(
        od, ad, w['ensemble_size'], w['hidden_dim'], w['lr'], w['weight_decay'], device, precision
    )
    wm_loss = wm.fit_batch(
        buf.sample(min(buf.size, 10000)),
        updates=int(tc['wm_updates']),
        batch_size=int(tc['batch_size']),
    )

    m = cfg['mpc']
    planner = CEMPlanner(
        wm, low, high, m['horizon'], m['candidates'], m['elites'], m['iterations'],
        m['alpha'], m['init_std'], m['min_std'], m['discount'], w.get('uncertainty_penalty', 0.0)
    )

    bc_states = []
    bc_targets = []
    obs, _ = env.reset(seed=seed + 101)
    planner.reset()
    collect = _pbar(
        range(1, bc_collect_steps + 1),
        show_progress,
        total=bc_collect_steps,
        desc=f"ZPRL MPC-BC collect {cfg['env']['id']} seed={seed}",
        leave=False,
    )
    for i in collect:
        plan = planner.plan(obs)
        action = np.clip(plan.actions[0], low, high)
        bc_states.append(obs.copy())
        bc_targets.append(action.copy())

        no, r, te, tr, _ = env.step(action)
        done = te or tr
        buf.add(obs, action, r, no, done)

        if i % int(tc['wm_update_interval']) == 0:
            wm_loss = wm.fit_batch(
                buf.sample(min(buf.size, 10000)),
                updates=int(tc['wm_updates']),
                batch_size=int(tc['batch_size']),
            )
            if show_progress:
                collect.set_postfix(wm=f'{wm_loss:.4f}')

        if done:
            obs, _ = env.reset()
            planner.reset()
        else:
            obs = no

    states = np.asarray(bc_states, dtype=np.float32)
    targets = np.asarray(bc_targets, dtype=np.float32)
    target_norm = np.clip((targets - mid) / np.maximum(span, 1e-6), -1, 1)

    base = BottleneckBasePolicy(od, ad, latent_dim, hidden=base_hidden_dim).to(device)
    bc = fit_behavior_clone(
        base,
        states,
        target_norm,
        epochs=bc_epochs,
        batch_size=int(tc['batch_size']),
        device=device,
        show_progress=show_progress,
    )

    states, targets, dagger_bc = _dagger_refine(
        cfg,
        base,
        planner,
        env,
        states,
        targets,
        device,
        low,
        high,
        mid,
        span,
        seed,
        dagger_rounds,
        dagger_steps,
        dagger_epochs,
        show_progress,
    )
    if np.isfinite(dagger_bc):
        bc = dagger_bc
    target_norm = np.clip((targets - mid) / np.maximum(span, 1e-6), -1, 1)
    bc_dataset_mse = _dataset_mse(base, states, target_norm, device)
    print(
        f'ZPRL base-policy BC: samples={len(states)} final_mse={bc_dataset_mse:.6f} '
        f'dagger={dagger_rounds}x{dagger_steps}'
    )

    out = Path(cfg['logging']['root']) / f"{cfg['env']['id']}__zprl_style__seed{seed}__{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out / 'config.yaml')

    base_rows = evaluate_base_policy(
        cfg,
        base,
        device,
        low,
        high,
        mid,
        span,
        seed,
        base_eval_episodes,
        show_progress,
    )
    base_eval_mean = np.nan
    if base_rows:
        base_eval_fields = [
            'eval_episode', 'eval_seed', 'train_seed', 'method', 'env',
            'episode_return', 'episode_length', 'success', 'mpc_ms',
            'prediction_mse', 'uncertainty', 'residual_norm',
            'effective_residual_norm', 'gate', 'residual_ramp',
            'action_d1', 'action_d2',
        ]
        blog = CSVLogger(str(out / 'base_eval.csv'), base_eval_fields)
        for row in base_rows:
            blog.write(row)
        bvals = np.asarray([r['episode_return'] for r in base_rows], dtype=float)
        base_eval_mean = float(bvals.mean())
        base_eval_std = float(bvals.std(ddof=1)) if len(bvals) > 1 else 0.0
        print(
            f"BASE_EVAL seed={seed} episodes={len(bvals)} "
            f"return={base_eval_mean:.3f}+/-{base_eval_std:.3f}"
        )
        if base_warn_threshold is not None and base_eval_mean < float(base_warn_threshold):
            print(
                f"WARNING: base policy return {base_eval_mean:.3f} is below "
                f"configured preflight threshold {float(base_warn_threshold):.3f}; "
                "treat this run as diagnostic before launching a 100k sweep."
            )

    # The critic observes ramp because the executed latent perturbation is
    # ramp * residual_scale * dz. residual_start_steps allows a short pure-base
    # phase so that a usable clone is not immediately perturbed by an untrained SAC.
    ctx_dim = od + latent_dim + 1
    agent = ResidualSAC(
        ctx_dim,
        latent_dim,
        hidden=cfg['sac']['hidden_dim'],
        lr=cfg['sac']['lr'],
        gamma=cfg['sac']['gamma'],
        tau=cfg['sac']['tau'],
        init_alpha=cfg['sac']['init_alpha'],
        target_entropy_scale=cfg['sac']['target_entropy_scale'],
        consistency_coef=zprl_consistency,
        device=device,
        precision=precision,
    )
    rb = ResidualReplay(max(100000, int(args.steps) + 10000), ctx_dim, latent_dim)

    fields = [
        'global_step', 'episode', 'method', 'env', 'seed', 'episode_return',
        'episode_length', 'success', 'mpc_ms', 'prediction_mse', 'uncertainty',
        'residual_norm', 'effective_residual_norm', 'gate', 'residual_ramp',
        'action_d1', 'action_d2', 'bc_loss', 'bc_dataset_mse',
    ]
    log = CSVLogger(str(out / 'episodes.csv'), fields)

    obs, _ = env.reset(seed=seed + 202)
    em = EpisodeMetrics()
    ep = 0
    pbar = _pbar(
        range(1, args.steps + 1),
        show_progress,
        total=args.steps,
        desc=f"{cfg['env']['id']} | zprl_style | seed={seed}",
    )

    for step in pbar:
        ramp = residual_ramp(step, latent_residual_start, latent_ramp_steps)
        z = _latent(base, obs, device)
        c = _context(base, obs, z, ramp, device)
        dz = agent.act(c)
        effective_dz = ramp * latent_residual_scale * dz
        z2 = torch.as_tensor(z + effective_dz, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            an = base.decode(z2).squeeze(0).cpu().numpy()
        action = np.clip(mid + span * an, low, high)

        no, r, te, tr, info = env.step(action)
        done = te or tr
        if done:
            nc = np.zeros(ctx_dim, np.float32)
        else:
            next_ramp = residual_ramp(step + 1, latent_residual_start, latent_ramp_steps)
            nz = _latent(base, no, device)
            nc = _context(base, no, nz, next_ramp, device)

        rb.add(c, dz, r, nc, done)
        em.step(
            r,
            action,
            resnorm=float(np.linalg.norm(dz)),
            effective_resnorm=float(np.linalg.norm(effective_dz)),
            gate=ramp,
            ramp=ramp,
        )

        if rb.size >= int(tc['batch_size']):
            agent.update(rb.sample(int(tc['batch_size'])))

        if done:
            row = em.finish(info, cfg['env'].get('success_return_threshold'))
            row.update(
                global_step=step,
                episode=ep,
                method='zprl_style',
                env=cfg['env']['id'],
                seed=seed,
                bc_loss=bc,
                bc_dataset_mse=bc_dataset_mse,
            )
            log.write(row)
            ep += 1
            em.reset()
            obs, _ = env.reset()
            if show_progress:
                pbar.set_postfix(
                    ep=ep,
                    ret=f"{row['episode_return']:.1f}",
                    length=int(row['episode_length']),
                    ramp=f'{ramp:.2f}',
                    bc=f'{bc_dataset_mse:.4f}',
                )
                if ep % 10 == 0:
                    tqdm.write(json.dumps(row, ensure_ascii=False))
            else:
                print(json.dumps(row, ensure_ascii=False))
        else:
            obs = no

    torch.save(
        {
            'base_policy': base.state_dict(),
            'agent': agent.state_dict(),
            'latent_dim': latent_dim,
            'config': cfg,
            'bc_dataset_mse': bc_dataset_mse,
            'base_eval_return_mean': base_eval_mean,
            'context_dim': ctx_dim,
        },
        out / 'final.pt',
    )

    n_eval = int(tc.get('final_eval_episodes', 3) if args.eval_episodes is None else args.eval_episodes)
    eval_rows = evaluate_zprl(
        cfg, base, agent, device, low, high, mid, span,
        latent_residual_scale, seed, n_eval, show_progress,
    )
    if eval_rows:
        eval_fields = [
            'eval_episode', 'eval_seed', 'train_seed', 'method', 'env',
            'episode_return', 'episode_length', 'success', 'mpc_ms',
            'prediction_mse', 'uncertainty', 'residual_norm',
            'effective_residual_norm', 'gate', 'residual_ramp',
            'action_d1', 'action_d2',
        ]
        elog = CSVLogger(str(out / 'eval.csv'), eval_fields)
        for row in eval_rows:
            elog.write(row)
        vals = np.asarray([r['episode_return'] for r in eval_rows], dtype=float)
        print(
            f"FINAL_EVAL method=zprl_style seed={seed} episodes={len(vals)} "
            f"return={vals.mean():.3f}+/-{vals.std(ddof=1) if len(vals)>1 else 0.0:.3f}"
        )

    env.close()
    print('output:', out)


if __name__ == '__main__':
    main()
