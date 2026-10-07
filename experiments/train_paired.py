"""Paired MPC vs MPC+residual protocol on one shared, frozen world model.

Why this exists
---------------
``experiments/train.py`` trains ``mpc_only`` and ``planning_residual`` as separate
runs with separate world models, so the recorded difference between them confounds
"the residual helps" with "this seed's world model happened to be better". The
project's own numbers show how badly that bites: HalfCheetah ``mpc_only`` is
2293 +/- 1054 across 5 seeds, so a 95% CI on an unpaired 5-seed difference is
roughly +/-990 -- while the effects being claimed are 10-120. Two separate result
tables in docs/ disagree on the SIGN for 2 of 3 shared environments.

This entry point removes that confound:

  warmup -> train WM while running MPC -> FREEZE WM -> eval Arm A (MPC)
         -> train ONLY the residual   -> eval Arm B (MPC + residual)

Both arms are evaluated with the same frozen world model, the same evaluation
seeds, and common random numbers for the planner, so the per-seed difference
isolates the residual. ``docs/PAPER_RUN_SAFEGUARD_20260917.md`` already recommends
exactly this for the LeWM arm; it was never applied to the state-space suite.

Budget honesty
--------------
``warmup + wm_steps + rl_steps`` is the total environment budget. Arm A therefore
sees a world model trained on fewer transitions than a full-length ``mpc_only``
run, so ``pair_summary.json`` records all three of: Arm A, Arm B, and the
configured reference budget, and the report says so explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.train import (
    _adaptive_gate_for_plan,
    _apply_planning_safeguard,
    _make_planner,
    _progress,
    _save_checkpoint,
    plan_chunk,
)
from mpcrl.config import load_yaml, save_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.gate import make_uncertainty_gate, residual_ramp
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.paired_stats import minimum_detectable_effect, paired_summary
from mpcrl.plan_cache import PlanCache
from mpcrl.planning_residual import SAFEGUARD_MODES, expand_temporal_residual
from mpcrl.replay import ResidualReplay, TransitionReplay
from mpcrl.residual_context import gated_strength, residual_context
from mpcrl.sac import ResidualSAC
from mpcrl.utils import accelerator_summary, configure_accelerator, set_seed
from mpcrl.world_model import EnsembleWorldModel


def world_model_fingerprint(wm) -> str:
    """Stable hash of the world model weights.

    Asserted equal before and after residual training. A world model that is
    silently still learning would invalidate the entire paired claim, so this is
    checked in code rather than assumed in a comment.
    """
    h = hashlib.sha256()
    for key, value in sorted(wm.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _crn_seed(eval_seed: int, t: int) -> int:
    # Distinct large odd multiplier so consecutive (episode, step) pairs do not
    # alias; keeps both arms drawing bit-identical CEM noise at each index.
    return (int(eval_seed) * 1000003 + int(t)) % (2 ** 31 - 1)


def run_arm(
    cfg, wm, agent, normalized_gate, device, low, high, act_dim, k,
    temporal_decay, gate_power, episodes, train_seed, arm, out_dir,
    show_progress=True, use_crn=True,
):
    """Evaluate one arm. ``agent=None`` is the MPC-only arm."""
    rc = cfg['residual']
    mp = cfg['mpc']
    span = (high - low) / 2
    env, _, _ = make_mujoco_env(cfg['env']['id'], train_seed + 10000)
    planner = _make_planner(wm, low, high, mp, cfg['world_model'])

    fields = [
        'arm', 'eval_episode', 'eval_seed', 'train_seed', 'method', 'env',
        'episode_return', 'episode_length', 'success', 'mpc_ms',
        'prediction_mse', 'uncertainty', 'residual_norm',
        'effective_residual_norm', 'gate', 'adaptive_gate', 'gate_z',
        'residual_ramp', 'safeguard_scale', 'safeguard_gain',
        'action_d1', 'action_d2',
    ]
    logger = CSVLogger(str(Path(out_dir) / 'eval.csv'), fields)
    rows = []

    try:
        it = _progress(range(episodes), show_progress, total=episodes,
                       desc=f'eval {arm} | seed={train_seed}', leave=False, unit='ep')
        for ep in it:
            eval_seed = 100000 + train_seed * 1000 + ep
            obs, _ = env.reset(seed=eval_seed)
            planner.reset()
            metrics = EpisodeMetrics()
            info = {}
            t = 0
            while True:
                if use_crn:
                    # Common random numbers: both arms draw identical CEM noise at
                    # every (episode, step) index, which removes the planner's own
                    # sampling variance from the paired difference.
                    torch.manual_seed(_crn_seed(eval_seed, t))
                plan = planner.plan(obs)
                base = plan_chunk(plan.actions, k)
                safeguard_scale = np.nan
                safeguard_gain = np.nan

                if agent is None:
                    adaptive_gate = 0.0
                    gate_z = np.nan
                    effective_gate = 0.0
                    residual = np.zeros(act_dim, dtype=np.float32)
                    effective_delta = np.zeros_like(base)
                    corrected = base
                else:
                    adaptive_gate, gate_z = _adaptive_gate_for_plan(
                        'planning_residual', rc, normalized_gate, plan.uncertainty, update=False)
                    effective_gate = gated_strength(adaptive_gate, 1.0, gate_power)
                    c = residual_context(obs, base, effective_gate)
                    residual = agent.act(c, deterministic=True).reshape(act_dim)
                    residual_chunk = expand_temporal_residual(
                        residual, k, act_dim, decay=temporal_decay)
                    proposed = (effective_gate * float(rc['residual_scale'])
                                * residual_chunk * span[None, :])
                    corrected, effective_delta, safeguard_scale, safeguard_gain = (
                        _apply_planning_safeguard(
                            cfg, wm, obs, plan, base, proposed, low, high,
                            agent=agent, context=c, residual=residual))

                action = np.clip(corrected[0], low, high)
                no, r, term, trunc, info = env.step(action)
                with torch.no_grad():
                    pred = wm.predict(obs[None], action[None])
                    pe = float(torch.mean(
                        (pred.next_obs.squeeze(0) - torch.as_tensor(no, device=device)) ** 2).cpu())
                metrics.step(
                    r, action, plan.planning_ms, pe, plan.uncertainty,
                    float(np.linalg.norm(residual)), effective_gate,
                    adaptive_gate=adaptive_gate, gate_z=gate_z,
                    ramp=1.0 if agent is not None else 0.0,
                    effective_resnorm=float(np.linalg.norm(effective_delta)),
                    cache_hit=np.nan, safeguard_scale=safeguard_scale,
                    safeguard_gain=safeguard_gain)
                obs = no
                t += 1
                if term or trunc:
                    break

            row = metrics.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({'arm': arm, 'eval_episode': ep, 'eval_seed': eval_seed,
                        'train_seed': train_seed, 'method': arm, 'env': cfg['env']['id']})
            logger.write(row)
            rows.append(row)
    finally:
        env.close()
    return rows


def train_world_model_phase(
    cfg, wm, wm_buf, env, obs, planner, cache, steps, out_dir, seed,
    show_progress=True,
):
    """Phase A: run MPC while fitting the world model. No residual involved."""
    tc = cfg['train']
    fields = ['phase', 'global_step', 'episode', 'method', 'env', 'seed',
              'episode_return', 'episode_length', 'success', 'mpc_ms',
              'prediction_mse', 'uncertainty', 'action_d1', 'action_d2', 'wm_loss']
    logger = CSVLogger(str(Path(out_dir) / 'wm_phase_episodes.csv'), fields)
    metrics = EpisodeMetrics()
    episode = 0
    wm_loss = float('nan')
    pbar = _progress(range(1, steps + 1), show_progress, total=steps,
                     desc=f"WM phase | {cfg['env']['id']} | seed={seed}")
    for step in pbar:
        plan, _ = cache.get_or_plan(obs, planner)
        action = np.clip(plan.actions[0], *[np.asarray(x) for x in (
            planner.low, planner.high)])
        no, r, term, trunc, info = env.step(action)
        wm_buf.add(obs, action, r, no, float(term))
        if step % int(tc['wm_update_interval']) == 0:
            wm_loss = wm.fit_batch(wm_buf.sample(min(wm_buf.size, 10000)),
                                   updates=int(tc['wm_updates']),
                                   batch_size=int(tc['batch_size']))
        with torch.no_grad():
            pred = wm.predict(obs[None], action[None])
            pe = float(torch.mean(
                (pred.next_obs.squeeze(0) - torch.as_tensor(no, device=wm.device)) ** 2).cpu())
        metrics.step(r, action, plan.planning_ms, pe, plan.uncertainty)
        if term or trunc:
            row = metrics.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({'phase': 'wm', 'global_step': step, 'episode': episode,
                        'method': 'mpc_only', 'env': cfg['env']['id'], 'seed': seed,
                        'wm_loss': wm_loss})
            logger.write(row)
            episode += 1
            metrics.reset()
            obs, _ = env.reset()
            planner.reset()
            cache.clear()
            if show_progress:
                pbar.set_postfix({'ep': episode, 'ret': f"{row['episode_return']:.1f}",
                                  'wm': f'{wm_loss:.4f}'})
        else:
            obs = no
    return obs, wm_loss


def train_residual_phase(
    cfg, wm, agent, rb, normalized_gate, env, obs, planner, cache, steps,
    k, act_dim, low, high, temporal_decay, gate_power, out_dir, seed,
    show_progress=True,
):
    """Phase B: train ONLY the residual. The world model must not move."""
    tc = cfg['train']
    rc = cfg['residual']
    sc = cfg['sac']
    span = (high - low) / 2
    ramp_steps = int(rc.get('ramp_steps', 1))
    hold_steps = max(1, int(rc.get('residual_hold_steps', 1)))
    batch_size = int(tc['batch_size'])
    rl_start = int(rc.get('rl_start_steps_in_phase', 0))

    fields = ['phase', 'global_step', 'episode', 'method', 'env', 'seed',
              'episode_return', 'episode_length', 'success', 'mpc_ms',
              'prediction_mse', 'uncertainty', 'residual_norm',
              'effective_residual_norm', 'gate', 'adaptive_gate', 'gate_z',
              'residual_ramp', 'safeguard_scale', 'safeguard_gain',
              'action_d1', 'action_d2', 'q_loss', 'actor_loss', 'alpha', 'v_loss',
              'adv_q_std_ratio', 'adv_mean']
    logger = CSVLogger(str(Path(out_dir) / 'episodes.csv'), fields)
    metrics = EpisodeMetrics()
    episode = 0
    hold = None
    last_sac = {}
    pbar = _progress(range(1, steps + 1), show_progress, total=steps,
                     desc=f"residual phase | {cfg['env']['id']} | seed={seed}")

    for step in pbar:
        plan, cache_hit = cache.get_or_plan(obs, planner)
        base = plan_chunk(plan.actions, k)
        new_decision = hold is None or hold['left'] <= 0
        adaptive_gate, gate_z = _adaptive_gate_for_plan(
            'planning_residual', rc, normalized_gate, plan.uncertainty, update=new_decision)
        ramp = residual_ramp(step, max(1, rl_start), ramp_steps)
        effective_gate = gated_strength(adaptive_gate, ramp, gate_power)
        c = residual_context(obs, base, effective_gate)

        if new_decision:
            residual = agent.act(c).reshape(act_dim)
            hold = {'context': c, 'residual': residual, 'reward': 0.0,
                    'discount': 1.0, 'left': hold_steps, 'applied': 0.0, 'n': 0}
        else:
            residual = hold['residual']

        residual_chunk = expand_temporal_residual(residual, k, act_dim, decay=temporal_decay)
        proposed = effective_gate * float(rc['residual_scale']) * residual_chunk * span[None, :]
        corrected, effective_delta, safeguard_scale, safeguard_gain = (
            _apply_planning_safeguard(cfg, wm, obs, plan, base, proposed, low, high,
                                      agent=agent, context=c, residual=residual))

        action = np.clip(corrected[0], low, high)
        no, r, term, trunc, info = env.step(action)
        episode_over = bool(term or trunc)
        bootstrap_done = float(term)

        with torch.no_grad():
            pred = wm.predict(obs[None], action[None])
            pe = float(torch.mean(
                (pred.next_obs.squeeze(0) - torch.as_tensor(no, device=wm.device)) ** 2).cpu())

        applied_scale = 1.0 if np.isnan(safeguard_scale) else float(safeguard_scale)
        hold['applied'] += applied_scale
        hold['n'] += 1
        hold['reward'] += hold['discount'] * float(r)
        hold['discount'] *= float(sc['gamma'])
        hold['left'] -= 1

        if episode_over or hold['left'] <= 0 or step == steps:
            if episode_over:
                if term:
                    nc = np.zeros(c.shape[0], np.float32)
                else:
                    nplan = planner.plan(no)
                    nbase = plan_chunk(nplan.actions, k)
                    na, _ = _adaptive_gate_for_plan(
                        'planning_residual', rc, normalized_gate, nplan.uncertainty, update=False)
                    nr = residual_ramp(step + 1, max(1, rl_start), ramp_steps)
                    nc = residual_context(no, nbase, gated_strength(na, nr, gate_power))
                cache.clear()
            else:
                nplan = planner.plan(no)
                nbase = plan_chunk(nplan.actions, k)
                na, _ = _adaptive_gate_for_plan(
                    'planning_residual', rc, normalized_gate, nplan.uncertainty, update=False)
                nr = residual_ramp(step + 1, max(1, rl_start), ramp_steps)
                nc = residual_context(no, nbase, gated_strength(na, nr, gate_power))
                cache.put(no, nplan)
            executed = hold['residual'] * (hold['applied'] / max(hold['n'], 1))
            rb.add(hold['context'], executed, hold['reward'], nc, bootstrap_done,
                   discount=hold['discount'])
            hold = None
        elif not episode_over:
            cache.put(no, planner.plan(no))

        metrics.step(r, action, plan.planning_ms, pe, plan.uncertainty,
                     float(np.linalg.norm(residual)), effective_gate,
                     adaptive_gate=adaptive_gate, gate_z=gate_z, ramp=ramp,
                     effective_resnorm=float(np.linalg.norm(effective_delta)),
                     cache_hit=cache_hit, safeguard_scale=safeguard_scale,
                     safeguard_gain=safeguard_gain)

        if rb.size >= batch_size and step >= rl_start:
            for _ in range(int(tc['rl_updates_per_step'])):
                last_sac = agent.update(rb.sample(batch_size))

        if episode_over:
            row = metrics.finish(info, cfg['env'].get('success_return_threshold'))
            diag = agent.diagnostics(rb.sample(min(rb.size, 1024))) if rb.size > 1 else {}
            row.update({'phase': 'residual', 'global_step': step, 'episode': episode,
                        'method': 'planning_residual', 'env': cfg['env']['id'], 'seed': seed,
                        'q_loss': last_sac.get('q_loss', np.nan),
                        'actor_loss': last_sac.get('actor_loss', np.nan),
                        'alpha': last_sac.get('alpha', np.nan),
                        'v_loss': last_sac.get('v_loss', np.nan),
                        'adv_q_std_ratio': diag.get('adv_q_std_ratio', np.nan),
                        'adv_mean': diag.get('adv_mean', np.nan)})
            logger.write(row)
            episode += 1
            metrics.reset()
            obs, _ = env.reset()
            planner.reset()
            cache.clear()
            hold = None
            if show_progress:
                pbar.set_postfix({'ep': episode, 'ret': f"{row['episode_return']:.1f}",
                                  'gate': f"{row['gate']:.2f}",
                                  'A/Q': f"{row['adv_q_std_ratio']:.1e}"})
        else:
            obs = no
    return obs


def train_paired(cfg, steps=None, run_name=None, eval_episodes=None,
                 show_progress=True, wm_fraction=None):
    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    seed = int(cfg['env'].get('seed', 0))
    set_seed(seed)
    env, obs, _ = make_mujoco_env(cfg['env']['id'], seed)
    obs_dim, act_dim = dims(env)
    low, high = action_bounds(env)

    tc = cfg['train']
    rc = cfg['residual']
    sc = cfg['sac']
    wm_cfg = cfg['world_model']
    rc.setdefault('safeguard_mode', 'off')
    if 'model_safeguard' in rc:
        raise SystemExit(
            'residual.model_safeguard was replaced by residual.safeguard_mode '
            f"({'|'.join(SAFEGUARD_MODES)})."
        )

    total = int(steps or tc['total_steps'])
    warmup = int(tc['warmup_steps'])
    frac = float(wm_fraction if wm_fraction is not None else tc.get('paired_wm_fraction', 0.65))
    budget = max(0, total - warmup)
    wm_steps = int(budget * frac)
    rl_steps = budget - wm_steps
    if wm_steps < 1 or rl_steps < 1:
        raise SystemExit(
            f'budget too small: total={total} warmup={warmup} -> wm={wm_steps} rl={rl_steps}')

    wm = EnsembleWorldModel(
        obs_dim, act_dim, wm_cfg['ensemble_size'], wm_cfg['hidden_dim'],
        wm_cfg['lr'], wm_cfg['weight_decay'], device, precision,
        stats_momentum=wm_cfg.get('stats_momentum', 0.99),
        bootstrap_prob=wm_cfg.get('bootstrap_prob', 0.8),
        grad_clip=wm_cfg.get('grad_clip', 10.0))
    wm_buf = TransitionReplay(max(total + 10000, 100000), obs_dim, act_dim)

    root = Path(cfg['logging'].get('root', 'runs'))
    tag = int(time.time())
    name = run_name or f"{cfg['env']['id']}__paired__seed{seed}__{tag}"
    base_out = root / f'{name}__armA_mpc'
    rl_out = root / f'{name}__armB_residual'
    for d in (base_out, rl_out):
        d.mkdir(parents=True, exist_ok=True)
        save_yaml(cfg, d / 'config.yaml')

    print(f'\n=== Stage 1/6: warmup {warmup} random steps ===', flush=True)
    for _ in _progress(range(warmup), show_progress, total=warmup, desc='warmup', leave=False):
        a = env.action_space.sample()
        no, r, term, trunc, _ = env.step(a)
        wm_buf.add(obs, a, r, no, float(term))
        obs = no
        if term or trunc:
            obs, _ = env.reset()
    wm.fit_batch(wm_buf.sample(min(wm_buf.size, 10000)),
                 updates=int(tc['wm_updates']), batch_size=int(tc['batch_size']))

    planner = _make_planner(wm, low, high, cfg['mpc'], wm_cfg)
    cache = PlanCache()
    k = int(rc['chunk_len'])
    temporal_decay = float(rc.get('temporal_decay', 0.8))
    gate_power = float(rc.get('gate_power', 1.0))

    print(f'\n=== Stage 2/6: train world model for {wm_steps} steps under MPC ===', flush=True)
    obs, wm_loss = train_world_model_phase(
        cfg, wm, wm_buf, env, obs, planner, cache, wm_steps, base_out, seed, show_progress)

    print('\n=== Stage 3/6: freeze the world model ===', flush=True)
    wm.freeze()
    fingerprint_before = world_model_fingerprint(wm)
    print(f'world model fingerprint: {fingerprint_before[:16]}', flush=True)
    _save_checkpoint(base_out / 'frozen_world_model.pt', wm, None, None, wm_steps, 'paired')

    n_eval = int(eval_episodes if eval_episodes is not None else tc.get('final_eval_episodes', 5))
    print(f'\n=== Stage 4/6: evaluate Arm A (MPC only), {n_eval} episodes ===', flush=True)
    rows_a = run_arm(cfg, wm, None, None, device, low, high, act_dim, k,
                     temporal_decay, gate_power, n_eval, seed, 'mpc_only', base_out,
                     show_progress)

    print(f'\n=== Stage 5/6: train residual only, {rl_steps} steps ===', flush=True)
    set_seed(seed + 12345)
    context_dim = obs_dim + k * act_dim + 1
    agent = ResidualSAC(
        context_dim, act_dim, sc['hidden_dim'], sc['lr'], sc['gamma'], sc['tau'],
        sc['init_alpha'], sc['target_entropy_scale'], rc.get('consistency_coef', 0.0),
        device, precision, critic=sc.get('critic', 'dueling'),
        obs_dim=obs_dim if sc.get('terminal_value', True) else None,
        actor_delay=int(sc.get('actor_delay', 2)),
        target_policy_noise=float(sc.get('target_policy_noise', 0.0)),
        trust_region_relative=bool(sc.get('trust_region_relative', True)))
    rb = ResidualReplay(max(rl_steps + 10000, 100000), context_dim, act_dim,
                        default_discount=sc['gamma'])
    normalized_gate = make_uncertainty_gate(rc) if rc.get('adaptive_gate', True) else None
    obs, _ = env.reset()
    planner.reset()
    cache.clear()
    train_residual_phase(cfg, wm, agent, rb, normalized_gate, env, obs, planner, cache,
                         rl_steps, k, act_dim, low, high, temporal_decay, gate_power,
                         rl_out, seed, show_progress)

    print(f'\n=== Stage 6/6: evaluate Arm B (MPC + residual), {n_eval} episodes ===', flush=True)
    rows_b = run_arm(cfg, wm, agent, normalized_gate, device, low, high, act_dim, k,
                     temporal_decay, gate_power, n_eval, seed, 'planning_residual', rl_out,
                     show_progress)

    fingerprint_after = world_model_fingerprint(wm)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            'The world model changed during residual training. The paired comparison is '
            f'invalid. before={fingerprint_before[:16]} after={fingerprint_after[:16]}')
    print(f'world model unchanged across both arms ({fingerprint_after[:16]})', flush=True)

    ra = np.array([r['episode_return'] for r in rows_a], dtype=float)
    rbv = np.array([r['episode_return'] for r in rows_b], dtype=float)
    # Episode-level pairing is valid because CRN pairs episodes, but the seed is
    # the independent unit for the headline claim.
    per_episode_delta = (rbv - ra).tolist()
    summary = {
        'env': cfg['env']['id'],
        'seed': seed,
        'warmup_steps': warmup,
        'wm_phase_steps': wm_steps,
        'rl_phase_steps': rl_steps,
        'total_env_steps': warmup + wm_steps + rl_steps,
        'eval_episodes': n_eval,
        'safeguard_mode': rc.get('safeguard_mode'),
        'cem_execute': cfg['mpc'].get('execute', 'best_sample'),
        'residual_hold_steps': rc.get('residual_hold_steps', 1),
        'world_model_fingerprint': fingerprint_before,
        'world_model_shared_and_frozen': True,
        'common_random_numbers': True,
        'armA_mpc_return_mean': float(ra.mean()),
        'armA_mpc_return_std': float(ra.std(ddof=1)) if len(ra) > 1 else 0.0,
        'armB_residual_return_mean': float(rbv.mean()),
        'armB_residual_return_std': float(rbv.std(ddof=1)) if len(rbv) > 1 else 0.0,
        'delta_mean_within_seed': float(rbv.mean() - ra.mean()),
        'per_episode_delta': per_episode_delta,
        'per_episode_paired': paired_summary(per_episode_delta),
        'note': (
            'Arm A uses a world model trained on warmup+wm_phase_steps transitions, which is '
            'less than a full-length mpc_only run. Compare the paired delta against the '
            'standalone mpc_only reference at the full budget before claiming an improvement.'
        ),
    }
    with open(rl_out / 'pair_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    _save_checkpoint(rl_out / 'final.pt', wm, agent, normalized_gate, rl_steps, 'paired')

    env.close()
    print('\n=== PAIRED RESULT (single seed) ===')
    print(f"Arm A  MPC only        : {ra.mean():.3f} +/- {summary['armA_mpc_return_std']:.3f}")
    print(f"Arm B  MPC + residual  : {rbv.mean():.3f} +/- {summary['armB_residual_return_std']:.3f}")
    print(f"delta (within seed)    : {summary['delta_mean_within_seed']:+.3f}")
    print(f'episodes with positive delta: '
          f"{summary['per_episode_paired']['n_positive']}/{n_eval}")
    print('\nA single seed is not a result. Run several seeds and use '
          'experiments.summarize_paired for the seed-level paired CI.')
    print('armA:', base_out)
    print('armB:', rl_out)
    return base_out, rl_out, summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', default='configs/local/halfcheetah.yaml')
    p.add_argument('--steps', type=int, help='total environment budget incl. warmup')
    p.add_argument('--seed', type=int)
    p.add_argument('--run-name')
    p.add_argument('--eval-episodes', type=int)
    p.add_argument('--wm-fraction', type=float,
                   help='fraction of the post-warmup budget spent training the world model '
                        '(default train.paired_wm_fraction or 0.65)')
    p.add_argument('--no-progress', action='store_true')
    a = p.parse_args()

    cfg = load_yaml(a.config)
    if a.seed is not None:
        cfg['env']['seed'] = a.seed
    train_paired(cfg, a.steps, a.run_name, a.eval_episodes,
                 show_progress=not a.no_progress, wm_fraction=a.wm_fraction)


if __name__ == '__main__':
    main()
