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
from mpcrl.gate import adaptive_uncertainty_gate, make_uncertainty_gate, residual_ramp
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.plan_cache import PlanCache
from mpcrl.planning_residual import (
    SAFEGUARD_MODES,
    expand_temporal_residual,
    safeguard_residual_plan,
)
from mpcrl.replay import ResidualReplay, TransitionReplay
from mpcrl.residual_context import gated_strength, residual_context
from mpcrl.sac import ResidualSAC
from mpcrl.utils import accelerator_summary, configure_accelerator, set_seed
from mpcrl.world_model import EnsembleWorldModel


METHODS = {'mpc_only', 'action_residual', 'planning_residual'}


def plan_chunk(plan_actions, k):
    p = plan_actions[:k]
    if len(p) < k:
        p = np.concatenate([p, np.repeat(p[-1:], k - len(p), axis=0)], axis=0)
    return p


def _progress(iterable, enabled: bool, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=0.5, **kwargs)


def _adaptive_gate_for_plan(method, rc, normalized_gate, uncertainty, update=True):
    if method != 'planning_residual':
        return float(rc.get('fixed_gate', 1.0)), np.nan
    if not rc.get('adaptive_gate', True):
        return float(rc.get('fixed_gate', 1.0)), np.nan
    if normalized_gate is not None:
        gout = normalized_gate(uncertainty, update=update)
        return gout.value, gout.normalized_uncertainty
    return adaptive_uncertainty_gate(
        uncertainty,
        rc['gate_threshold'],
        rc['gate_temperature'],
    ), np.nan


def _make_planner(wm, low, high, mp, wm_cfg):
    return CEMPlanner(
        wm,
        low,
        high,
        mp['horizon'],
        mp['candidates'],
        mp['elites'],
        mp['iterations'],
        mp['alpha'],
        mp['init_std'],
        mp['min_std'],
        mp['discount'],
        wm_cfg.get('uncertainty_penalty', 0.0),
        execute=mp.get('execute', 'best_sample'),
    )


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


def _apply_planning_safeguard(
    cfg,
    wm,
    obs,
    plan,
    base,
    proposed_delta,
    low,
    high,
    agent=None,
    context=None,
    residual=None,
):
    """Apply the residual, optionally shrunk/vetoed by the configured shield.

    ``residual.safeguard_mode`` selects the evidence the veto uses; see
    ``mpcrl.planning_residual.safeguard_residual_plan``. It defaults to ``off``:
    the legacy ``model`` mode re-scores with the very objective CEM already
    maximised, which structurally prevents the residual from ever exceeding the
    MPC baseline, so it must be opted into explicitly as an ablation.
    """
    rc = cfg['residual']
    mode = str(rc.get('safeguard_mode', 'off')).lower()
    if mode == 'off':
        return base + proposed_delta, proposed_delta, np.nan, np.nan

    terminal_value_fn = None
    if mode == 'model_value':
        if agent is None or agent.state_value is None:
            raise SystemExit(
                "residual.safeguard_mode='model_value' needs a SAC agent with a state-value "
                'head; set sac.terminal_value: true.'
            )
        terminal_value_fn = agent.terminal_value

    sg = safeguard_residual_plan(
        wm,
        obs,
        plan.actions,
        proposed_delta,
        low,
        high,
        discount=cfg['mpc']['discount'],
        uncertainty_penalty=cfg['world_model'].get('uncertainty_penalty', 0.0),
        scales=rc.get('safeguard_scales', [1.0, 0.5, 0.25, 0.0]),
        min_improvement=rc.get('safeguard_min_improvement', 0.0),
        mode=mode,
        agent=agent,
        context=context,
        residual=residual,
        terminal_value_fn=terminal_value_fn,
    )
    corrected = sg.corrected_plan[:len(base)]
    actual_delta = corrected - base
    return corrected, actual_delta, sg.scale, sg.predicted_gain


def _save_checkpoint(path, wm, agent, gate, step, method, wm_buf=None, rb=None, keep_replay=False):
    """Write a checkpoint that is actually sufficient to resume from.

    Checkpoints used to be written every N steps and never read by anything in the
    repo, so a killed run restarted from zero. Replay buffers are large, so they
    are only included when ``train.checkpoint_replay`` is set.

    Buffers are stored as torch tensors, not numpy arrays, so the file stays
    loadable under ``torch.load(weights_only=True)`` (the default since torch 2.6).
    """
    ck = {'world_model': wm.state_dict(), 'step': int(step), 'method': method}
    if agent is not None:
        ck['agent'] = agent.state_dict()
    if gate is not None:
        ck['gate_state'] = gate.state_dict()
    if keep_replay:
        if wm_buf is not None:
            ck['wm_buf'] = _dump_buffer(wm_buf, ('obs', 'action', 'reward', 'next_obs', 'done'))
        if rb is not None:
            ck['residual_buf'] = _dump_buffer(
                rb, ('context', 'residual', 'reward', 'next_context', 'done', 'discount'))
    tmp = Path(str(path) + '.tmp')
    torch.save(ck, tmp)
    tmp.replace(path)
    return path


def _dump_buffer(buf, keys):
    out = {'size': int(buf.size), 'ptr': int(buf.ptr)}
    for key in keys:
        out[key] = torch.from_numpy(np.asarray(getattr(buf, key)[:buf.size]))
    return out


def _restore_buffer(buf, saved, keys):
    n = int(saved['size'])
    for key in keys:
        value = saved[key]
        getattr(buf, key)[:n] = value.cpu().numpy() if torch.is_tensor(value) else value
    buf.size = n
    buf.ptr = int(saved['ptr'])


def _latest_checkpoint(out_dir):
    cks = sorted(
        Path(out_dir).glob('checkpoint_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1]),
    )
    return cks[-1] if cks else None


def evaluate_final(
    cfg,
    method,
    wm,
    agent,
    normalized_gate,
    device,
    low,
    high,
    act_dim,
    k,
    temporal_decay,
    gate_power,
    episodes,
    train_seed,
    show_progress=True,
):
    """Final evaluation with fixed environment/CEM seeds and deterministic SAC mean action.

    Evaluation RNG is isolated and restored afterwards so evaluation never changes
    the training RNG stream. The uncertainty gate is read-only here.
    """
    if episodes <= 0:
        return []

    wm_cfg = cfg['world_model']
    mp = cfg['mpc']
    rc = cfg['residual']
    span = (high - low) / 2
    rng_state = _save_rng_state()
    rows = []

    eval_env, _, _ = make_mujoco_env(cfg['env']['id'], train_seed + 10000)
    eval_planner = _make_planner(wm, low, high, mp, wm_cfg)

    try:
        ep_iter = _progress(
            range(episodes),
            show_progress,
            total=episodes,
            desc=f"final eval {cfg['env']['id']} | {method} | seed={train_seed}",
            leave=False,
            unit='ep',
        )
        for ep in ep_iter:
            eval_seed = 100000 + train_seed * 1000 + ep
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)

            obs, _ = eval_env.reset(seed=eval_seed)
            eval_planner.reset()
            metrics = EpisodeMetrics()
            done = False
            info = {}

            while not done:
                plan = eval_planner.plan(obs)
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
                        method, rc, normalized_gate, plan.uncertainty, update=False
                    )
                    effective_gate = gated_strength(adaptive_gate, 1.0, gate_power)
                    c = residual_context(obs, base, effective_gate)
                    residual = agent.act(c, deterministic=True).reshape(act_dim)

                    if method == 'planning_residual':
                        residual_chunk = expand_temporal_residual(
                            residual, k, act_dim, decay=temporal_decay
                        )
                    else:
                        residual_chunk = residual.reshape(1, act_dim)

                    proposed_delta = (
                        effective_gate
                        * float(rc['residual_scale'])
                        * residual_chunk
                        * span[None, :]
                    )
                    if method == 'planning_residual':
                        corrected, effective_delta, safeguard_scale, safeguard_gain = (
                            _apply_planning_safeguard(
                                cfg, wm, obs, plan, base, proposed_delta, low, high,
                                agent=agent, context=c, residual=residual,
                            )
                        )
                    else:
                        effective_delta = proposed_delta
                        corrected = base + effective_delta

                action = np.clip(corrected[0], low, high)
                no, r, term, trunc, info = eval_env.step(action)
                done = term or trunc

                with torch.no_grad():
                    pred = wm.predict(obs[None], action[None])
                    pe = float(torch.mean(
                        (pred.next_obs.squeeze(0) - torch.as_tensor(no, device=device)) ** 2
                    ).cpu())

                metrics.step(
                    r,
                    action,
                    plan.planning_ms,
                    pe,
                    plan.uncertainty,
                    float(np.linalg.norm(residual)),
                    effective_gate,
                    adaptive_gate=adaptive_gate,
                    gate_z=gate_z,
                    ramp=1.0 if agent is not None else 0.0,
                    effective_resnorm=float(np.linalg.norm(effective_delta)),
                    cache_hit=np.nan,
                    safeguard_scale=safeguard_scale,
                    safeguard_gain=safeguard_gain,
                )
                obs = no

            row = metrics.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({
                'eval_episode': ep,
                'eval_seed': eval_seed,
                'train_seed': train_seed,
                'method': method,
                'env': cfg['env']['id'],
            })
            rows.append(row)
            if show_progress:
                ep_iter.set_postfix(ret=f"{row['episode_return']:.1f}", length=row['episode_length'])
    finally:
        eval_env.close()
        _restore_rng_state(rng_state)

    return rows


def train(
    cfg,
    method='planning_residual',
    steps=None,
    run_name=None,
    show_progress=True,
    eval_episodes=None,
    resume=False,
):
    assert method in METHODS

    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    seed = int(cfg['env'].get('seed', 0))
    set_seed(seed)
    env, obs, _ = make_mujoco_env(cfg['env']['id'], seed)
    obs_dim, act_dim = dims(env)
    low, high = action_bounds(env)
    span = (high - low) / 2

    tc = cfg['train']
    total = int(steps or tc['total_steps'])
    wm_cfg = cfg['world_model']
    mp = cfg['mpc']
    rc = cfg['residual']
    # No setdefault that changes the method: the shield is off unless a config
    # asks for it, and whatever is chosen is recorded in the saved config.yaml.
    rc.setdefault('safeguard_mode', 'off')
    rc.setdefault('safeguard_scales', [1.0, 0.5, 0.25, 0.0])
    rc.setdefault('safeguard_min_improvement', 0.0)
    if 'model_safeguard' in rc:
        raise SystemExit(
            "residual.model_safeguard was replaced by residual.safeguard_mode "
            f"({'|'.join(SAFEGUARD_MODES)}). Set safeguard_mode: model to reproduce the old "
            'model_safeguard: true behaviour, or off to disable the shield.'
        )
    sc = cfg['sac']

    wm = EnsembleWorldModel(
        obs_dim,
        act_dim,
        wm_cfg['ensemble_size'],
        wm_cfg['hidden_dim'],
        wm_cfg['lr'],
        wm_cfg['weight_decay'],
        device,
        precision,
        stats_momentum=wm_cfg.get('stats_momentum', 0.99),
        bootstrap_prob=wm_cfg.get('bootstrap_prob', 0.8),
        grad_clip=wm_cfg.get('grad_clip', 10.0),
    )
    wm_buf = TransitionReplay(max(total + 10000, 100000), obs_dim, act_dim)

    warmup_steps = int(tc['warmup_steps'])
    for _ in _progress(
        range(warmup_steps),
        show_progress,
        total=warmup_steps,
        desc=f"warmup {cfg['env']['id']} seed={seed}",
        leave=False,
    ):
        a = env.action_space.sample()
        no, r, term, trunc, _ = env.step(a)
        done = term or trunc
        wm_buf.add(obs, a, r, no, done)
        obs = no
        if done:
            obs, _ = env.reset()

    wm_loss = wm.fit_batch(
        wm_buf.sample(min(wm_buf.size, 10000)),
        updates=int(tc['wm_updates']),
        batch_size=int(tc['batch_size']),
    )

    planner = _make_planner(wm, low, high, mp, wm_cfg)

    k = 1 if method == 'action_residual' else int(rc['chunk_len'])
    residual_dim = act_dim
    context_dim = obs_dim + k * act_dim + 1
    temporal_decay = float(rc.get('temporal_decay', 0.8))
    gate_power = float(rc.get('gate_power', 1.0)) if method == 'planning_residual' else 1.0

    agent = None
    rb = None
    if method != 'mpc_only':
        agent = ResidualSAC(
            context_dim,
            residual_dim,
            sc['hidden_dim'],
            sc['lr'],
            sc['gamma'],
            sc['tau'],
            sc['init_alpha'],
            sc['target_entropy_scale'],
            rc.get('consistency_coef', 0.0) if method == 'planning_residual' else 0.0,
            device,
            precision,
            critic=sc.get('critic', 'dueling'),
            obs_dim=obs_dim if sc.get('terminal_value', True) else None,
            actor_delay=int(sc.get('actor_delay', 2)),
            target_policy_noise=float(sc.get('target_policy_noise', 0.0)),
            trust_region_relative=bool(sc.get('trust_region_relative', True)),
        )
        rb = ResidualReplay(
            max(total + 10000, 100000), context_dim, residual_dim,
            default_discount=sc['gamma'],
        )

    normalized_gate = (
        make_uncertainty_gate(rc)
        if method == 'planning_residual' and rc.get('adaptive_gate', True)
        else None
    )
    rl_start_local = max(1, int(tc['rl_start_steps']) - warmup_steps)
    ramp_steps = int(rc.get('ramp_steps', 1))
    # One residual decision is held for this many environment steps. The base plan
    # is still replanned every step (the controller stays closed-loop), but the
    # residual's reward is accumulated over the whole hold, which raises the
    # variance of the return attributable to the residual by ~k and its
    # signal-to-noise by ~sqrt(k). At 1 this reduces to the old per-step residual.
    hold_steps = max(1, int(rc.get('residual_hold_steps', 1)))
    wm_freeze_step = int(tc.get('wm_freeze_step', 0))
    cache = PlanCache()

    root = Path(cfg['logging'].get('root', 'runs'))
    name = run_name or f"{cfg['env']['id']}__{method}__seed{seed}__{int(time.time())}"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- resume
    start_step = 1
    resume_ck = _latest_checkpoint(out) if resume else None
    if resume_ck is not None:
        ck = torch.load(resume_ck, map_location=device)
        if ck.get('method') != method:
            raise SystemExit(
                f"{resume_ck} was written by method={ck.get('method')!r}, not {method!r}."
            )
        wm.load_state_dict(ck['world_model'])
        if agent is not None and 'agent' in ck:
            agent.load_bundle(ck['agent'])
        if normalized_gate is not None and 'gate_state' in ck:
            normalized_gate.load_state_dict(ck['gate_state'])
        if 'wm_buf' in ck:
            _restore_buffer(wm_buf, ck['wm_buf'], ('obs', 'action', 'reward', 'next_obs', 'done'))
        if rb is not None and 'residual_buf' in ck:
            _restore_buffer(rb, ck['residual_buf'],
                            ('context', 'residual', 'reward', 'next_context', 'done', 'discount'))
        start_step = int(ck['step']) + 1
        print(f'resumed from {resume_ck} at step {start_step}', flush=True)
        if 'wm_buf' not in ck:
            print(
                'WARNING: checkpoint has no replay buffer (train.checkpoint_replay was false), '
                'so the world model resumes with an empty buffer and will refit from new data.',
                flush=True,
            )

    save_yaml(cfg, out / 'config.yaml')

    fields = [
        'global_step', 'episode', 'method', 'env', 'seed',
        'episode_return', 'episode_length', 'success', 'mpc_ms',
        'prediction_mse', 'uncertainty', 'residual_norm',
        'effective_residual_norm', 'gate', 'adaptive_gate', 'gate_z',
        'residual_ramp', 'mpc_cache_hit_rate', 'safeguard_scale',
        'safeguard_gain', 'action_d1', 'action_d2', 'wm_loss',
        'q_loss', 'actor_loss', 'alpha', 'v_loss',
        'adv_q_std_ratio', 'adv_mean',
    ]
    logger = CSVLogger(str(out / 'episodes.csv'), fields,
                       on_exists='append' if resume_ck is not None else 'rotate')
    metrics = EpisodeMetrics()
    episode = 0
    last_sac = {}
    obs, _ = env.reset(seed=seed + 1)
    planner.reset()
    cache.clear()
    console_every = max(1, int(tc.get('console_log_interval_episodes', 10)))

    pbar = _progress(
        range(start_step, total + 1),
        show_progress,
        total=total - start_step + 1,
        desc=f"{cfg['env']['id']} | {method} | seed={seed}",
    )

    # State for the current residual hold window.
    hold = None
    wm_frozen = False

    for step in pbar:
        plan, cache_hit = cache.get_or_plan(obs, planner)
        base = plan_chunk(plan.actions, k)
        safeguard_scale = np.nan
        safeguard_gain = np.nan

        if agent is None:
            adaptive_gate = 0.0
            gate_z = np.nan
            ramp = 0.0
            effective_gate = 0.0
            residual = np.zeros(act_dim, dtype=np.float32)
            corrected = base
            effective_delta = np.zeros_like(base)
            c = None
        else:
            new_decision = hold is None or hold['left'] <= 0
            adaptive_gate, gate_z = _adaptive_gate_for_plan(
                method, rc, normalized_gate, plan.uncertainty, update=new_decision
            )
            ramp = residual_ramp(step, rl_start_local, ramp_steps)
            effective_gate = gated_strength(adaptive_gate, ramp, gate_power)
            c = residual_context(obs, base, effective_gate)

            if new_decision:
                residual = agent.act(c).reshape(act_dim)
                hold = {
                    'context': c,
                    'residual': residual,
                    'reward': 0.0,
                    'discount': 1.0,
                    'left': hold_steps,
                    'applied': 0.0,
                    'n': 0,
                }
            else:
                # Keep acting on the residual chosen at the start of the window so
                # the stored transition describes a single, coherent decision.
                residual = hold['residual']

            if method == 'planning_residual':
                residual_chunk = expand_temporal_residual(
                    residual, k, act_dim, decay=temporal_decay
                )
            else:
                residual_chunk = residual.reshape(1, act_dim)

            proposed_delta = (
                effective_gate
                * float(rc['residual_scale'])
                * residual_chunk
                * span[None, :]
            )
            if method == 'planning_residual':
                corrected, effective_delta, safeguard_scale, safeguard_gain = (
                    _apply_planning_safeguard(
                        cfg, wm, obs, plan, base, proposed_delta, low, high,
                        agent=agent, context=c, residual=residual,
                    )
                )
            else:
                effective_delta = proposed_delta
                corrected = base + effective_delta

        action = np.clip(corrected[0], low, high)
        no, r, term, trunc, info = env.step(action)
        # `terminated` and `truncated` mean different things for bootstrapping:
        # a time-limit truncation is NOT an absorbing state, so its value must
        # still be bootstrapped. HalfCheetah and Reacher never terminate, so
        # collapsing the two taught the critic that the episode genuinely ends.
        episode_over = bool(term or trunc)
        bootstrap_done = float(term)

        with torch.no_grad():
            pred = wm.predict(obs[None], action[None])
            pe = float(torch.mean(
                (pred.next_obs.squeeze(0) - torch.as_tensor(no, device=device)) ** 2
            ).cpu())

        wm_buf.add(obs, action, r, no, bootstrap_done)
        if wm_freeze_step and not wm_frozen and step >= wm_freeze_step:
            # A world model that keeps moving makes the MPC base policy -- and so
            # the residual's MDP -- non-stationary for the whole run.
            wm.freeze()
            wm_frozen = True
            tqdm.write(f'world model frozen at step {step}') if show_progress else print(
                f'world model frozen at step {step}', flush=True)
        if not wm_frozen and step % int(tc['wm_update_interval']) == 0:
            wm_loss = wm.fit_batch(
                wm_buf.sample(min(wm_buf.size, 10000)),
                updates=int(tc['wm_updates']),
                batch_size=int(tc['batch_size']),
            )

        if agent is not None:
            # Credit the residual that was actually EXECUTED, not the one proposed.
            # With a shield active the executed scale is often 0, and recording the
            # raw proposal against a reward it did not cause is pure label noise.
            applied_scale = 1.0 if np.isnan(safeguard_scale) else float(safeguard_scale)
            hold['applied'] += applied_scale
            hold['n'] += 1
            hold['reward'] += hold['discount'] * float(r)
            hold['discount'] *= float(sc['gamma'])
            hold['left'] -= 1

            window_over = episode_over or hold['left'] <= 0 or step == total
            if window_over:
                if episode_over:
                    if term:
                        nc = np.zeros(context_dim, np.float32)
                    else:
                        # Truncation: `no` is a real state, so build its real
                        # context instead of a zero vector the critic can never see.
                        nplan = planner.plan(no)
                        nbase = plan_chunk(nplan.actions, k)
                        next_adaptive, _ = _adaptive_gate_for_plan(
                            method, rc, normalized_gate, nplan.uncertainty, update=False
                        )
                        next_ramp = residual_ramp(step + 1, rl_start_local, ramp_steps)
                        nc = residual_context(
                            no, nbase, gated_strength(next_adaptive, next_ramp, gate_power)
                        )
                    cache.clear()
                else:
                    nplan = planner.plan(no)
                    nbase = plan_chunk(nplan.actions, k)
                    next_adaptive, _ = _adaptive_gate_for_plan(
                        method, rc, normalized_gate, nplan.uncertainty, update=False
                    )
                    next_ramp = residual_ramp(step + 1, rl_start_local, ramp_steps)
                    next_gate = gated_strength(next_adaptive, next_ramp, gate_power)
                    nc = residual_context(no, nbase, next_gate)
                    cache.put(no, nplan)

                executed = hold['residual'] * (hold['applied'] / max(hold['n'], 1))
                rb.add(
                    hold['context'], executed, hold['reward'], nc,
                    bootstrap_done, discount=hold['discount'],
                )
                hold = None
            elif not episode_over:
                # Mid-window: still refresh the plan cache so the next step reuses
                # this plan instead of paying for a second CEM solve.
                nplan = planner.plan(no)
                cache.put(no, nplan)

        metrics.step(
            r,
            action,
            plan.planning_ms,
            pe,
            plan.uncertainty,
            float(np.linalg.norm(residual)),
            effective_gate,
            adaptive_gate=adaptive_gate,
            gate_z=gate_z,
            ramp=ramp,
            effective_resnorm=float(np.linalg.norm(effective_delta)),
            cache_hit=cache_hit,
            safeguard_scale=safeguard_scale,
            safeguard_gain=safeguard_gain,
        )

        if agent is not None and rb.size >= int(tc['batch_size']) and step >= rl_start_local:
            for _ in range(int(tc['rl_updates_per_step'])):
                last_sac = agent.update(rb.sample(int(tc['batch_size'])))

        if episode_over:
            row = metrics.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({
                'global_step': step,
                'episode': episode,
                'method': method,
                'env': cfg['env']['id'],
                'seed': seed,
                'wm_loss': wm_loss,
            })
            if agent is not None:
                diag = agent.diagnostics(rb.sample(min(rb.size, 1024))) if rb.size > 1 else {}
                row.update({
                    'q_loss': last_sac.get('q_loss', np.nan),
                    'actor_loss': last_sac.get('actor_loss', np.nan),
                    'alpha': last_sac.get('alpha', np.nan),
                    'v_loss': last_sac.get('v_loss', np.nan),
                    'adv_q_std_ratio': diag.get('adv_q_std_ratio', np.nan),
                    'adv_mean': diag.get('adv_mean', np.nan),
                })
            logger.write(row)
            episode += 1
            metrics.reset()
            obs, _ = env.reset()
            planner.reset()
            cache.clear()
            hold = None

            if show_progress:
                pbar.set_postfix({
                    'ep': episode,
                    'ret': f"{row['episode_return']:.1f}",
                    'len': int(row['episode_length']),
                    'wm': f"{wm_loss:.4f}",
                    'gate': f"{row['gate']:.2f}" if np.isfinite(row['gate']) else 'nan',
                    'safe': f"{row['safeguard_scale']:.2f}" if np.isfinite(row['safeguard_scale']) else 'nan',
                    'A/Q': f"{row.get('adv_q_std_ratio', np.nan):.1e}" if agent is not None else '-',
                })
                if episode % console_every == 0:
                    tqdm.write(json.dumps(row, ensure_ascii=False))
            else:
                print(json.dumps(row, ensure_ascii=False))
        else:
            obs = no

        if step % int(tc.get('checkpoint_interval', 5000)) == 0:
            _save_checkpoint(out / f'checkpoint_{step}.pt', wm, agent, normalized_gate,
                             step, method, wm_buf, rb, keep_replay=bool(tc.get('checkpoint_replay', False)))

    _save_checkpoint(out / 'final.pt', wm, agent, normalized_gate, total, method)

    n_eval = int(tc.get('final_eval_episodes', 3) if eval_episodes is None else eval_episodes)
    eval_rows = evaluate_final(
        cfg, method, wm, agent, normalized_gate, device, low, high,
        act_dim, k, temporal_decay, gate_power, n_eval, seed, show_progress,
    )
    if eval_rows:
        eval_fields = [
            'eval_episode', 'eval_seed', 'train_seed', 'method', 'env',
            'episode_return', 'episode_length', 'success', 'mpc_ms',
            'prediction_mse', 'uncertainty', 'residual_norm',
            'effective_residual_norm', 'gate', 'adaptive_gate', 'gate_z',
            'residual_ramp', 'safeguard_scale', 'safeguard_gain',
            'action_d1', 'action_d2',
        ]
        eval_logger = CSVLogger(str(out / 'eval.csv'), eval_fields)
        for row in eval_rows:
            eval_logger.write(row)
        vals = np.asarray([r['episode_return'] for r in eval_rows], dtype=float)
        print(
            f"FINAL_EVAL method={method} seed={seed} episodes={len(vals)} "
            f"return={vals.mean():.3f}+/-{vals.std(ddof=1) if len(vals)>1 else 0.0:.3f}"
        )

    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/halfcheetah.yaml')
    p.add_argument('--method', choices=sorted(METHODS), default='planning_residual')
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--run-name')
    p.add_argument('--eval-episodes', type=int)
    p.add_argument('--no-progress', action='store_true')
    p.add_argument(
        '--resume',
        action='store_true',
        help='continue from the newest checkpoint_*.pt in the run directory. '
             'Requires --run-name so the directory can be found. Set '
             'train.checkpoint_replay: true to also restore the replay buffers.',
    )
    a = p.parse_args()

    if a.resume and not a.run_name:
        raise SystemExit('--resume requires --run-name so the run directory can be located.')

    cfg = load_yaml(a.config)
    if a.seed is not None:
        cfg['env']['seed'] = a.seed
    print(
        'output:',
        train(
            cfg, a.method, a.steps, a.run_name,
            show_progress=not a.no_progress,
            eval_episodes=a.eval_episodes,
            resume=a.resume,
        ),
    )


if __name__ == '__main__':
    main()
