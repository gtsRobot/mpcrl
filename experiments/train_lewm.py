from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mpcrl.config import load_yaml, save_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.lewm import LeWorldModel, RewardProbe
from mpcrl.lewm_planner import LeWMCEMPlanner
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.utils import (
    accelerator_summary,
    autocast_context,
    configure_accelerator,
    set_seed,
)


def resize_frame(img, size):
    x = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return (
        F.interpolate(x, (size, size), mode='bilinear', align_corners=False)
        .squeeze(0)
        .permute(1, 2, 0)
        .byte()
        .numpy()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rtx5090/lewm_halfcheetah.yaml')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--episodes', type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg['env']['seed']
    cfg['env']['seed'] = seed
    set_seed(seed)

    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    size = int(cfg['env']['render_size'])
    env, _, _ = make_mujoco_env(cfg['env']['id'], seed, render_mode='rgb_array')
    _, ad = dims(env)
    low, high = action_bounds(env)

    # ------------------------------------------------------------------
    # 1) Collect a common pixel/action/reward transition set.
    # ------------------------------------------------------------------
    data = []
    frame = resize_frame(env.render(), size)
    for _ in range(int(cfg['collect']['transitions'])):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        next_frame = resize_frame(env.render(), size)
        data.append((frame, action, next_frame, reward))
        frame = next_frame
        if terminated or truncated:
            env.reset()
            frame = resize_frame(env.render(), size)

    # ------------------------------------------------------------------
    # 2) Train the adapted LeWorldModel objective.
    # ------------------------------------------------------------------
    lc = cfg['lewm']
    model = LeWorldModel(
        ad,
        latent_dim=lc['latent_dim'],
        hidden=lc.get('hidden_dim', 256),
        sigreg_lambda=lc['sigreg_lambda'],
        sigreg_projections=lc['sigreg_projections'],
        sigreg_knots=lc['sigreg_knots'],
        encoder=lc.get('encoder', 'vit'),
        predictor=lc.get('predictor', 'ar_transformer'),
        image_size=size,
        patch_size=int(lc.get('patch_size', 16)),
        encoder_depth=int(lc.get('encoder_depth', 6)),
        encoder_heads=int(lc.get('encoder_heads', 3)),
        predictor_depth=int(lc.get('predictor_depth', 4)),
        predictor_heads=int(lc.get('predictor_heads', 3)),
        # This entry point trains on single transitions without a history buffer,
        # so the predictor is configured history=1 rather than silently padding
        # by repeating the current frame.
        history=1,
        sigreg_scale_by_n=bool(lc.get('sigreg_scale_by_n', False)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lc['lr']))
    batch_size = int(lc['batch_size'])

    for epoch in range(int(lc['epochs'])):
        random.shuffle(data)
        logs = []
        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = data[i:i + batch_size]
            frames = torch.as_tensor(
                np.stack([x[0] for x in batch]), device=device
            ).permute(0, 3, 1, 2)
            actions = torch.as_tensor(
                np.stack([x[1] for x in batch]), dtype=torch.float32, device=device
            )
            next_frames = torch.as_tensor(
                np.stack([x[2] for x in batch]), device=device
            ).permute(0, 3, 1, 2)
            with autocast_context(device, precision):
                loss, info = model.loss(frames, actions, next_frames)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            logs.append(info)
        print(
            f"LeWM epoch {epoch + 1}: "
            f"total={np.mean([x['total'] for x in logs]):.5f} "
            f"pred={np.mean([x['pred'] for x in logs]):.5f} "
            f"sigreg={np.mean([x['sigreg'] for x in logs]):.5f}"
        )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # ------------------------------------------------------------------
    # 3) MuJoCo adaptation: fit a reward probe on frozen LeWM latents.
    # This is deliberately kept separate from the two-term LeWM objective.
    # ------------------------------------------------------------------
    probe = RewardProbe(lc['latent_dim'], ad).to(device)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=cfg['reward_probe']['lr'])
    for epoch in range(int(cfg['reward_probe']['epochs'])):
        random.shuffle(data)
        losses = []
        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = data[i:i + batch_size]
            frames = torch.as_tensor(
                np.stack([x[0] for x in batch]), device=device
            ).permute(0, 3, 1, 2)
            actions = torch.as_tensor(
                np.stack([x[1] for x in batch]), dtype=torch.float32, device=device
            )
            rewards = torch.as_tensor(
                [[x[3]] for x in batch], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                z = model.encode(frames)
            with autocast_context(device, precision):
                pred_reward = probe(z, actions)
            loss = F.mse_loss(pred_reward.float(), rewards.float())
            probe_opt.zero_grad(set_to_none=True)
            loss.backward()
            probe_opt.step()
            losses.append(float(loss.detach()))
        print(f'Reward probe epoch {epoch + 1}: loss={np.mean(losses):.5f}')

    mc = cfg['mpc']
    planner = LeWMCEMPlanner(
        model,
        probe,
        low,
        high,
        mc['horizon'],
        mc['candidates'],
        mc['elites'],
        mc['iterations'],
        mc['init_std'],
        mc['min_std'],
        mc['discount'],
        device,
        precision,
        alpha=float(mc.get('alpha', 0.2)),
        execute=mc.get('execute', 'elite_mean'),
    )

    run_root = Path(cfg.get('logging', {}).get('root', 'runs'))
    out = run_root / f"{cfg['env']['id']}__lewm_mpc__seed{seed}__{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out / 'config.yaml')

    episode_fields = [
        'global_step', 'episode', 'method', 'env', 'seed', 'episode_return',
        'episode_length', 'success', 'mpc_ms', 'prediction_mse', 'uncertainty',
        'residual_norm', 'effective_residual_norm', 'gate', 'residual_ramp',
        'action_d1', 'action_d2',
    ]
    eval_fields = [
        'eval_episode', 'eval_seed', 'train_seed', 'method', 'env',
        'episode_return', 'episode_length', 'success', 'mpc_ms',
        'prediction_mse', 'uncertainty', 'residual_norm',
        'effective_residual_norm', 'gate', 'residual_ramp', 'action_d1', 'action_d2',
    ]
    episode_log = CSVLogger(str(out / 'episodes.csv'), episode_fields)
    eval_log = CSVLogger(str(out / 'eval.csv'), eval_fields)

    total_steps = 0
    n_eval = int(args.episodes or cfg['eval']['episodes'])
    eval_returns = []
    for episode in range(n_eval):
        eval_seed = 300000 + seed * 1000 + episode
        env.reset(seed=eval_seed)
        planner.reset() if hasattr(planner, 'reset') else None
        em = EpisodeMetrics()
        info = {}
        while True:
            frame = resize_frame(env.render(), size)
            plan = planner.plan(frame)
            _, reward, terminated, truncated, info = env.step(plan.actions[0])
            total_steps += 1
            em.step(reward, plan.actions[0], mpc_ms=plan.planning_ms)
            if terminated or truncated:
                break

        metrics = em.finish(info, cfg['env'].get('success_return_threshold'))
        eval_returns.append(float(metrics['episode_return']))

        train_row = dict(metrics)
        train_row.update(
            global_step=total_steps,
            episode=episode,
            method='lewm_mpc',
            env=cfg['env']['id'],
            seed=seed,
        )
        episode_log.write(train_row)

        eval_row = dict(metrics)
        eval_row.update(
            eval_episode=episode,
            eval_seed=eval_seed,
            train_seed=seed,
            method='lewm_mpc',
            env=cfg['env']['id'],
        )
        eval_log.write(eval_row)
        print(eval_row)

    torch.save(
        {
            'lewm': model.state_dict(),
            'reward_probe': probe.state_dict(),
            'config': cfg,
            'eval_return_mean': float(np.mean(eval_returns)),
        },
        out / 'final.pt',
    )
    env.close()

    std = float(np.std(eval_returns, ddof=1)) if len(eval_returns) > 1 else 0.0
    print(
        f"FINAL_EVAL method=lewm_mpc seed={seed} episodes={len(eval_returns)} "
        f"return={np.mean(eval_returns):.3f}+/-{std:.3f}"
    )
    print('output:', out)


if __name__ == '__main__':
    main()
