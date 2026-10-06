from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ActionSafeguardResult:
    action: np.ndarray
    effective_residual: np.ndarray
    scale: float
    predicted_gain: float


def normalize_action(action, low, high):
    """Map an environment action to approximately [-1, 1]."""
    action = np.asarray(action, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    half = np.maximum((high - low) * 0.5, 1e-6)
    mid = (high + low) * 0.5
    return np.clip((action - mid) / half, -1.0, 1.0).astype(np.float32)


def apply_action_residual(base_action, residual, low, high, scale, ramp=1.0):
    """Apply a bounded RL residual to the first MPC action."""
    base = np.asarray(base_action, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    half = (high - low) * 0.5
    effective = np.clip(float(ramp), 0.0, 1.0) * float(scale) * half * residual
    action = np.clip(base + effective, low, high).astype(np.float32)
    return action, effective.astype(np.float32)


def safeguard_action_residual(
    planner,
    frame,
    base_plan,
    residual,
    low,
    high,
    residual_scale,
    ramp=1.0,
    scales=(1.0, 0.5, 0.25, 0.0),
    min_improvement=0.0,
):
    """Shrink/reject an RL action residual using the frozen LeWM planner score.

    Candidate sequences share the original MPC plan; only the first action is
    corrected.  Scale 0 is always evaluated, so the learned residual cannot be
    selected when the frozen LeWM+reward-probe predicts it to underperform the
    original MPC plan by the configured margin.
    """
    plan = np.asarray(base_plan, dtype=np.float32)
    if plan.ndim != 2 or len(plan) == 0:
        raise ValueError(f'base_plan must be non-empty [H,A], got {plan.shape}')
    base = plan[0]
    _, requested = apply_action_residual(
        base, residual, low, high, residual_scale, ramp=ramp
    )
    clean_scales = []
    candidates = []
    for value in tuple(scales) + (0.0,):
        s = float(np.clip(value, 0.0, 1.0))
        if any(abs(s - old) < 1e-12 for old in clean_scales):
            continue
        candidate = plan.copy()
        candidate[0] = np.clip(base + s * requested, low, high)
        candidates.append(candidate)
        clean_scales.append(s)

    scores = np.asarray(
        planner.score_sequences(frame, np.stack(candidates, axis=0)),
        dtype=np.float64,
    ).reshape(-1)
    baseline_idx = clean_scales.index(0.0)
    baseline_score = float(scores[baseline_idx])
    best_idx = int(np.nanargmax(scores))
    best_gain = float(scores[best_idx] - baseline_score)
    best_scale = clean_scales[best_idx]
    if best_scale == 0.0 or best_gain < float(min_improvement):
        best_idx = baseline_idx
        best_scale = 0.0
        best_gain = 0.0

    action = candidates[best_idx][0].astype(np.float32)
    effective = (action - base).astype(np.float32)
    return ActionSafeguardResult(action, effective, float(best_scale), best_gain)


def effective_rank(latents, eps=1e-12):
    """Entropy effective rank of a batch of latent representations."""
    if torch.is_tensor(latents):
        x = latents.detach().float().cpu().numpy()
    else:
        x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        return float('nan')
    x = x - np.mean(x, axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    s = np.maximum(s, 0.0)
    total = float(s.sum())
    if not np.isfinite(total) or total <= eps:
        return 0.0
    p = s / total
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def residual_ramp(step, start_steps, ramp_steps):
    if step < int(start_steps):
        return 0.0
    if int(ramp_steps) <= 0:
        return 1.0
    return float(np.clip((step - int(start_steps)) / float(ramp_steps), 0.0, 1.0))
