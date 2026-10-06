from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch


SAFEGUARD_MODES = ('off', 'critic', 'model', 'model_value')


@dataclass
class SafeguardResult:
    scale: float
    predicted_gain: float
    corrected_plan: np.ndarray


def expand_temporal_residual(residual, chunk_len: int, action_dim: int, decay: float = 0.8) -> np.ndarray:
    """Expand one action-space residual into a smooth MPC-plan correction."""
    k = max(int(chunk_len), 1)
    ad = int(action_dim)
    d = float(np.clip(decay, 0.0, 1.0))
    r = np.asarray(residual, dtype=np.float32).reshape(ad)
    scales = np.power(d, np.arange(k, dtype=np.float32))[:, None]
    return scales * r[None, :]


def _candidate_plans(base, delta, low, high, scales):
    k = min(len(delta), len(base))
    candidates = []
    clean = []
    for value in tuple(scales) + (0.0,):
        s = float(np.clip(value, 0.0, 1.0))
        if any(abs(s - old) < 1e-12 for old in clean):
            continue
        plan = base.copy()
        plan[:k] = np.clip(base[:k] + s * delta[:k], low, high)
        candidates.append(plan)
        clean.append(s)
    return candidates, clean


def _pick(scores, clean_scales, min_improvement):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.any(np.isfinite(scores)):
        # Every candidate scored NaN/inf: fall back to the unmodified plan.
        return clean_scales.index(0.0), 0.0, 0.0
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    baseline_idx = clean_scales.index(0.0)
    baseline_score = float(scores[baseline_idx])
    best_idx = int(np.argmax(scores))
    best_gain = float(scores[best_idx] - baseline_score)
    best_scale = clean_scales[best_idx]
    if best_scale == 0.0 or not np.isfinite(best_gain) or best_gain < float(min_improvement):
        return baseline_idx, 0.0, 0.0
    return best_idx, best_scale, best_gain


def safeguard_residual_plan(
    world_model,
    obs,
    base_plan,
    delta_chunk,
    action_low,
    action_high,
    discount=0.99,
    uncertainty_penalty=0.0,
    scales=(1.0, 0.5, 0.25, 0.0),
    min_improvement=0.0,
    mode='model',
    agent=None,
    context=None,
    residual=None,
    terminal_value_fn=None,
) -> SafeguardResult:
    """Shrink or reject a residual correction before it reaches the environment.

    ``mode`` decides *what evidence* the veto is based on, and this choice is the
    difference between a shield that adds information and one that cannot:

    - ``model``: re-score with the same world model and the same H-step rollout
      objective that CEM already approximately argmaxed. Because the base plan is
      near that objective's optimum, a correction can essentially only be accepted
      when it recovers CEM's own optimisation slack -- it can never let the
      residual correct model bias or reach beyond the horizon. This caps the method
      at approximately MPC and is kept only as an explicit ablation.
    - ``model_value``: same rollout, plus ``discount**H * V(s_H)`` from the agent's
      state-value head. Now the shield scores something CEM did not: at gamma=0.99,
      H=20 that terminal term carries >80% of the discounted return.
    - ``critic``: accept iff the learned advantage ``A(context, scale*residual)``
      beats the unmodified plan by ``min_improvement``. Grounded in observed
      environment reward rather than in the model CEM optimised.
    - ``off``: apply the residual unchanged.
    """
    base = np.asarray(base_plan, dtype=np.float32)
    delta = np.asarray(delta_chunk, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f'base_plan must be [H,A], got {base.shape}')
    if delta.ndim != 2 or delta.shape[1] != base.shape[1]:
        raise ValueError(f'delta_chunk must be [K,{base.shape[1]}], got {delta.shape}')
    if min(len(delta), len(base)) <= 0:
        return SafeguardResult(0.0, 0.0, base.copy())

    if mode == 'off':
        k = min(len(delta), len(base))
        plan = base.copy()
        plan[:k] = np.clip(base[:k] + delta[:k], low, high)
        return SafeguardResult(1.0, float('nan'), plan)

    if mode not in SAFEGUARD_MODES:
        raise ValueError(f'mode must be one of {SAFEGUARD_MODES}, got {mode!r}')

    candidates, clean_scales = _candidate_plans(base, delta, low, high, scales)

    if mode == 'critic':
        if agent is None or context is None or residual is None:
            raise ValueError("mode='critic' requires agent, context and residual")
        ctx = np.asarray(context, dtype=np.float32).reshape(1, -1)
        res = np.asarray(residual, dtype=np.float32).reshape(1, -1)
        ctxs = np.repeat(ctx, len(clean_scales), axis=0)
        residuals = np.concatenate([s * res for s in clean_scales], axis=0)
        scores = agent.advantage_batch(ctxs, residuals)
    else:
        kwargs = {}
        if mode == 'model_value':
            if terminal_value_fn is None:
                raise ValueError("mode='model_value' requires terminal_value_fn")
            kwargs['terminal_value_fn'] = terminal_value_fn
        with torch.no_grad():
            score_t, _ = world_model.rollout_return(
                obs,
                np.stack(candidates, axis=0),
                float(discount),
                float(uncertainty_penalty),
                **kwargs,
            )
        scores = score_t.detach().float().cpu().numpy()

    best_idx, best_scale, best_gain = _pick(scores, clean_scales, min_improvement)
    return SafeguardResult(
        scale=float(best_scale),
        predicted_gain=float(best_gain),
        corrected_plan=np.asarray(candidates[best_idx], dtype=np.float32),
    )
