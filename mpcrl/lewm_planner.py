"""Latent CEM planner over a frozen LeWM.

Differences from the previous version, all of which were reachable bugs:

- ``torch.topk(score, self.k)`` had no ``min(elites, candidates)`` guard, so
  lowering ``mpc.candidates`` for a light local test raised
  ``RuntimeError: selected index k out of range``.
- ``best`` was initialised to ``None`` and only assigned inside
  ``if float(vals[0]) > best_score``. Since ``torch.topk`` ranks NaN above every
  finite value, one NaN candidate made that test false for an entire iteration;
  if it held for all iterations ``plan.actions`` was ``None`` and the caller
  crashed on ``plan.actions[0]``.
- No warm start and no mean/std momentum, unlike the state-space ``CEMPlanner``,
  which made this baseline gratuitously weaker and less temporally consistent --
  and the paper's headline number is the paired lewm_mpc_rl vs lewm_mpc delta.
- ``amp_dtype`` was computed and never used, so ``precision: bf16`` was a no-op
  for the dominant cost.

The predictor is autoregressive over an observation history, so the planner
carries a latent history deque and rolls it forward per candidate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


_BAD_SCORE = -1e30


@dataclass
class LatentPlan:
    actions: np.ndarray
    score: float
    planning_ms: float
    degenerate: bool = False


class LeWMCEMPlanner:
    def __init__(
        self, model, reward_probe, low, high, horizon=12, candidates=256,
        elites=32, iterations=4, init_std=.7, min_std=.05, discount=.99,
        device='cpu', precision='fp32', alpha=0.2, execute='elite_mean',
    ):
        self.model = model
        self.probe = reward_probe
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.h = int(horizon)
        self.n = int(candidates)
        self.k = max(1, min(int(elites), int(candidates)))
        self.it = int(iterations)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.discount = float(discount)
        self.alpha = float(alpha)
        self.execute = execute
        self.device = torch.device(device)
        self.amp_dtype = torch.bfloat16 if precision == 'bf16' else (
            torch.float16 if precision == 'fp16' else None)
        self.history = int(getattr(model, 'history', 1))
        self.prev_mean = None
        self._frames = []

    # ------------------------------------------------------------------ history
    def reset(self):
        self.prev_mean = None
        self._frames = []

    def observe(self, frame):
        """Push a frame onto the history deque used for action-conditioned rollout.

        Idempotent for a repeated frame: ``plan(frame)`` and then
        ``score_sequences(frame, ...)`` (which the action safeguard does for the
        same observation) must not push the same frame twice and corrupt history.
        """
        arr = np.ascontiguousarray(frame)
        if self._frames and np.array_equal(self._frames[-1], arr):
            return
        self._frames.append(arr)
        if len(self._frames) > self.history:
            self._frames = self._frames[-self.history:]

    def _as_batch(self, frames):
        x = torch.as_tensor(np.stack(frames), device=self.device)
        return x.permute(0, 3, 1, 2)

    def _encode_current(self, frame, count):
        """Encode the newest frame plus history, expanded to `count` candidates."""
        self.observe(frame)
        frames = list(self._frames)
        while len(frames) < self.history:
            frames.insert(0, frames[0])
        z_all = self.model.encode(self._as_batch(frames))          # [H, T, D]
        z = z_all[-1:].expand(int(count), -1, -1)                  # newest
        hist = (z_all[:-1].unsqueeze(0).expand(int(count), -1, -1, -1)
                if self.history > 1 else None)
        return z.contiguous(), (hist.contiguous() if hist is not None else None)

    # -------------------------------------------------------------------- scoring
    def _score_tensor(self, z, hist, seq):
        ret = torch.zeros(len(seq), device=self.device)
        g = 1.0
        for t in range(seq.shape[1]):
            a = seq[:, t]
            ret = ret + g * self.probe(z, a).float().squeeze(-1)
            nxt = self.model.latent_step(z, a, history=hist)
            if hist is not None:
                hist = torch.cat([hist[:, 1:], z.unsqueeze(1)], dim=1) if hist.shape[1] > 1 \
                    else z.unsqueeze(1)
            z = nxt
            g *= self.discount
        return ret

    @torch.no_grad()
    def score_sequences(self, frame, action_sequences):
        """Score fixed action sequences with the same frozen LeWM/reward probe."""
        seq = torch.as_tensor(action_sequences, dtype=torch.float32, device=self.device)
        if seq.ndim != 3 or seq.shape[-1] != len(self.low):
            raise ValueError(
                f'action_sequences must be [N,H,{len(self.low)}], got {tuple(seq.shape)}')
        amp = self.device.type == 'cuda' and self.amp_dtype is not None
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp):
            z, hist = self._encode_current(frame, len(seq))
            out = self._score_tensor(z, hist, seq)
        return torch.nan_to_num(out.float(), nan=_BAD_SCORE).cpu().numpy()

    # ----------------------------------------------------------------- planning
    def _init_mean(self):
        mid = (self.low + self.high) / 2
        if self.prev_mean is None:
            return np.tile(mid, (self.h, 1))
        return np.concatenate([self.prev_mean[1:], self.prev_mean[-1:]], axis=0).copy()

    @torch.no_grad()
    def plan(self, frame):
        t0 = time.perf_counter()
        ad = len(self.low)
        low = torch.as_tensor(self.low, device=self.device)
        high = torch.as_tensor(self.high, device=self.device)
        mean = torch.as_tensor(self._init_mean(), dtype=torch.float32, device=self.device)
        std = ((high - low) / 2 * self.init_std).repeat(self.h, 1)

        fallback = mean.clamp(low, high).detach().cpu().numpy()
        best_sample = None
        best_score = -float('inf')
        last_elite_mean = None
        degenerate = True

        amp = self.device.type == 'cuda' and self.amp_dtype is not None
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp):
            z0, hist0 = self._encode_current(frame, self.n)
            for _ in range(self.it):
                seq = (mean[None] + torch.randn(self.n, self.h, ad, device=self.device)
                       * std[None]).clamp(low, high)
                score = self._score_tensor(z0, hist0, seq).float()
                score = torch.nan_to_num(score, nan=_BAD_SCORE, posinf=_BAD_SCORE,
                                         neginf=_BAD_SCORE)
                vals, idx = torch.topk(score, self.k)
                elite = seq[idx]
                new_mean = elite.mean(0)
                new_std = elite.std(0, unbiased=False).clamp_min(self.min_std)
                if float(vals[0]) > _BAD_SCORE:
                    degenerate = False
                    last_elite_mean = new_mean.clamp(low, high)
                    if float(vals[0]) > best_score:
                        best_score = float(vals[0])
                        best_sample = seq[idx[0]].detach().cpu().numpy()
                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                std = self.alpha * std + (1 - self.alpha) * new_std

        if self.execute == 'best_sample':
            actions = best_sample if best_sample is not None else fallback
        elif self.execute == 'elite_mean':
            actions = (last_elite_mean.detach().cpu().numpy()
                       if last_elite_mean is not None else fallback)
        else:
            actions = mean.clamp(low, high).detach().cpu().numpy()

        self.prev_mean = mean.detach().cpu().numpy()
        return LatentPlan(actions, best_score, (time.perf_counter() - t0) * 1000, degenerate)
