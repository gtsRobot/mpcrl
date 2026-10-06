from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch


# Sentinel used to replace non-finite candidate scores. Must be far below any
# realistic return so such candidates never win, while still being orderable
# (unlike NaN, which torch.topk sorts as *greater* than every finite value).
_BAD_SCORE = -1e30

EXECUTE_MODES = ('best_sample', 'elite_mean', 'cem_mean')


@dataclass
class PlanResult:
    actions: np.ndarray
    score: float
    uncertainty: float
    planning_ms: float
    degenerate: bool = False


class CEMPlanner:
    """Cross-entropy-method planner over a learned world model.

    ``execute`` selects which sequence is returned for execution:

    - ``best_sample``: the single highest-scoring sampled sequence. Maximises the
      learned model's predicted return and therefore also maximises *model
      exploitation*; kept as the default only so that existing configs do not
      silently change behaviour.
    - ``elite_mean``: the mean of the elite set (PETS / TD-MPC convention). Much
      less prone to exploiting model error; recommended.
    - ``cem_mean``: the final smoothed distribution mean.
    """

    def __init__(self, world_model, action_low, action_high, horizon=15, candidates=512,
                 elites=64, iterations=5, alpha=0.2, init_std=0.6, min_std=0.05,
                 discount=0.99, uncertainty_penalty=0.0, execute='best_sample'):
        self.model = world_model
        self.low = np.asarray(action_low, np.float32); self.high = np.asarray(action_high, np.float32)
        self.horizon, self.candidates, self.iterations = horizon, candidates, iterations
        # A config that asks for more elites than candidates is a configuration
        # error, not a reason to crash inside torch.topk. Clamp once, here.
        self.elites = max(1, min(int(elites), int(candidates)))
        self.alpha, self.init_std, self.min_std, self.discount = alpha, init_std, min_std, discount
        self.uncertainty_penalty = uncertainty_penalty
        if execute not in EXECUTE_MODES:
            raise ValueError(f'execute must be one of {EXECUTE_MODES}, got {execute!r}')
        self.execute = execute
        self.action_dim = len(self.low); self.prev_mean = None

    def reset(self): self.prev_mean = None

    def _init_mean(self):
        if self.prev_mean is None: return np.tile((self.low+self.high)/2, (self.horizon,1))
        shifted = np.concatenate([self.prev_mean[1:], self.prev_mean[-1:]], axis=0)
        return shifted.copy()

    def plan(self, obs) -> PlanResult:
        t0=time.perf_counter(); device=self.model.device
        mean = torch.as_tensor(self._init_mean(), dtype=torch.float32, device=device)
        span = torch.as_tensor((self.high-self.low)/2, dtype=torch.float32, device=device)
        std = span * self.init_std
        low=torch.as_tensor(self.low,device=device); high=torch.as_tensor(self.high,device=device)

        # Fall back to the (shifted, clipped) warm-start plan rather than None if
        # every CEM iteration turns out degenerate. A None here used to propagate
        # into plan_chunk() as `TypeError: 'NoneType' object is not subscriptable`
        # and kill multi-hour runs.
        fallback = mean.clamp(low, high).detach().cpu().numpy()
        best_sample_seq = None
        best_score = -float('inf'); best_sample_unc = 0.0
        last_elite_mean = None; last_elite_unc = 0.0
        degenerate = True

        for _ in range(self.iterations):
            noise=torch.randn(self.candidates,self.horizon,self.action_dim,device=device)
            seq=(mean.unsqueeze(0)+noise*std.unsqueeze(0)).clamp(low,high)
            scores, unc = self.model.rollout_return(obs, seq, self.discount, self.uncertainty_penalty)
            # A single NaN would otherwise be ranked top by topk and then fail the
            # `> best_score` test, silently skipping this whole iteration, while
            # still polluting the elite set and dragging mean/std with it.
            scores = torch.nan_to_num(scores, nan=_BAD_SCORE, posinf=_BAD_SCORE, neginf=_BAD_SCORE)
            unc = torch.nan_to_num(unc, nan=0.0, posinf=0.0, neginf=0.0)

            vals, idx = torch.topk(scores, k=self.elites, largest=True)
            elite=seq[idx]; new_mean=elite.mean(0); new_std=elite.std(0,unbiased=False).clamp_min(self.min_std)

            if float(vals[0]) > _BAD_SCORE:
                degenerate = False
                last_elite_mean = new_mean.clamp(low, high)
                last_elite_unc = float(unc[idx].mean())
                if float(vals[0]) > best_score:
                    best_score = float(vals[0])
                    best_sample_seq = seq[idx[0]].detach().cpu().numpy()
                    best_sample_unc = float(unc[idx[0]])

            mean=self.alpha*mean+(1-self.alpha)*new_mean
            std=self.alpha*std+(1-self.alpha)*new_std

        # Selection semantics differ per mode on purpose: `best_sample` is an
        # argmax over every sequence ever scored, whereas the two averaging modes
        # take the *final* iteration's estimate, since CEM has converged by then.
        if self.execute == 'best_sample':
            actions = best_sample_seq if best_sample_seq is not None else fallback
            uncertainty = best_sample_unc
        elif self.execute == 'elite_mean':
            actions = last_elite_mean.detach().cpu().numpy() if last_elite_mean is not None else fallback
            uncertainty = last_elite_unc
        else:  # cem_mean
            actions = mean.clamp(low, high).detach().cpu().numpy()
            uncertainty = last_elite_unc

        self.prev_mean = mean.detach().cpu().numpy()
        ms=(time.perf_counter()-t0)*1000.0
        return PlanResult(actions, best_score, uncertainty, ms, degenerate)
