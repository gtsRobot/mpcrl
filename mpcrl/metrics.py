from __future__ import annotations

import csv
import os

import numpy as np


def infer_success(info, episode_return, threshold=None):
    for key in ('success', 'is_success'):
        if key in info:
            return float(bool(info[key]))
    if threshold is not None:
        return float(episode_return >= float(threshold))
    return float('nan')


class EpisodeMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ret = 0.0
        self.n = 0
        self.mpc_ms = []
        self.pred_err = []
        self.unc = []
        self.resnorm = []
        self.gates = []
        self.adaptive_gates = []
        self.gate_z = []
        self.ramps = []
        self.effective_resnorm = []
        self.cache_hits = []
        self.safeguard_scales = []
        self.safeguard_gains = []
        self.actions = []

    def step(
        self,
        reward,
        action,
        mpc_ms=np.nan,
        pred_err=np.nan,
        unc=np.nan,
        resnorm=np.nan,
        gate=np.nan,
        adaptive_gate=np.nan,
        gate_z=np.nan,
        ramp=np.nan,
        effective_resnorm=np.nan,
        cache_hit=np.nan,
        safeguard_scale=np.nan,
        safeguard_gain=np.nan,
    ):
        self.ret += float(reward)
        self.n += 1
        self.actions.append(np.asarray(action, np.float32).copy())
        self.mpc_ms.append(mpc_ms)
        self.pred_err.append(pred_err)
        self.unc.append(unc)
        self.resnorm.append(resnorm)
        self.gates.append(gate)
        self.adaptive_gates.append(adaptive_gate)
        self.gate_z.append(gate_z)
        self.ramps.append(ramp)
        self.effective_resnorm.append(effective_resnorm)
        self.cache_hits.append(float(cache_hit) if cache_hit is not None else np.nan)
        self.safeguard_scales.append(safeguard_scale)
        self.safeguard_gains.append(safeguard_gain)

    def finish(self, info, success_threshold=None):
        a = np.asarray(self.actions)
        d1 = np.linalg.norm(np.diff(a, axis=0), axis=1).mean() if len(a) > 1 else np.nan
        d2 = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1).mean() if len(a) > 2 else np.nan

        def nm(x):
            x = np.asarray(x, float)
            return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float('nan')

        return {
            'episode_return': self.ret,
            'episode_length': self.n,
            'success': infer_success(info, self.ret, success_threshold),
            'mpc_ms': nm(self.mpc_ms),
            'prediction_mse': nm(self.pred_err),
            'uncertainty': nm(self.unc),
            'residual_norm': nm(self.resnorm),
            'gate': nm(self.gates),
            'adaptive_gate': nm(self.adaptive_gates),
            'gate_z': nm(self.gate_z),
            'residual_ramp': nm(self.ramps),
            'effective_residual_norm': nm(self.effective_resnorm),
            'mpc_cache_hit_rate': nm(self.cache_hits),
            'safeguard_scale': nm(self.safeguard_scales),
            'safeguard_gain': nm(self.safeguard_gains),
            'action_d1': float(d1),
            'action_d2': float(d2),
        }


class CSVLogger:
    """Append-only CSV writer that writes its header eagerly.

    The header is written at construction rather than on the first ``write``.
    Previously a run that finished no episode left a genuinely empty 0-byte file,
    and every downstream ``pd.read_csv`` raised ``EmptyDataError`` -- which made
    any short run (or any run of a task whose episode is longer than the step
    budget) crash the aggregation step rather than report zero episodes.

    ``on_exists`` controls what happens when the target already has rows:
    ``rotate`` (default) moves the old file aside, ``append`` keeps the legacy
    behaviour, ``error`` refuses. Silently appending to a previous run's CSV when
    a ``--run-name`` is reused is how two different runs end up averaged together.
    """

    def __init__(self, path, fieldnames, on_exists='rotate'):
        self.path = path
        self.fieldnames = list(fieldnames)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            if on_exists == 'error':
                raise FileExistsError(
                    f'{path} already contains data. Use a fresh --run-name, or pass '
                    "on_exists='rotate'/'append'."
                )
            if on_exists == 'rotate':
                stamp = 0
                while True:
                    backup = f'{path}.bak{stamp}' if stamp else f'{path}.bak'
                    if not os.path.exists(backup):
                        break
                    stamp += 1
                os.replace(path, backup)

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def write(self, row):
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, '') for k in self.fieldnames}
            )
