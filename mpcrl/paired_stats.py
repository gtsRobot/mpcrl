"""Paired-comparison statistics for the MPC vs MPC+residual protocol.

Deliberately dependency-free (no scipy): the repo's requirements.txt does not
list it, and everything needed here is either a small lookup table or an exact
enumeration over at most a few hundred sign patterns.
"""
from __future__ import annotations

import itertools
import math

# Two-sided 97.5% Student-t quantiles by degrees of freedom. Using the normal
# 1.96 instead (as summarize_eval.py used to) understates the interval by 42% at
# n=5, which is exactly the regime this project runs in.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 50: 2.009,
    60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}


def t_quantile_975(df: int) -> float:
    """Two-sided 95% (i.e. 97.5% one-tail) t quantile, interpolating the table."""
    if df < 1:
        return float('nan')
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    if df > keys[-1]:
        return 1.960
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    w = (df - lo) / (hi - lo)
    return _T975[lo] * (1 - w) + _T975[hi] * w


def wilcoxon_signed_rank_exact(deltas):
    """Exact two-sided signed-rank p-value. Returns (p, n_used, p_floor).

    ``p_floor`` is the smallest p this n could possibly produce: 2/2**n. At n=5
    that is 0.0625, so a 5-seed study CANNOT reach p<0.05 no matter how clean the
    effect is. Report it so nobody reads a null result as evidence of no effect.
    """
    vals = [float(d) for d in deltas if d == d and d != 0.0]
    n = len(vals)
    if n == 0:
        return float('nan'), 0, float('nan')
    p_floor = 2.0 / (2 ** n)
    if n > 20:  # normal approximation, enumeration gets expensive
        ranks = _ranks([abs(v) for v in vals])
        w = sum(r for v, r in zip(vals, ranks) if v > 0)
        mean = n * (n + 1) / 4.0
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mean) / sd if sd > 0 else 0.0
        p = math.erfc(abs(z) / math.sqrt(2.0))
        return min(1.0, p), n, p_floor

    ranks = _ranks([abs(v) for v in vals])
    observed = sum(r for v, r in zip(vals, ranks) if v > 0)
    total = n * (n + 1) / 2.0
    stat = max(observed, total - observed)
    count = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum(r for s, r in zip(signs, ranks) if s)
        if max(w, total - w) >= stat - 1e-12:
            count += 1
    return min(1.0, count / (2 ** n)), n, p_floor


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def paired_summary(deltas):
    """Full paired report for a list of per-seed differences (arm B - arm A)."""
    vals = [float(d) for d in deltas if d == d]
    n = len(vals)
    out = {
        'n_seeds': n,
        'mean_delta': float('nan'),
        'sd_delta': float('nan'),
        'sem_delta': float('nan'),
        'ci95_low': float('nan'),
        'ci95_high': float('nan'),
        'ci95_half_width': float('nan'),
        'n_positive': sum(1 for v in vals if v > 0),
        'n_negative': sum(1 for v in vals if v < 0),
        'wilcoxon_p': float('nan'),
        'wilcoxon_p_floor': float('nan'),
        'significant_at_95': False,
    }
    if n == 0:
        return out
    mean = sum(vals) / n
    out['mean_delta'] = mean
    if n < 2:
        out['sd_delta'] = 0.0
        out['sem_delta'] = 0.0
        return out
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    half = t_quantile_975(n - 1) * sem
    p, _, floor = wilcoxon_signed_rank_exact(vals)
    out.update({
        'sd_delta': sd, 'sem_delta': sem,
        'ci95_low': mean - half, 'ci95_high': mean + half, 'ci95_half_width': half,
        'wilcoxon_p': p, 'wilcoxon_p_floor': floor,
        'significant_at_95': bool(mean - half > 0 or mean + half < 0),
    })
    return out


def minimum_detectable_effect(sd_delta, n_seeds):
    """Smallest true paired effect a study of this size can resolve at 95%.

    Publish this BEFORE launching a sweep: if the MDE exceeds the effect you hope
    to show, the experiment is not worth the GPU hours at that n.
    """
    if n_seeds < 2 or not (sd_delta == sd_delta):
        return float('nan')
    return t_quantile_975(n_seeds - 1) * sd_delta / math.sqrt(n_seeds)
