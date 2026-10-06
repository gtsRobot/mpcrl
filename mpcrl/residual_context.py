from __future__ import annotations

import numpy as np


def residual_context(obs, chunk, effective_gate: float) -> np.ndarray:
    """Build SAC context for a gated residual controller.

    The effective gate must be observable by the critic because it changes how
    a raw residual proposal is mapped to the environment action.
    """
    return np.concatenate([
        np.asarray(obs, dtype=np.float32).reshape(-1),
        np.asarray(chunk, dtype=np.float32).reshape(-1),
        np.asarray([effective_gate], dtype=np.float32),
    ])


def gated_strength(adaptive_gate: float, ramp: float, power: float = 1.0) -> float:
    """Combine uncertainty gate and training ramp with optional conservatism."""
    a = float(np.clip(adaptive_gate, 0.0, 1.0))
    r = float(np.clip(ramp, 0.0, 1.0))
    p = max(float(power), 1e-6)
    return float((a ** p) * r)
