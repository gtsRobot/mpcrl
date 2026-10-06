from __future__ import annotations
import os
import sys
import numpy as np


# MuJoCo offscreen rendering needs an OpenGL context. Gymnasium tries
# glfw -> egl -> osmesa and swallows every failure, so on a headless Linux server
# the glfw attempt fails first and can land on software rendering (~100x slower)
# rather than erroring out. Default to EGL there, which is what an NVIDIA server
# actually has. On Windows/macOS glfw is the working backend, so leave it alone.
_DEFAULT_HEADLESS_GL = 'egl'


def ensure_offscreen_gl(backend: str | None = None) -> str | None:
    """Set MUJOCO_GL for offscreen rendering on headless Linux if unset.

    Returns the backend in effect, or None if the choice was left to gymnasium.
    """
    existing = os.environ.get('MUJOCO_GL')
    if existing:
        return existing
    override = backend or os.environ.get('MPCRL_MUJOCO_GL')
    if override:
        os.environ['MUJOCO_GL'] = override
        return override
    if not sys.platform.startswith('linux'):
        return None
    # A Linux box with no DISPLAY cannot create a glfw window.
    if os.environ.get('DISPLAY'):
        return None
    os.environ['MUJOCO_GL'] = _DEFAULT_HEADLESS_GL
    return _DEFAULT_HEADLESS_GL


def _contiguous_rgb_frame(frame):
    """Return an RGB frame with positive C-contiguous strides.

    Some MuJoCo/Gymnasium render paths return vertically flipped NumPy views
    with a negative stride. PyTorch cannot convert those views with
    ``torch.as_tensor``. Normal contiguous frames are returned without a copy;
    negative/non-contiguous views are copied once at the environment boundary.
    """
    if frame is None:
        return None
    arr = np.asarray(frame)
    return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)


def make_mujoco_env(env_id: str, seed: int, render_mode=None):
    if render_mode == 'rgb_array':
        # Must happen before the first env is constructed: gymnasium reads
        # MUJOCO_GL when it creates the offscreen viewer.
        ensure_offscreen_gl()

    try:
        import gymnasium as gym
    except ImportError as e:
        raise RuntimeError(
            'gymnasium is missing. Install it with: pip install "gymnasium[mujoco]" mujoco'
        ) from e

    env = gym.make(env_id, render_mode=render_mode)
    if render_mode == 'rgb_array':
        class _ContiguousRenderWrapper(gym.Wrapper):
            def render(self):
                return _contiguous_rgb_frame(self.env.render())

        env = _ContiguousRenderWrapper(env)

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, np.asarray(obs, dtype=np.float32), info


def dims(env):
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError(
            'this training entry point requires 1-D continuous observation/action spaces'
        )
    return int(env.observation_space.shape[0]), int(env.action_space.shape[0])


def action_bounds(env):
    return env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32)
