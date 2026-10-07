"""GPU preflight. Run this first on any new machine.

All output is ASCII on purpose: this is the first thing executed on a fresh
server, and a non-UTF-8 locale with redirected output would otherwise turn a
diagnostic into a UnicodeEncodeError.
"""
import sys
from pathlib import Path

# Run directly as `python scripts/check_rtx5090.py`, so the repo root is not on
# sys.path yet and `import mpcrl` would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        'ERROR: CUDA is not available. Check the NVIDIA driver, and that torch was '
        'installed from the CUDA 12.8+ index:\n'
        '  pip install torch --index-url https://download.pytorch.org/whl/cu128'
    )

name = torch.cuda.get_device_name(0)
prop = torch.cuda.get_device_properties(0)
cc = (prop.major, prop.minor)
print('GPU:', name)
print('Compute capability:', f'{cc[0]}.{cc[1]}')
print('VRAM GiB:', round(prop.total_memory / 1024 ** 3, 2))
print('BF16 supported:', torch.cuda.is_bf16_supported())
print('GPU count:', torch.cuda.device_count())

if '5090' not in name:
    print('WARNING: this GPU is not an RTX 5090. The code still runs on any CUDA GPU; '
          'use configs/local/*.yaml for smaller cards.')
if cc < (12, 0):
    print('WARNING: this GPU is not Blackwell SM 12.x.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('ERROR: this CUDA/PyTorch build has no BF16 support.')

x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print('BF16 CUDA smoke test:', tuple(y.shape), y.dtype)

# Headless rendering is what the LeWM/pixel entry points need; report it here so
# a render failure is not discovered an hour into a job.
try:
    from mpcrl.envs import ensure_offscreen_gl
    print('MUJOCO_GL:', ensure_offscreen_gl() or '<left to gymnasium>')
except Exception as exc:  # pragma: no cover - diagnostic only
    print('WARNING: could not probe MUJOCO_GL:', exc)

print('environment check: PASS')
