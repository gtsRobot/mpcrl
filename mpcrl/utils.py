from __future__ import annotations
import os,random,torch,numpy as np
from contextlib import nullcontext

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def pick_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def configure_accelerator(hardware=None):
    """Select and configure the compute device.

    Set ``hardware.require_cuda: true`` (or MPCRL_REQUIRE_CUDA=1) to make a
    missing GPU a hard error. Without it a server whose torch was installed
    CPU-only silently runs ~100x slower, which looks like a hang rather than a
    misconfiguration.
    """
    hardware=hardware or {}; device=pick_device()
    require=bool(hardware.get('require_cuda',False)) or os.environ.get('MPCRL_REQUIRE_CUDA')=='1'
    if device.type!='cuda':
        msg=('CUDA is not available. torch.cuda.is_available() is False -- check the NVIDIA driver '
             'and that torch was installed from the CUDA 12.8+ index '
             '(pip install torch --index-url https://download.pytorch.org/whl/cu128).')
        if require: raise RuntimeError(msg)
        print('WARNING: '+msg+' Falling back to CPU, which is roughly 100x slower.',flush=True)
        return device
    torch.backends.cuda.matmul.allow_tf32=bool(hardware.get('allow_tf32',True)); torch.backends.cudnn.allow_tf32=bool(hardware.get('allow_tf32',True)); torch.backends.cudnn.benchmark=bool(hardware.get('cudnn_benchmark',True)); torch.set_float32_matmul_precision(str(hardware.get('matmul_precision','high')))
    return device
def amp_dtype_from_name(name):
    name=(name or 'fp32').lower(); return torch.bfloat16 if name in {'bf16','bfloat16'} else (torch.float16 if name in {'fp16','float16','half'} else None)
def autocast_context(device,precision='fp32'):
    device=torch.device(device); dtype=amp_dtype_from_name(precision)
    return torch.autocast(device_type='cuda',dtype=dtype) if device.type=='cuda' and dtype is not None else nullcontext()
def accelerator_summary(device,precision):
    if device.type!='cuda': return 'device=cpu precision=fp32'
    p=torch.cuda.get_device_properties(device); return f'device={p.name} compute_capability={p.major}.{p.minor} vram={p.total_memory/1024**3:.1f}GiB precision={precision}'
def ensure_dir(path): os.makedirs(path,exist_ok=True); return path
