from __future__ import annotations
from dataclasses import dataclass
import math


def _sigmoid(x: float) -> float:
    x = max(min(float(x), 30.0), -30.0)
    return float(1.0 / (1.0 + math.exp(-x)))


def adaptive_uncertainty_gate(uncertainty: float, threshold: float = 0.03, temperature: float = 0.02) -> float:
    """Legacy absolute-scale gate: higher uncertainty -> larger RL correction weight."""
    t = max(float(temperature), 1e-6)
    return _sigmoid((float(uncertainty) - float(threshold)) / t)


def residual_ramp(step: int, start_step: int, ramp_steps: int) -> float:
    if int(step) < int(start_step): return 0.0
    n=max(int(ramp_steps),1)
    return float(min(1.0,max(0.0,(int(step)-int(start_step)+1)/n)))


@dataclass
class GateOutput:
    value: float
    normalized_uncertainty: float
    running_mean: float
    running_std: float


class RunningUncertaintyGate:
    """Sigmoid gate on the EMA z-score of the world model's uncertainty signal.

    ``initial_std`` seeds the running variance before enough samples exist. Pass
    a non-positive value to derive it from the first observed uncertainty instead
    of a fixed absolute constant -- the uncertainty signal is in *return* units
    (see ``EnsembleWorldModel.rollout_return``), whose magnitude varies by orders
    of magnitude across tasks, so a hardcoded 0.05 makes the gate saturate for
    the first few hundred steps on some environments.
    """

    def __init__(self,momentum=0.995,initial_std=0.05,z_threshold=0.5,z_temperature=0.75,min_gate=0.05,max_gate=0.95,eps=1e-6,relative_initial_std=0.25):
        self.momentum=float(momentum); self.eps=float(eps)
        self.relative_initial_std=max(float(relative_initial_std),eps)
        self.adaptive_initial_std=float(initial_std)<=0.0
        self.initial_std=max(float(initial_std),eps) if not self.adaptive_initial_std else eps
        self.z_threshold=float(z_threshold); self.z_temperature=max(float(z_temperature),eps)
        self.min_gate=float(min_gate); self.max_gate=float(max_gate)
        self.count=0; self.mean=0.0; self.var=self.initial_std**2
    @property
    def std(self): return float(math.sqrt(max(self.var,self.eps**2)))
    def _update(self,value):
        value=float(value)
        if self.count==0:
            self.mean=value
            if self.adaptive_initial_std:
                self.initial_std=max(self.relative_initial_std*abs(value),self.eps)
            self.var=self.initial_std**2
        else:
            old=self.mean; m=self.momentum; self.mean=m*self.mean+(1-m)*value; d=value-old; self.var=m*self.var+(1-m)*d*d
        self.count+=1
    def __call__(self,uncertainty,update=True):
        u=float(uncertainty)
        if self.count==0: z=0.0; mean=u; std=self.initial_std
        else: mean=self.mean; std=self.std; z=(u-mean)/max(std,self.eps)
        raw=_sigmoid((z-self.z_threshold)/self.z_temperature); gate=min(self.max_gate,max(self.min_gate,raw))
        if update:self._update(u)
        return GateOutput(float(gate),float(z),float(mean),float(std))
    def state_dict(self): return {'count':int(self.count),'mean':float(self.mean),'var':float(self.var),'initial_std':float(self.initial_std)}
    def load_state_dict(self,state):
        self.count=int(state.get('count',0)); self.mean=float(state.get('mean',0.0))
        self.initial_std=max(float(state.get('initial_std',self.initial_std)),self.eps)
        self.var=max(float(state.get('var',self.initial_std**2)),self.eps**2)


def make_uncertainty_gate(residual_cfg):
    if str(residual_cfg.get('gate_mode','legacy')).lower() not in {'running_zscore','normalized','ema_zscore'}: return None
    return RunningUncertaintyGate(momentum=residual_cfg.get('gate_momentum',0.995),initial_std=residual_cfg.get('gate_initial_std',0.05),z_threshold=residual_cfg.get('gate_z_threshold',0.5),z_temperature=residual_cfg.get('gate_z_temperature',0.75),min_gate=residual_cfg.get('gate_min',0.05),max_gate=residual_cfg.get('gate_max',0.95),relative_initial_std=residual_cfg.get('gate_relative_initial_std',0.25))
