from __future__ import annotations
import argparse, copy
from mpcrl.config import load_yaml
from experiments.train import train

ABLATIONS={
 'full':{},
 'no_gate':{'residual':{'adaptive_gate':False,'fixed_gate':1.0}},
 'no_uncertainty_penalty':{'world_model':{'uncertainty_penalty':0.0}},
 'no_consistency':{'residual':{'consistency_coef':0.0}},
 'chunk1':{'residual':{'chunk_len':1}},
 'chunk5':{'residual':{'chunk_len':5}},
 'small_residual':{'residual':{'residual_scale':0.10}},
 'large_residual':{'residual':{'residual_scale':0.30}},
}

def merge(d,u):
    out=copy.deepcopy(d)
    for k,v in u.items(): out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/halfcheetah.yaml'); p.add_argument('--steps',type=int,default=30000); p.add_argument('--seed',type=int,default=0); a=p.parse_args(); base=load_yaml(a.config); base['env']['seed']=a.seed
    for name,ov in ABLATIONS.items():
        cfg=merge(base,ov); train(cfg,'planning_residual',a.steps,run_name=f"ablation__{cfg['env']['id']}__{name}__seed{a.seed}")
if __name__=='__main__': main()
