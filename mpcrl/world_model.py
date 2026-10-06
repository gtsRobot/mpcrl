from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class MLPDynamics(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_dim=256): super().__init__(); self.net=nn.Sequential(nn.Linear(obs_dim+action_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,obs_dim+1))
    def forward(self,obs,action): return self.net(torch.cat([obs,action],dim=-1))


class EnsembleMLPDynamics(nn.Module):
    """All ensemble members in one module, evaluated with batched matmuls.

    Mathematically identical to a list of ``MLPDynamics`` (same per-member init
    distribution), but a forward pass over M members costs 3 ``baddbmm`` calls
    instead of 3*M ``addmm`` calls. CEM planning issues
    ``iterations * horizon * M`` forward passes per plan, which made the previous
    Python-level member loop kernel-launch bound rather than FLOP bound.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, ensemble_size):
        super().__init__()
        self.m = int(ensemble_size)
        din, dout = obs_dim + action_dim, obs_dim + 1
        self.w1, self.b1 = self._layer(self.m, din, hidden_dim)
        self.w2, self.b2 = self._layer(self.m, hidden_dim, hidden_dim)
        self.w3, self.b3 = self._layer(self.m, hidden_dim, dout)

    @staticmethod
    def _layer(m, fan_in, fan_out):
        # nn.Linear's default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for both
        # weight and bias. Reproduced per member so ensembles behave as before.
        bound = 1.0 / np.sqrt(fan_in)
        w = nn.Parameter(torch.empty(m, fan_in, fan_out).uniform_(-bound, bound))
        b = nn.Parameter(torch.empty(m, 1, fan_out).uniform_(-bound, bound))
        return w, b

    def forward(self, x):
        """x: [M, B, obs+act] -> [M, B, obs+1]."""
        h = torch.baddbmm(self.b1, x, self.w1)
        h = F.silu(h)
        h = torch.baddbmm(self.b2, h, self.w2)
        h = F.silu(h)
        return torch.baddbmm(self.b3, h, self.w3)


@dataclass
class WMPrediction: next_obs: torch.Tensor; reward: torch.Tensor; uncertainty: torch.Tensor


class EnsembleWorldModel(nn.Module):
    def __init__(self,obs_dim,action_dim,ensemble_size=5,hidden_dim=256,lr=1e-3,weight_decay=1e-5,device='cpu',precision='fp32',
                 stats_momentum=0.99,bootstrap_prob=0.8,grad_clip=10.0):
        super().__init__()
        self.obs_dim=obs_dim; self.action_dim=action_dim; self.ensemble_size=int(ensemble_size)
        self.device=torch.device(device)
        self.amp_dtype=torch.bfloat16 if precision=='bf16' else (torch.float16 if precision=='fp16' else None)
        self.stats_momentum=float(stats_momentum); self.bootstrap_prob=float(bootstrap_prob); self.grad_clip=float(grad_clip)
        self.members=EnsembleMLPDynamics(obs_dim,action_dim,hidden_dim,self.ensemble_size)
        self.optimizer=torch.optim.AdamW(self.members.parameters(),lr=lr,weight_decay=weight_decay)
        self.register_buffer('obs_mean',torch.zeros(obs_dim)); self.register_buffer('obs_std',torch.ones(obs_dim))
        self.register_buffer('act_mean',torch.zeros(action_dim)); self.register_buffer('act_std',torch.ones(action_dim))
        self.register_buffer('delta_mean',torch.zeros(obs_dim)); self.register_buffer('delta_std',torch.ones(obs_dim))
        self.register_buffer('rew_mean',torch.zeros(1)); self.register_buffer('rew_std',torch.ones(1))
        self.register_buffer('stats_initialized',torch.zeros((),dtype=torch.bool))
        self.frozen=False
        self.to(self.device)

    # ---------------------------------------------------------------- statistics
    @torch.no_grad()
    def update_stats(self,batch):
        """Blend batch statistics into the running normalizers.

        Previously these buffers were *overwritten* from a freshly resampled 10k
        subset on every refit, so the input scaling shifted discontinuously under
        already-trained weights. An EMA keeps them stable; the first call
        initializes directly so early training is not biased toward zeros/ones.
        """
        def st(x):
            t=torch.as_tensor(x,dtype=torch.float32,device=self.device)
            return t.mean(0),t.std(0).clamp_min(1e-2)
        o,a,no,r=batch['obs'],batch['action'],batch['next_obs'],batch['reward']
        pairs=((self.obs_mean,self.obs_std,st(o)),(self.act_mean,self.act_std,st(a)),
               (self.delta_mean,self.delta_std,st(no-o)),(self.rew_mean,self.rew_std,st(r)))
        if not bool(self.stats_initialized):
            for mean_buf,std_buf,(m,s) in pairs: mean_buf.copy_(m); std_buf.copy_(s)
            self.stats_initialized.fill_(True)
        else:
            k=self.stats_momentum
            for mean_buf,std_buf,(m,s) in pairs:
                mean_buf.mul_(k).add_(m,alpha=1-k); std_buf.mul_(k).add_(s,alpha=1-k).clamp_min_(1e-2)

    def _norm_inputs(self,obs,action): return (obs-self.obs_mean)/self.obs_std,(action-self.act_mean)/self.act_std
    def _decode(self,obs,y): delta=y[...,:self.obs_dim]*self.delta_std+self.delta_mean; reward=y[...,self.obs_dim:]*self.rew_std+self.rew_mean; return obs+delta,reward

    # ------------------------------------------------------------------ training
    def freeze(self):
        """Stop all further learning. Used by the paired protocol."""
        self.frozen=True
        for p in self.members.parameters(): p.requires_grad_(False)
        self.eval()
        return self

    def fit_batch(self,batch,updates=50,batch_size=256):
        if self.frozen:
            raise RuntimeError('fit_batch called on a frozen world model; the paired protocol requires it to stay fixed.')
        self.update_stats(batch)
        obs=torch.as_tensor(batch['obs'],dtype=torch.float32,device=self.device)
        action=torch.as_tensor(batch['action'],dtype=torch.float32,device=self.device)
        next_obs=torch.as_tensor(batch['next_obs'],dtype=torch.float32,device=self.device)
        reward=torch.as_tensor(batch['reward'],dtype=torch.float32,device=self.device)
        n=len(obs); m=self.ensemble_size; bs=min(batch_size,n); losses=[]; skipped=0
        for _ in range(updates):
            # Independent index draws per member give each member a different
            # bootstrap sample of the buffer, which is what makes cross-member
            # disagreement an epistemic signal rather than just init noise.
            idx=torch.randint(0,n,(m,bs),device=self.device)
            o=obs[idx]; a=action[idx]; no=next_obs[idx]; r=reward[idx]
            if self.bootstrap_prob<1.0:
                keep=(torch.rand(m,bs,1,device=self.device)<self.bootstrap_prob).float()
            else:
                keep=None
            on,an=self._norm_inputs(o,a)
            target=torch.cat([((no-o)-self.delta_mean)/self.delta_std,(r-self.rew_mean)/self.rew_std],-1)
            pred=self.members(torch.cat([on,an],dim=-1))
            se=(pred.float()-target.float()).pow(2)
            loss=(se*keep).sum()/keep.sum().clamp_min(1.0)/target.shape[-1] if keep is not None else se.mean()
            if not torch.isfinite(loss):
                # A diverged member must not be allowed to write inf/nan into the
                # weights: that turns every CEM score into NaN downstream.
                self.optimizer.zero_grad(set_to_none=True); skipped+=1; continue
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            if self.grad_clip>0: torch.nn.utils.clip_grad_norm_(self.members.parameters(),self.grad_clip)
            self.optimizer.step(); losses.append(float(loss.detach()))
        self.last_skipped_updates=skipped
        return float(np.mean(losses)) if losses else float('nan')

    # ---------------------------------------------------------------- prediction
    def _members_forward(self,obs_m,act_m):
        """obs_m/act_m: [M,B,*] -> decoded (next_obs [M,B,obs], reward [M,B,1])."""
        on,an=self._norm_inputs(obs_m,act_m)
        y=self.members(torch.cat([on,an],dim=-1)).float()
        return self._decode(obs_m,y)

    def predict(self,obs,action):
        if not torch.is_tensor(obs): obs=torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        if not torch.is_tensor(action): action=torch.as_tensor(action,dtype=torch.float32,device=self.device)
        obs=obs.to(self.device).float(); action=action.to(self.device).float()
        m=self.ensemble_size
        obs_m=obs.unsqueeze(0).expand(m,*obs.shape); act_m=action.unsqueeze(0).expand(m,*action.shape)
        nexts,rews=self._members_forward(obs_m,act_m)
        return WMPrediction(nexts.mean(0),rews.mean(0),
                            nexts.var(0,unbiased=False).mean(-1,keepdim=True)+rews.var(0,unbiased=False))

    @torch.no_grad()
    def rollout_return(self,start_obs,action_sequences,discount=.99,uncertainty_penalty=0.,
                       terminal_value_fn=None):
        """Score action sequences by rolling every ensemble member separately.

        Each member propagates its *own* state, so trajectories diverge and the
        returned uncertainty is the cross-member spread of whole-horizon returns
        rather than a horizon-average of one-step disagreements. The previous
        implementation advanced the ensemble *mean* state, which made member
        trajectories identical by construction and the uncertainty signal
        effectively constant.

        The score is the MOPO-style lower confidence bound
        ``mean_members(R) - uncertainty_penalty * std_members(R)``. Note this
        changes what ``uncertainty_penalty`` means: it is now a coefficient on
        *return* standard deviations, not on a horizon-averaged one-step variance.

        ``terminal_value_fn(obs)`` optionally adds ``discount**horizon * V(s_H)``,
        giving the score access to the ``gamma**H`` (>80% at H=20, gamma=0.99) of
        discounted value that a finite-horizon rollout structurally ignores.
        """
        acts=torch.as_tensor(action_sequences,dtype=torch.float32,device=self.device)
        n,h,_=acts.shape; m=self.ensemble_size
        obs=torch.as_tensor(start_obs,dtype=torch.float32,device=self.device).reshape(1,-1).expand(n,-1)
        obs=obs.unsqueeze(0).expand(m,n,-1).contiguous()
        ret=torch.zeros((m,n),device=self.device); g=1.0
        amp = self.device.type=='cuda' and self.amp_dtype is not None
        with torch.autocast(device_type='cuda',dtype=self.amp_dtype,enabled=amp):
            for t in range(h):
                a_t=acts[:,t].unsqueeze(0).expand(m,n,-1)
                nexts,rews=self._members_forward(obs,a_t)
                ret=ret+g*rews.squeeze(-1)
                obs=nexts
                g*=discount
            if terminal_value_fn is not None:
                ret=ret+g*terminal_value_fn(obs.reshape(m*n,-1)).reshape(m,n).float()
        ret=ret.float()
        spread=ret.std(0,unbiased=False) if m>1 else torch.zeros(n,device=self.device)
        return ret.mean(0)-uncertainty_penalty*spread, spread
