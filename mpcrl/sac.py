from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(inp,out,hidden):
    return nn.Sequential(nn.Linear(inp,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out))


class GaussianActor(nn.Module):
    def __init__(self, context_dim, action_dim, hidden=256):
        super().__init__(); self.net=mlp(context_dim, action_dim*2, hidden); self.action_dim=action_dim
    def forward(self, x):
        mean, log_std = self.net(x).chunk(2,-1); log_std=log_std.clamp(LOG_STD_MIN,LOG_STD_MAX)
        return mean, log_std
    def sample(self,x):
        mean, log_std=self(x); std=log_std.exp(); dist=torch.distributions.Normal(mean,std); z=dist.rsample(); a=torch.tanh(z)
        logp=dist.log_prob(z)-torch.log(1-a.pow(2)+1e-6); return a,logp.sum(-1,keepdim=True),torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, context_dim, action_dim, hidden=256): super().__init__(); self.net=mlp(context_dim+action_dim,1,hidden)
    def forward(self,c,a): return self.net(torch.cat([c,a],-1))


class DuelingCritic(nn.Module):
    """Q(c, a) = V(c) + A(c, a), with A(c, 0) subtracted out.

    A residual controller only perturbs the base action by a few percent, so the
    residual-dependent part of Q is a tiny fraction of Q itself (on HalfCheetah
    Q ~ 250 while the residual's contribution is O(1)). Regressing the whole Q
    with a single head buries that signal in the value baseline's approximation
    error. Splitting the heads lets the advantage head specialise, and anchoring
    it at ``a = 0`` (the "execute the plan unchanged" action) makes ``A`` directly
    readable as "is intervening here worth it", which the gate and the shield both
    want. The subtraction is what makes the decomposition identified -- without it
    V and A can drift by an arbitrary function of c.
    """

    def __init__(self, context_dim, action_dim, hidden=256):
        super().__init__()
        self.v = mlp(context_dim, 1, hidden)
        self.a = mlp(context_dim + action_dim, 1, hidden)
        self.action_dim = action_dim

    def value(self, c):
        return self.v(c)

    def advantage(self, c, a):
        zero = torch.zeros(c.shape[0], self.action_dim, device=c.device, dtype=c.dtype)
        return self.a(torch.cat([c, a], -1)) - self.a(torch.cat([c, zero], -1))

    def forward(self, c, a):
        return self.v(c) + self.advantage(c, a)


class StateValue(nn.Module):
    """V(obs) trained on the residual arm's own transitions.

    Used as a terminal value for the H-step world-model rollout. With gamma=0.99
    and H=20, gamma**H = 0.818, i.e. >80% of the discounted return lies beyond the
    planner's horizon and is structurally invisible to CEM. Supplying V(s_H) is
    what lets the residual optimise something the planner did not.
    """

    def __init__(self, obs_dim, hidden=256):
        super().__init__(); self.net = mlp(obs_dim, 1, hidden)

    def forward(self, obs): return self.net(obs)


class ResidualSAC:
    def __init__(self, context_dim, residual_dim, hidden=256, lr=3e-4, gamma=0.99, tau=0.005,
                 init_alpha=0.2, target_entropy_scale=1.0, consistency_coef=0.0, device='cpu', precision='fp32',
                 critic='dueling', obs_dim=None, actor_delay=2, target_policy_noise=0.0,
                 trust_region_relative=True):
        self.device=torch.device(device); self.gamma=gamma; self.tau=tau; self.consistency_coef=consistency_coef
        self.precision=precision; self.amp_dtype=torch.bfloat16 if precision == 'bf16' else (torch.float16 if precision == 'fp16' else None)
        self.critic_kind=critic; self.actor_delay=max(1,int(actor_delay))
        self.target_policy_noise=float(target_policy_noise)
        self.trust_region_relative=bool(trust_region_relative)
        self.obs_dim=int(obs_dim) if obs_dim is not None else None
        self._updates=0
        self._last_logs={}

        make=(lambda: DuelingCritic(context_dim,residual_dim,hidden)) if critic=='dueling' else (lambda: Critic(context_dim,residual_dim,hidden))
        self.actor=GaussianActor(context_dim,residual_dim,hidden).to(self.device)
        self.q1=make().to(self.device); self.q2=make().to(self.device)
        self.tq1=make().to(self.device); self.tq2=make().to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict()); self.tq2.load_state_dict(self.q2.state_dict())
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.q_opt=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.log_alpha=torch.tensor(np.log(init_alpha),device=self.device,requires_grad=True)
        self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=lr)
        self.target_entropy=-float(residual_dim)*target_entropy_scale

        # Optional state-only terminal value, used to give the shield access to
        # the beyond-horizon return. Needs obs_dim so it can slice obs out of the
        # context (residual_context puts obs first).
        self.state_value=None
        if self.obs_dim is not None:
            self.state_value=StateValue(self.obs_dim,hidden).to(self.device)
            self.tstate_value=StateValue(self.obs_dim,hidden).to(self.device)
            self.tstate_value.load_state_dict(self.state_value.state_dict())
            self.v_opt=torch.optim.Adam(self.state_value.parameters(),lr=lr)

    @property
    def alpha(self): return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, context, deterministic=False):
        c=torch.as_tensor(context,dtype=torch.float32,device=self.device).reshape(1,-1)
        a,_,mean=self.actor.sample(c); return (mean if deterministic else a).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def advantage(self, context, residual):
        """Conservative (min over the twin critics) advantage of a residual."""
        if self.critic_kind!='dueling': raise RuntimeError('advantage() requires critic="dueling"')
        c=torch.as_tensor(context,dtype=torch.float32,device=self.device).reshape(1,-1)
        a=torch.as_tensor(residual,dtype=torch.float32,device=self.device).reshape(1,-1)
        return float(torch.min(self.q1.advantage(c,a),self.q2.advantage(c,a)).squeeze())

    @torch.no_grad()
    def advantage_batch(self, contexts, residuals):
        if self.critic_kind!='dueling': raise RuntimeError('advantage_batch() requires critic="dueling"')
        c=torch.as_tensor(contexts,dtype=torch.float32,device=self.device)
        a=torch.as_tensor(residuals,dtype=torch.float32,device=self.device)
        return torch.min(self.q1.advantage(c,a),self.q2.advantage(c,a)).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def terminal_value(self, obs):
        """Callable for EnsembleWorldModel.rollout_return(terminal_value_fn=...)."""
        if self.state_value is None: raise RuntimeError('terminal_value requires obs_dim to be set')
        o=obs if torch.is_tensor(obs) else torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        return self.state_value(o.to(self.device).float()).squeeze(-1)

    def update(self,batch):
        b={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in batch.items()}
        c,a,r,nc,d=b['context'],b['residual'],b['reward'],b['next_context'],b['done']
        discount=b['discount'] if 'discount' in b else None
        g=discount if discount is not None else self.gamma
        amp_enabled=self.device.type == 'cuda' and self.amp_dtype is not None
        self._updates+=1

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp_enabled):
            na,nlogp,nmean=self.actor.sample(nc)
            if self.target_policy_noise>0:
                # TD3 target-policy smoothing: stops the actor from exploiting a
                # sharp critic maximum that is an artefact of narrow action support.
                noise=(torch.randn_like(nmean)*self.target_policy_noise).clamp(-0.5,0.5)
                na=(nmean+noise).clamp(-1.0,1.0)
                nlogp=torch.zeros_like(nlogp)
            tq=torch.min(self.tq1(nc,na),self.tq2(nc,na))-self.alpha.detach()*nlogp
            target=r+g*(1-d)*tq
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp_enabled):
            q1,q2=self.q1(c,a),self.q2(c,a)
            qloss=F.mse_loss(q1.float(),target.float())+F.mse_loss(q2.float(),target.float())
        self.q_opt.zero_grad(set_to_none=True); qloss.backward(); self.q_opt.step()

        # State-only value head, trained on the same transitions.
        vloss=float('nan')
        if self.state_value is not None:
            od=self.obs_dim
            obs,next_obs=c[:,:od],nc[:,:od]
            with torch.no_grad():
                v_target=r+g*(1-d)*self.tstate_value(next_obs)
            vpred=self.state_value(obs)
            vl=F.mse_loss(vpred.float(),v_target.float())
            self.v_opt.zero_grad(set_to_none=True); vl.backward(); self.v_opt.step()
            vloss=float(vl.detach())

        actor_loss=self._last_logs.get('actor_loss',float('nan'))
        consistency=self._last_logs.get('consistency',float('nan'))
        if self._updates % self.actor_delay == 0:
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp_enabled):
                pa,logp,_=self.actor.sample(c); q=torch.min(self.q1(c,pa),self.q2(c,pa))
                pen=pa.float().pow(2).mean()
                # Anchor the trust-region penalty to the scale of the objective it
                # competes with. At a fixed 0.05 it was ~5000x too small to
                # constrain anything on HalfCheetah (|Q| ~ 250), which is why the
                # no_consistency ablation never showed a difference.
                scale=q.float().abs().mean().detach().clamp_min(1.0) if self.trust_region_relative else 1.0
                al=(self.alpha.detach()*logp.float()-q.float()).mean()+self.consistency_coef*scale*pen
            self.actor_opt.zero_grad(set_to_none=True); al.backward(); self.actor_opt.step()
            alpha_loss=-(self.log_alpha*(logp.detach()+self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_opt.step()
            actor_loss=float(al.detach()); consistency=float(pen.detach())
            with torch.no_grad():
                for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.lerp_(p.data,self.tau)
                for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.lerp_(p.data,self.tau)
                if self.state_value is not None:
                    for p,tp in zip(self.state_value.parameters(),self.tstate_value.parameters()): tp.data.lerp_(p.data,self.tau)

        self._last_logs={'q_loss':float(qloss.detach()),'actor_loss':actor_loss,
                         'alpha':float(self.alpha.detach()),'consistency':consistency,'v_loss':vloss}
        return dict(self._last_logs)

    def diagnostics(self,batch):
        """std(A)/std(Q) on a batch -- the falsification gate for the advantage critic."""
        if self.critic_kind!='dueling': return {}
        with torch.no_grad():
            c=torch.as_tensor(batch['context'],dtype=torch.float32,device=self.device)
            a=torch.as_tensor(batch['residual'],dtype=torch.float32,device=self.device)
            adv=torch.min(self.q1.advantage(c,a),self.q2.advantage(c,a))
            q=torch.min(self.q1(c,a),self.q2(c,a))
            sa=float(adv.std()); sq=float(q.std())
            return {'adv_std':sa,'q_std':sq,'adv_q_std_ratio':sa/max(sq,1e-12),'adv_mean':float(adv.mean())}

    def state_dict(self):
        out={'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict(),
             'tq1':self.tq1.state_dict(),'tq2':self.tq2.state_dict(),'log_alpha':self.log_alpha.detach().cpu(),
             'critic_kind':self.critic_kind}
        if self.state_value is not None:
            out['state_value']=self.state_value.state_dict(); out['tstate_value']=self.tstate_value.state_dict()
        return out

    def load_bundle(self, bundle):
        saved=bundle.get('critic_kind',self.critic_kind)
        if saved!=self.critic_kind:
            raise RuntimeError(f'checkpoint critic={saved!r} but this agent was built with critic={self.critic_kind!r}')
        self.actor.load_state_dict(bundle['actor']); self.q1.load_state_dict(bundle['q1']); self.q2.load_state_dict(bundle['q2'])
        self.tq1.load_state_dict(bundle['tq1']); self.tq2.load_state_dict(bundle['tq2'])
        with torch.no_grad(): self.log_alpha.copy_(bundle['log_alpha'].to(self.device))
        if self.state_value is not None and 'state_value' in bundle:
            self.state_value.load_state_dict(bundle['state_value']); self.tstate_value.load_state_dict(bundle['tstate_value'])
