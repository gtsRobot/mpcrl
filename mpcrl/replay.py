from __future__ import annotations
import numpy as np


class TransitionReplay:
    def __init__(self,capacity,obs_dim,action_dim):
        self.capacity,self.size,self.ptr=capacity,0,0; self.obs=np.zeros((capacity,obs_dim),np.float32); self.action=np.zeros((capacity,action_dim),np.float32); self.reward=np.zeros((capacity,1),np.float32); self.next_obs=np.zeros((capacity,obs_dim),np.float32); self.done=np.zeros((capacity,1),np.float32)
    def add(self,obs,action,reward,next_obs,done):
        i=self.ptr; self.obs[i],self.action[i]=obs,action; self.reward[i],self.next_obs[i],self.done[i]=reward,next_obs,done; self.ptr=(self.ptr+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def sample(self,batch_size):
        idx=np.random.randint(0,self.size,size=batch_size); return {k:getattr(self,k)[idx] for k in ('obs','action','reward','next_obs','done')}


class ResidualReplay:
    """Replay for the residual controller.

    ``discount`` is stored per transition so a single residual decision that was
    held for k environment steps can be replayed as one k-step transition
    (``reward`` = discounted k-step reward sum, ``discount`` = gamma**k). Storing
    it per row keeps variable-length holds (an episode ending mid-hold) correct.
    """

    def __init__(self,capacity,context_dim,residual_dim,default_discount=0.99):
        self.capacity,self.size,self.ptr=capacity,0,0
        self.context=np.zeros((capacity,context_dim),np.float32)
        self.residual=np.zeros((capacity,residual_dim),np.float32)
        self.reward=np.zeros((capacity,1),np.float32)
        self.next_context=np.zeros((capacity,context_dim),np.float32)
        self.done=np.zeros((capacity,1),np.float32)
        self.discount=np.full((capacity,1),float(default_discount),np.float32)
    def add(self,context,residual,reward,next_context,done,discount=None):
        i=self.ptr; self.context[i],self.residual[i]=context,residual
        self.reward[i],self.next_context[i],self.done[i]=reward,next_context,done
        if discount is not None: self.discount[i]=discount
        self.ptr=(self.ptr+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def sample(self,batch_size):
        idx=np.random.randint(0,self.size,size=batch_size); return {k:getattr(self,k)[idx] for k in ('context','residual','reward','next_context','done','discount')}
