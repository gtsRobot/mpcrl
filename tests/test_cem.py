import unittest, numpy as np, torch
from mpcrl.cem import CEMPlanner
class Toy:
    device=torch.device('cpu')
    def rollout_return(self,start,seq,discount=.99,uncertainty_penalty=0):
        s=torch.as_tensor(seq); score=-(s*s).sum((1,2)); return score,torch.zeros_like(score)
class T(unittest.TestCase):
    def test_cem_zero(self):
        np.random.seed(0); torch.manual_seed(0); p=CEMPlanner(Toy(),[-1], [1],horizon=5,candidates=256,elites=32,iterations=5,init_std=1.0); r=p.plan(np.array([0.])); self.assertLess(abs(r.actions[0,0]),0.25)
if __name__=='__main__': unittest.main()
