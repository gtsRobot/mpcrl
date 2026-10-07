import unittest, numpy as np
from mpcrl.sac import ResidualSAC
class T(unittest.TestCase):
    def test_update(self):
        agent=ResidualSAC(7,3,hidden=32); n=16; b={'context':np.random.randn(n,7).astype('f'),'residual':np.random.uniform(-1,1,(n,3)).astype('f'),'reward':np.random.randn(n,1).astype('f'),'next_context':np.random.randn(n,7).astype('f'),'done':np.zeros((n,1),'f')}; out=agent.update(b); self.assertIn('actor_loss',out)
if __name__=='__main__': unittest.main()
