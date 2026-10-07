import unittest, numpy as np, torch
from mpcrl.world_model import EnsembleWorldModel
class T(unittest.TestCase):
    def test_shapes(self):
        np.random.seed(0); n=128; o=np.random.randn(n,4).astype('f'); a=np.random.randn(n,2).astype('f'); no=o+0.1*np.pad(a,((0,0),(0,2))); r=-(a*a).sum(1,keepdims=True).astype('f'); b={'obs':o,'action':a,'next_obs':no,'reward':r}; m=EnsembleWorldModel(4,2,ensemble_size=2,hidden_dim=32); m.fit_batch(b,updates=2,batch_size=64); p=m.predict(o[:5],a[:5]); self.assertEqual(tuple(p.next_obs.shape),(5,4)); self.assertEqual(tuple(p.reward.shape),(5,1))
if __name__=='__main__': unittest.main()
