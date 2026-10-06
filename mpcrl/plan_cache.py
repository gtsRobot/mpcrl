from __future__ import annotations
import numpy as np

class PlanCache:
    def __init__(self): self.clear()
    def clear(self): self._obs=None; self._plan=None
    def put(self,obs,plan): self._obs=np.asarray(obs,dtype=np.float32).copy(); self._plan=plan
    def get_or_plan(self,obs,planner):
        current=np.asarray(obs,dtype=np.float32)
        if self._plan is not None and self._obs is not None and np.array_equal(current,self._obs):
            plan=self._plan; self.clear(); return plan,True
        self.clear(); return planner.plan(obs),False
