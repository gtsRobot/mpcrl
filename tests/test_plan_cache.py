import unittest
import numpy as np
from mpcrl.plan_cache import PlanCache


class DummyPlanner:
    def __init__(self): self.calls = 0
    def plan(self, obs):
        self.calls += 1
        return {'obs': np.asarray(obs).copy(), 'call': self.calls}


class T(unittest.TestCase):
    def test_cached_plan_is_consumed_without_replanning(self):
        cache = PlanCache(); planner = DummyPlanner(); obs = np.array([1.0, 2.0], np.float32)
        first, hit = cache.get_or_plan(obs, planner)
        self.assertFalse(hit); self.assertEqual(planner.calls, 1)
        next_obs = np.array([2.0, 3.0], np.float32)
        cached = planner.plan(next_obs)
        cache.put(next_obs, cached)
        got, hit = cache.get_or_plan(next_obs, planner)
        self.assertTrue(hit); self.assertEqual(planner.calls, 2); self.assertIs(got, cached)
        # Cache is one-shot: requesting the same state again must plan again.
        _, hit = cache.get_or_plan(next_obs, planner)
        self.assertFalse(hit); self.assertEqual(planner.calls, 3)


if __name__ == '__main__':
    unittest.main()
