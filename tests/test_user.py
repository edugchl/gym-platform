import unittest

import numpy as np

from gym_platform import UserEnv, WorldEnv


class TestUserEnv(unittest.TestCase):
    def test_reward_direction(self):
        world_env = WorldEnv(ratio=3600)
        user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
        obs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
        actions = np.array([0, 1])
        directions = [True, True, False, True,
                    False, False, True, False]
        
        for action in actions:
            for ob in obs:
                reward = user_env._compute_reward(ob, action)
                pred_direction = (reward > 0)
                ans = directions.pop(0)
                print(action, ob, pred_direction, ans)
                self.assertEqual(pred_direction, ans)
                

if __name__ == '__main__':
    unittest.main()