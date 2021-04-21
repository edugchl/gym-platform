import unittest

from gym_platform import UserEnv


class TestUserEnv(unittest.TestCase):
    def test_reward_direction(self, env):
        obs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
        actions = np.array([0, 1])
        directions = [False, False, False, True,
                    True, True, True, False]
        
        for action in actions:
            for ob in obs:
                reward = env._compute_reward(ob, action)
                pred_direction = reward > 0
                self.assertEqual(pred_direction, directions.pop())



if __name__ == '__main__':
    unittest.main()