from gym.envs.registration import register, make
from gym_platform.envs.learning_platform import LearningPlatform
from gym_platform.envs.user import User
from gym_platform.envs.clock import Clock


register(
    id='platform-v0',
    entry_point='gym_platform.envs:LearningPlatform',
)
