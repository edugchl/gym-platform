from gym.envs.registration import register, make
from gym_platform.envs.platform_env import PlatformEnv
from gym_platform.envs.user import UserEnv
from gym_platform.envs.world import WorldEnv


register(
    id='platform-v0',
    entry_point='gym_platform.envs:PlatformEnv',
)
