from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

from roverswarm.config import WorldConfig
from roverswarm.envs import MultiRoverParallelEnv, SingleRoverEnv


def test_gymnasium_api():
    check_env(SingleRoverEnv(WorldConfig(width=9, height=9, num_agents=1, max_steps=8)), skip_render_check=False)


def test_pettingzoo_parallel_api():
    env = MultiRoverParallelEnv(WorldConfig(width=9, height=9, num_agents=3, max_steps=8))
    parallel_api_test(env, num_cycles=40)

