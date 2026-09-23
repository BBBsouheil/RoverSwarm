from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from roverswarm.config import WorldConfig
from roverswarm.shared_vec_env import SharedSwarmVecEnv


def test_shared_ppo_optimizes_saves_loads_and_predicts(tmp_path: Path):
    env = SharedSwarmVecEnv(WorldConfig(width=9, height=9, num_agents=3, max_steps=8), seed=11)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs={"net_arch": [16]}, device="cpu", seed=11)
    before = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    model.learn(total_timesteps=24)
    assert any(not torch.equal(before[key], value) for key, value in model.policy.state_dict().items())
    path = tmp_path / "ppo_test"
    model.save(path)
    assert path.with_suffix(".zip").exists()
    loaded = PPO.load(path, device="cpu")
    actions, _ = loaded.predict(env.reset(), deterministic=True)
    assert np.asarray(actions).shape == (3,)
    assert np.all((np.asarray(actions) >= 0) & (np.asarray(actions) < 5))

