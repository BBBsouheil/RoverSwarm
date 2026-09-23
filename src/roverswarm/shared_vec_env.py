from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .config import WorldConfig
from .world import RoverWorld


class SharedSwarmVecEnv(VecEnv):
    """Adapte un monde synchrone en lot d'agents pour un PPO à paramètres partagés.

    SB3 voit N emplacements vectorisés. Contrairement à un VecEnv ordinaire,
    ils partagent un monde : les N actions sont appliquées ensemble et le lot de N
    observations suivant est renvoyé. Une transition SB3 est une transition agent ;
    un pas du monde produit donc N transitions corrélées.
    """

    def __init__(self, config: WorldConfig, seed: int = 0, map_name: str | None = None):
        self.world = RoverWorld(config, map_name=map_name)
        self.world.render_mode = "rgb_array"
        initial = self.world.reset(seed=seed)
        first = initial[self.world.agent_ids[0]]
        import gymnasium as gym
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=first.shape, dtype=np.float32)
        action_space = gym.spaces.Discrete(5)
        super().__init__(len(self.world.agent_ids), observation_space, action_space)
        self._actions: np.ndarray | None = None
        self._next_seed: int | None = seed
        self.world_steps = 0

    def _batch(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([observations[aid] for aid in self.world.agent_ids]).astype(np.float32)

    def reset(self) -> np.ndarray:
        observations = self.world.reset(seed=self._next_seed)
        self._next_seed = None
        self.reset_infos = [
            {"coverage": self.world.coverage, "active": True} for _ in self.world.agent_ids
        ]
        return self._batch(observations)

    def step_async(self, actions: np.ndarray) -> None:
        values = np.asarray(actions).reshape(-1)
        if len(values) != self.num_envs:
            raise ValueError(f"{self.num_envs} actions attendues")
        self._actions = values

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_async doit précéder step_wait")
        action_dict = {aid: int(self._actions[i]) for i, aid in enumerate(self.world.agent_ids)}
        result = self.world.step(action_dict)
        self.world_steps += 1
        observations = self._batch(result.observations)
        rewards = np.asarray([result.rewards[aid] for aid in self.world.agent_ids], dtype=np.float32)
        dones = np.asarray([
            result.terminations[aid] or result.truncations[aid] for aid in self.world.agent_ids
        ], dtype=bool)
        infos = [dict(result.infos[aid]) for aid in self.world.agent_ids]
        if bool(dones.all()):
            terminal = observations.copy()
            truncated = all(result.truncations.values())
            observations = self._batch(self.world.reset(seed=None))
            for i, info in enumerate(infos):
                info["terminal_observation"] = terminal[i]
                info["TimeLimit.truncated"] = truncated
        self._actions = None
        return observations, rewards, dones, infos

    def close(self) -> None:
        return None

    def get_images(self) -> list[np.ndarray]:
        image = self.world.render_rgb()
        return [image for _ in range(self.num_envs)]

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._next_seed = seed
        return [None if seed is None else seed + i for i in range(self.num_envs)]

    def _indices(self, indices: None | int | Iterable[int]) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        value = getattr(self.world, attr_name)
        return [value for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.world, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        if method_name == "action_masks":
            masks = self.world.action_masks()
            return [masks[i] for i in self._indices(indices)]
        result = getattr(self.world, method_name)(*method_args, **method_kwargs)
        return [result for _ in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._indices(indices)]
