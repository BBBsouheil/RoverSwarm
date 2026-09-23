from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from .config import WorldConfig
from .world import RoverWorld


class SingleRoverEnv(gym.Env[np.ndarray, int]):
    """Interface Gymnasium mono-robot du moteur commun."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, config: WorldConfig | None = None, map_name: str | None = None):
        cfg = config or WorldConfig(num_agents=1)
        cfg.num_agents = 1
        self.world = RoverWorld(cfg, map_name=map_name)
        initial = self.world.reset(seed=0)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=initial["rover_0"].shape, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.render_mode = "rgb_array"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self.world.reset(seed=seed)["rover_0"]
        return obs, {"coverage": self.world.coverage, "active": True}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.world.step({"rover_0": int(action)})
        return (
            result.observations["rover_0"], result.rewards["rover_0"],
            result.terminations["rover_0"], result.truncations["rover_0"],
            result.infos["rover_0"],
        )

    def render(self) -> np.ndarray:
        return self.world.render_rgb()

    def action_masks(self) -> np.ndarray:
        return self.world.action_mask("rover_0")

    def close(self) -> None:
        return None


class MultiRoverParallelEnv(ParallelEnv[str, np.ndarray, int]):
    """Interface PettingZoo ParallelEnv, agents fixes jusqu'à la fin globale."""

    metadata = {"name": "roverswarm_parallel_v0", "render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, config: WorldConfig | None = None, map_name: str | None = None):
        cfg = config or WorldConfig(num_agents=3)
        self.world = RoverWorld(cfg, map_name=map_name)
        initial = self.world.reset(seed=0)
        self.possible_agents = list(self.world.agent_ids)
        self.agents = self.possible_agents[:]
        shape = initial[self.possible_agents[0]].shape
        self._observation_spaces = {
            aid: spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32) for aid in self.possible_agents
        }
        self._action_spaces = {aid: spaces.Discrete(5) for aid in self.possible_agents}
        self.render_mode = "rgb_array"

    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        self.agents = self.possible_agents[:]
        observations = self.world.reset(seed=seed)
        infos = {aid: {"coverage": self.world.coverage, "active": True} for aid in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("Une action est requise pour chaque emplacement d'agent actif dans l'API")
        result = self.world.step(actions)
        if all(result.terminations.values()) or all(result.truncations.values()):
            self.agents = []
        return result.observations, result.rewards, result.terminations, result.truncations, result.infos

    def render(self) -> np.ndarray:
        return self.world.render_rgb()

    def close(self) -> None:
        return None
