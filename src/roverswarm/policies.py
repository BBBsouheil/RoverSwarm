from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from .world import ACTION_DELTAS, FREE, OBSTACLE, UNKNOWN, RoverWorld


class Policy(Protocol):
    label: str
    def actions(self, world: RoverWorld) -> dict[str, int]: ...


class RandomPolicy:
    label = "ALÉATOIRE"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def actions(self, world: RoverWorld) -> dict[str, int]:
        return {aid: int(self.rng.integers(0, 5)) for aid in world.agent_ids}


class FrontierPolicy:
    """Planification BFS vers la frontière la plus proche sur la seule mémoire locale."""

    label = "FRONTIÈRES"

    @staticmethod
    def _frontier(memory: np.ndarray, cell: tuple[int, int]) -> bool:
        y, x = cell
        if memory[y, x] != FREE:
            return False
        for action in range(4):
            dy, dx = ACTION_DELTAS[action]
            ny, nx = y + dy, x + dx
            if 0 <= ny < memory.shape[0] and 0 <= nx < memory.shape[1] and memory[ny, nx] == UNKNOWN:
                return True
        return False

    def _action(self, world: RoverWorld, aid: str) -> int:
        if not world.active[aid]:
            return 4
        memory = world.memories[aid]
        start = world.positions[aid]
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        goal: tuple[int, int] | None = None
        while queue:
            cell = queue.popleft()
            if self._frontier(memory, cell):
                goal = cell
                break
            y, x = cell
            for action in range(4):
                dy, dx = ACTION_DELTAS[action]
                nxt = (y + dy, x + dx)
                ny, nx = nxt
                if (0 <= ny < memory.shape[0] and 0 <= nx < memory.shape[1]
                        and memory[ny, nx] == FREE and nxt not in parent):
                    parent[nxt] = (cell, action)
                    queue.append(nxt)
        if goal is None:
            return 4
        if goal == start:
            y, x = start
            for action in range(4):
                dy, dx = ACTION_DELTAS[action]
                ny, nx = y + dy, x + dx
                if 0 <= ny < memory.shape[0] and 0 <= nx < memory.shape[1] and memory[ny, nx] == UNKNOWN:
                    return action
            return 4
        cursor = goal
        while parent[cursor] is not None and parent[cursor][0] != start:
            cursor = parent[cursor][0]
        edge = parent[cursor]
        return 4 if edge is None else edge[1]

    def actions(self, world: RoverWorld) -> dict[str, int]:
        return {aid: self._action(world, aid) for aid in world.agent_ids}


class PPOPolicy:
    label = "PPO CHARGÉ"

    def __init__(self, model_path: str, device: str = "auto", deterministic: bool = True):
        from stable_baselines3 import PPO
        from pathlib import Path
        path = Path(model_path)
        if not path.exists() and path.suffix != ".zip" and path.with_suffix(".zip").exists():
            path = path.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"Modèle PPO introuvable: {path.resolve()}")
        self.model = PPO.load(path, device=device)
        self.deterministic = deterministic
        self.label = "PPO DÉTERMINISTE" if deterministic else "PPO STOCHASTIQUE"

    def set_seed(self, seed: int) -> None:
        self.model.set_random_seed(seed)

    def actions(self, world: RoverWorld) -> dict[str, int]:
        batch = np.stack([world.observation(aid) for aid in world.agent_ids])
        actions, _ = self.model.predict(batch, deterministic=self.deterministic)
        return {aid: int(np.asarray(actions).reshape(-1)[i]) for i, aid in enumerate(world.agent_ids)}


class MaskedPPOPolicy(PPOPolicy):
    def __init__(self, model_path: str, device: str = "auto", deterministic: bool = True):
        from pathlib import Path
        from sb3_contrib import MaskablePPO
        path = Path(model_path)
        if not path.exists() and path.suffix != ".zip" and path.with_suffix(".zip").exists():
            path = path.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"Modèle PPO masqué introuvable: {path.resolve()}")
        self.model = MaskablePPO.load(path, device=device)
        self.deterministic = deterministic
        self.label = "PPO MASQUÉ" if deterministic else "PPO MASQUÉ STOCHASTIQUE"

    def actions(self, world: RoverWorld) -> dict[str, int]:
        batch = np.stack([world.observation(aid) for aid in world.agent_ids])
        actions, _ = self.model.predict(
            batch, action_masks=world.action_masks(), deterministic=self.deterministic
        )
        return {aid: int(np.asarray(actions).reshape(-1)[i]) for i, aid in enumerate(world.agent_ids)}
