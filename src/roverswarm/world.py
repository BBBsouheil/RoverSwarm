from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import WorldConfig

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OBSTACLE = np.int8(1)

ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # haut
    1: (1, 0),   # bas
    2: (0, -1),  # gauche
    3: (0, 1),   # droite
    4: (0, 0),   # attendre
}


@dataclass(slots=True, frozen=True)
class Packet:
    sender: int
    receiver: int
    sender_position: tuple[int, int]
    cells: tuple[tuple[int, int, int], ...]
    due_step: int
    size_bytes: int


@dataclass(slots=True)
class StepResult:
    observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    terminations: dict[str, bool]
    truncations: dict[str, bool]
    infos: dict[str, dict[str, object]]


FIXED_MAPS: dict[str, tuple[str, ...]] = {
    "open7": (
        ".......", ".......", ".......", ".......", ".......", ".......", ".......",
    ),
    "rooms9": (
        ".........", ".###.###.", ".........", ".#..#..#.", ".........",
        ".#..#..#.", ".........", ".###.###.", ".........",
    ),
    "corridor9": (
        ".........", "####.####", ".........", ".#######.", ".........",
        ".#######.", ".........", "####.####", ".........",
    ),
}


class RoverWorld:
    """Moteur synchrone, sans dépendance Gym/PettingZoo/Pygame.

    Le capteur révèle toutes les cellules du carré de rayon Chebyshev configuré,
    sans modèle d'occlusion. La carte vraie ne figure jamais dans l'observation.
    """

    def __init__(self, config: WorldConfig, map_name: str | None = None):
        config.validate()
        self.config = config
        self.map_name = map_name
        self.agent_ids = tuple(f"rover_{i}" for i in range(config.num_agents))
        self.rng = np.random.default_rng()
        self.comm_rng = np.random.default_rng()
        self.grid = np.zeros((config.height, config.width), dtype=np.int8)
        self.accessible = np.ones_like(self.grid, dtype=bool)
        self.positions: dict[str, tuple[int, int]] = {}
        self.starts: dict[str, tuple[int, int]] = {}
        self.energy: dict[str, float] = {}
        self.active: dict[str, bool] = {}
        self.memories: dict[str, np.ndarray] = {}
        self.local_known: dict[str, np.ndarray] = {}
        self.visible: dict[str, np.ndarray] = {}
        self.seen_at: dict[str, np.ndarray] = {}
        self.visits: dict[str, np.ndarray] = {}
        self.trajectories: dict[str, list[tuple[int, int]]] = {}
        self.last_sender: dict[str, tuple[int, int] | None] = {}
        self.last_message_step: dict[str, int] = {}
        self.pending_packets: list[Packet] = []
        self.step_count = 0
        self.seed_value: int | None = None
        self.team_known_accessible = np.zeros_like(self.grid, dtype=bool)
        self.metrics: dict[str, float | int] = {}

    @property
    def observation_size(self) -> int:
        if self.config.observation_version == 3:
            return 6 * self.grid.size + 9
        if self.config.observation_version == 2:
            return 5 * self.grid.size + 9
        return 2 * self.grid.size + 7

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        # Avec seed=None, on poursuit les flux existants : les épisodes suivants
        # restent reproductibles sans rejouer indéfiniment la même carte.
        if seed is not None or self.seed_value is None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)
            comm_seed = None if seed is None else (int(seed) ^ 0x5EEDC0DE)
            self.comm_rng = np.random.default_rng(comm_seed)
        self.grid = self._build_map()
        self.accessible = self._largest_free_component(self.grid)
        candidates = list(map(tuple, np.argwhere(self.accessible)))
        if len(candidates) < self.config.num_agents:
            raise RuntimeError("Carte sans assez de cellules accessibles pour les robots")
        indices = self.rng.choice(len(candidates), size=self.config.num_agents, replace=False)
        self.positions = {aid: candidates[int(indices[i])] for i, aid in enumerate(self.agent_ids)}
        self.starts = dict(self.positions)
        self.energy = {aid: float(self.config.initial_energy) for aid in self.agent_ids}
        self.active = {aid: True for aid in self.agent_ids}
        self.memories = {
            aid: np.full(self.grid.shape, UNKNOWN, dtype=np.int8) for aid in self.agent_ids
        }
        self.local_known = {
            aid: np.zeros(self.grid.shape, dtype=bool) for aid in self.agent_ids
        }
        self.visible = {aid: np.zeros(self.grid.shape, dtype=bool) for aid in self.agent_ids}
        self.seen_at = {aid: np.full(self.grid.shape, -1, dtype=np.int32) for aid in self.agent_ids}
        self.visits = {aid: np.zeros(self.grid.shape, dtype=np.int16) for aid in self.agent_ids}
        self.trajectories = {aid: [self.positions[aid]] for aid in self.agent_ids}
        for aid in self.agent_ids:
            self.visits[aid][self.positions[aid]] = 1
        self.last_sender = {aid: None for aid in self.agent_ids}
        self.last_message_step = {aid: -10**9 for aid in self.agent_ids}
        self.pending_packets = []
        self.step_count = 0
        self.metrics = {
            "collisions": 0, "redundant_moves": 0, "energy_consumed": 0.0,
            "messages_emitted": 0, "messages_received": 0, "messages_lost": 0,
            "communication_bytes": 0,
        }
        for aid in self.agent_ids:
            self._sense(aid)
        self.team_known_accessible = self._team_known_accessible()
        return self.observations()

    def _build_map(self) -> np.ndarray:
        if self.map_name:
            if self.map_name not in FIXED_MAPS:
                raise ValueError(f"Carte fixe inconnue: {self.map_name}")
            rows = FIXED_MAPS[self.map_name]
            grid = np.array([[OBSTACLE if c == "#" else FREE for c in row] for row in rows], dtype=np.int8)
            self.config.height, self.config.width = grid.shape
            return grid
        grid = (self.rng.random((self.config.height, self.config.width)) < self.config.obstacle_probability).astype(np.int8)
        grid[[0, -1], :] = OBSTACLE
        grid[:, [0, -1]] = OBSTACLE
        # Une vaste zone centrale libre évite les cartes sans intérêt.
        cy, cx = self.config.height // 2, self.config.width // 2
        grid[max(1, cy - 2):cy + 3, max(1, cx - 2):cx + 3] = FREE
        return grid

    @staticmethod
    def _largest_free_component(grid: np.ndarray) -> np.ndarray:
        unseen = grid == FREE
        best: list[tuple[int, int]] = []
        height, width = grid.shape
        for y, x in map(tuple, np.argwhere(unseen)):
            if not unseen[y, x]:
                continue
            component: list[tuple[int, int]] = []
            queue = deque([(y, x)])
            unseen[y, x] = False
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for dy, dx in ACTION_DELTAS.values():
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and unseen[ny, nx]:
                        unseen[ny, nx] = False
                        queue.append((ny, nx))
            if len(component) > len(best):
                best = component
        mask = np.zeros_like(grid, dtype=bool)
        if best:
            yy, xx = zip(*best)
            mask[np.asarray(yy), np.asarray(xx)] = True
        return mask

    def _sense(self, aid: str) -> None:
        y, x = self.positions[aid]
        r = self.config.sensor_radius
        y0, y1 = max(0, y - r), min(self.grid.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(self.grid.shape[1], x + r + 1)
        mask = np.zeros(self.grid.shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        self.visible[aid] = mask
        self.memories[aid][mask] = self.grid[mask]
        self.local_known[aid][mask] = True
        self.seen_at[aid][mask] = self.step_count

    def _team_known_accessible(self) -> np.ndarray:
        known = np.zeros_like(self.grid, dtype=bool)
        for aid in self.agent_ids:
            known |= self.memories[aid] == FREE
        return known & self.accessible

    @property
    def coverage(self) -> float:
        total = int(self.accessible.sum())
        return float(self.team_known_accessible.sum() / total) if total else 0.0

    def observation(self, aid: str) -> np.ndarray:
        memory = self.memories[aid]
        encoded = np.where(memory == UNKNOWN, 0.0, np.where(memory == FREE, 0.5, 1.0)).astype(np.float32)
        y, x = self.positions[aid]
        h, w = self.grid.shape
        sender = self.last_sender[aid]
        if sender is None:
            sy, sx, recent = -1.0, -1.0, 0.0
        else:
            sy = sender[0] / max(1, h - 1)
            sx = sender[1] / max(1, w - 1)
            recent = max(0.0, 1.0 - (self.step_count - self.last_message_step[aid]) / max(1, self.config.max_steps))
        legacy_scalars = np.asarray([
            y / max(1, h - 1), x / max(1, w - 1),
            self.energy[aid] / self.config.initial_energy, float(self.active[aid]),
            sy, sx, recent,
        ], dtype=np.float32)
        if self.config.observation_version == 1:
            return np.concatenate((encoded.ravel(), self.visible[aid].astype(np.float32).ravel(), legacy_scalars))
        marker = np.zeros(self.grid.shape, dtype=np.float32)
        marker[y, x] = 1.0
        action_knowledge: list[float] = []
        for action in range(4):
            dy, dx = ACTION_DELTAS[action]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or memory[ny, nx] == OBSTACLE:
                action_knowledge.append(-1.0)
            elif memory[ny, nx] == FREE:
                action_knowledge.append(1.0)
            else:
                action_knowledge.append(0.0)
        spatial: tuple[np.ndarray, ...] = (
            (memory == FREE).astype(np.float32),
            (memory == OBSTACLE).astype(np.float32),
            self.visible[aid].astype(np.float32),
            np.minimum(self.visits[aid], 5).astype(np.float32) / 5.0,
            marker,
        )
        if self.config.observation_version == 3:
            received_only = ((memory != UNKNOWN) & ~self.local_known[aid]).astype(np.float32)
            spatial = (*spatial, received_only)
        scalars = np.asarray([
            self.energy[aid] / self.config.initial_energy, float(self.active[aid]),
            sy, sx, recent, *action_knowledge,
        ], dtype=np.float32)
        return np.concatenate((*[channel.ravel() for channel in spatial], scalars))

    def observations(self) -> dict[str, np.ndarray]:
        return {aid: self.observation(aid) for aid in self.agent_ids}

    def action_mask(self, aid: str) -> np.ndarray:
        """Masque de sûreté local : interdit uniquement les obstacles déjà connus."""
        mask = np.ones(5, dtype=bool)
        if not self.active[aid]:
            mask[:4] = False
            return mask
        y, x = self.positions[aid]
        memory = self.memories[aid]
        for action in range(4):
            dy, dx = ACTION_DELTAS[action]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]):
                mask[action] = False
            elif memory[ny, nx] == OBSTACLE:
                mask[action] = False
        return mask

    def action_masks(self) -> np.ndarray:
        return np.stack([self.action_mask(aid) for aid in self.agent_ids])

    def step(self, actions: dict[str, int]) -> StepResult:
        expected = set(self.agent_ids)
        if set(actions) != expected:
            raise ValueError(f"Actions attendues pour {sorted(expected)}")
        if any(int(action) not in ACTION_DELTAS for action in actions.values()):
            raise ValueError("Action hors de l'espace Discrete(5)")
        self.step_count += 1
        active_before = dict(self.active)
        origins = dict(self.positions)
        proposed = dict(origins)
        moving: set[str] = set()
        collisions: set[str] = set()
        revisited: set[str] = set()
        for aid in self.agent_ids:
            action = int(actions[aid])
            if not self.active[aid] or action == 4:
                continue
            moving.add(aid)
            y, x = origins[aid]
            dy, dx = ACTION_DELTAS[action]
            target = (y + dy, x + dx)
            ty, tx = target
            if not (0 <= ty < self.grid.shape[0] and 0 <= tx < self.grid.shape[1]) or self.grid[ty, tx] == OBSTACLE:
                collisions.add(aid)
            else:
                proposed[aid] = target

        groups: dict[tuple[int, int], list[str]] = {}
        for aid in self.agent_ids:
            groups.setdefault(proposed[aid], []).append(aid)
        for group in groups.values():
            if len(group) > 1:
                collisions.update(aid for aid in group if aid in moving)

        for i, aid in enumerate(self.agent_ids):
            for bid in self.agent_ids[i + 1:]:
                if proposed[aid] == origins[bid] and proposed[bid] == origins[aid] and aid in moving and bid in moving:
                    collisions.update((aid, bid))

        changed = True
        while changed:
            changed = False
            blocked_cells = {origins[aid] for aid in self.agent_ids if aid not in moving or aid in collisions}
            for aid in self.agent_ids:
                if aid in moving and aid not in collisions and proposed[aid] in blocked_cells:
                    collisions.add(aid)
                    changed = True

        for aid in self.agent_ids:
            action = int(actions[aid])
            if not self.active[aid]:
                continue
            before = self.energy[aid]
            cost = self.config.wait_cost if action == 4 else self.config.move_cost
            if aid in collisions:
                cost += self.config.collision_extra_cost
            self.energy[aid] = max(0.0, before - cost)
            self.metrics["energy_consumed"] = float(self.metrics["energy_consumed"]) + min(before, cost)
            if aid in moving and aid not in collisions:
                destination = proposed[aid]
                if self.visits[aid][destination] > 0:
                    self.metrics["redundant_moves"] = int(self.metrics["redundant_moves"]) + 1
                    revisited.add(aid)
                self.positions[aid] = destination
                self.visits[aid][destination] += 1
            if self.energy[aid] <= 0.0:
                self.active[aid] = False
            self.trajectories[aid].append(self.positions[aid])

        self.metrics["collisions"] = int(self.metrics["collisions"]) + len(collisions)
        before_known = self.team_known_accessible.copy()
        before_agent_known = {aid: self.memories[aid] == FREE for aid in self.agent_ids}
        for aid in self.agent_ids:
            if self.active[aid]:
                self._sense(aid)
            else:
                self.visible[aid].fill(False)
        # Le crédit individuel appartient au capteur qui a réellement découvert
        # la cellule. Une copie reçue par radio ne doit jamais être récompensée
        # comme une observation locale.
        locally_sensed_new_by_agent = {
            aid: ((self.memories[aid] == FREE) & ~before_agent_known[aid] & self.accessible)
            for aid in self.agent_ids
        }
        comm_delta = self._communication_step()
        self.team_known_accessible = self._team_known_accessible()
        new_team_mask = self.team_known_accessible & ~before_known
        newly_known = int(new_team_mask.sum())
        credited: dict[str, float] = {aid: 0.0 for aid in self.agent_ids}
        sensed_new_by_agent = {
            aid: locally_sensed_new_by_agent[aid] & new_team_mask
            for aid in self.agent_ids
        }
        for y, x in map(tuple, np.argwhere(new_team_mask)):
            observers = [aid for aid in self.agent_ids if sensed_new_by_agent[aid][y, x]]
            if observers:
                share = 1.0 / len(observers)
                for aid in observers:
                    credited[aid] += share
        rewards: dict[str, float] = {}
        reward_components_by_agent: dict[str, dict[str, float]] = {}
        if (self.config.observation_version == 1 and self.config.individual_novelty_reward == 0.0
                and self.config.revisit_penalty == 0.0 and self.config.wait_penalty == 0.0):
            legacy_components = {
                "novelty": self.config.novelty_reward * newly_known,
                "collision": -self.config.collision_penalty * len(collisions),
                "time": -self.config.time_penalty,
            }
            for aid in self.agent_ids:
                reward_components_by_agent[aid] = dict(legacy_components)
                rewards[aid] = float(sum(legacy_components.values()))
        else:
            active_count = max(1, sum(active_before.values()))
            for aid in self.agent_ids:
                if not active_before[aid]:
                    components = {"team_novelty": 0.0, "credited_novelty": 0.0, "collision": 0.0, "revisit": 0.0, "wait": 0.0, "time": 0.0}
                else:
                    components = {
                        "team_novelty": self.config.novelty_reward * newly_known / active_count,
                        "credited_novelty": self.config.individual_novelty_reward * credited[aid],
                        "collision": -self.config.collision_penalty * float(aid in collisions),
                        "revisit": -self.config.revisit_penalty * float(aid in revisited),
                        "wait": -self.config.wait_penalty * float(int(actions[aid]) == 4),
                        "time": -self.config.time_penalty,
                    }
                reward_components_by_agent[aid] = components
                rewards[aid] = float(sum(components.values()))
        success = self.coverage >= self.config.target_coverage
        exhausted = not any(self.active.values())
        terminated = success or exhausted
        truncated = self.step_count >= self.config.max_steps and not terminated
        observations = self.observations()
        infos: dict[str, dict[str, object]] = {}
        for aid in self.agent_ids:
            infos[aid] = {
                "active": self.active[aid], "coverage": self.coverage,
                "collision": aid in collisions, "new_accessible_cells": newly_known,
                "credited_new_cells": credited[aid],
                "reward_components": dict(reward_components_by_agent[aid]), "world_step": self.step_count,
                "termination_reason": "success" if success else ("energy_exhausted" if exhausted else None),
                "communication": dict(comm_delta),
            }
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminations={aid: terminated for aid in self.agent_ids},
            truncations={aid: truncated for aid in self.agent_ids},
            infos=infos,
        )

    def _communication_step(self) -> dict[str, int]:
        delta = {"emitted": 0, "received": 0, "lost": 0, "bytes": 0}
        cfg = self.config.comm
        if not cfg.enabled:
            return delta
        if self.step_count % cfg.frequency == 0 and cfg.capacity > 0:
            for sender in self.agent_ids:
                if not self.active[sender]:
                    continue
                known_coords = np.argwhere(self.memories[sender] != UNKNOWN)
                ranked = sorted(
                    map(tuple, known_coords),
                    key=lambda cell: (-int(self.seen_at[sender][cell]), cell[0], cell[1]),
                )[:cfg.capacity]
                cells = tuple((y, x, int(self.memories[sender][y, x])) for y, x in ranked)
                if not cells:
                    continue
                delta["emitted"] += 1
                self.metrics["messages_emitted"] = int(self.metrics["messages_emitted"]) + 1
                sy, sx = self.positions[sender]
                for receiver in self.agent_ids:
                    if receiver == sender or not self.active[receiver]:
                        continue
                    ry, rx = self.positions[receiver]
                    if abs(sy - ry) + abs(sx - rx) > cfg.range:
                        continue
                    size = 5 + 5 * len(cells)  # uint16 y/x + uint8 count; cellule: uint16 y/x + uint8 valeur
                    delta["bytes"] += size
                    self.metrics["communication_bytes"] = int(self.metrics["communication_bytes"]) + size
                    if self.comm_rng.random() < cfg.loss_probability:
                        delta["lost"] += 1
                        self.metrics["messages_lost"] = int(self.metrics["messages_lost"]) + 1
                    else:
                        self.pending_packets.append(Packet(
                            sender=int(sender.rsplit("_", 1)[1]), receiver=int(receiver.rsplit("_", 1)[1]),
                            sender_position=(sy, sx), cells=cells,
                            due_step=self.step_count + cfg.delay, size_bytes=size,
                        ))
        remaining: list[Packet] = []
        for packet in self.pending_packets:
            if packet.due_step > self.step_count:
                remaining.append(packet)
                continue
            receiver = f"rover_{packet.receiver}"
            for y, x, value in packet.cells:
                self.memories[receiver][y, x] = np.int8(value)
                self.seen_at[receiver][y, x] = self.step_count
            self.last_sender[receiver] = packet.sender_position
            self.last_message_step[receiver] = self.step_count
            delta["received"] += 1
            self.metrics["messages_received"] = int(self.metrics["messages_received"]) + 1
        self.pending_packets = remaining
        return delta

    def render_rgb(self, cell_size: int = 16) -> np.ndarray:
        """Vue omnisciente réservée au diagnostic; sans effet sur les observations."""
        h, w = self.grid.shape
        image = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
        colors = {int(FREE): (38, 48, 62), int(OBSTACLE): (11, 15, 23)}
        for y in range(h):
            for x in range(w):
                image[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size] = colors[int(self.grid[y, x])]
        palette = ((62, 207, 142), (78, 153, 255), (255, 174, 66), (221, 92, 128))
        for i, aid in enumerate(self.agent_ids):
            y, x = self.positions[aid]
            pad = max(1, cell_size // 4)
            image[y*cell_size+pad:(y+1)*cell_size-pad, x*cell_size+pad:(x+1)*cell_size-pad] = palette[i % len(palette)]
        return image

    def set_positions_for_test(self, positions: Iterable[tuple[int, int]]) -> None:
        """Utilitaire explicite pour les tests des mouvements conjoints."""
        values = list(positions)
        if len(values) != len(self.agent_ids):
            raise ValueError("Nombre de positions incorrect")
        self.positions = dict(zip(self.agent_ids, values, strict=True))
        self.starts = dict(self.positions)
        self.trajectories = {aid: [self.positions[aid]] for aid in self.agent_ids}
        for aid in self.agent_ids:
            self._sense(aid)
