from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass(slots=True)
class CommConfig:
    enabled: bool = False
    range: int = 6
    frequency: int = 3
    capacity: int = 8
    delay: int = 1
    loss_probability: float = 0.05

    def validate(self) -> None:
        if self.range < 0 or self.frequency < 1 or self.capacity < 0 or self.delay < 0:
            raise ValueError("Paramètres de communication invalides")
        if not 0.0 <= self.loss_probability <= 1.0:
            raise ValueError("loss_probability doit être dans [0, 1]")


@dataclass(slots=True)
class WorldConfig:
    width: int = 24
    height: int = 24
    num_agents: int = 3
    obstacle_probability: float = 0.18
    sensor_radius: int = 2
    initial_energy: float = 160.0
    move_cost: float = 1.0
    wait_cost: float = 0.2
    collision_extra_cost: float = 0.5
    max_steps: int = 180
    target_coverage: float = 0.90
    observation_version: int = 1
    novelty_reward: float = 1.0
    individual_novelty_reward: float = 0.0
    collision_penalty: float = 0.35
    revisit_penalty: float = 0.0
    wait_penalty: float = 0.0
    time_penalty: float = 0.01
    comm: CommConfig = field(default_factory=CommConfig)

    def validate(self) -> None:
        if self.width < 5 or self.height < 5:
            raise ValueError("La carte doit mesurer au moins 5 x 5")
        if self.num_agents < 1:
            raise ValueError("num_agents doit être positif")
        if not 0.0 <= self.obstacle_probability < 0.65:
            raise ValueError("obstacle_probability doit être dans [0, 0.65[")
        if self.sensor_radius < 0 or self.initial_energy <= 0 or self.max_steps < 1:
            raise ValueError("Rayon, énergie ou durée invalides")
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target_coverage doit être dans ]0, 1]")
        if self.observation_version not in (1, 2, 3):
            raise ValueError("observation_version doit valoir 1, 2 ou 3")
        if min(self.novelty_reward, self.individual_novelty_reward, self.collision_penalty, self.revisit_penalty, self.wait_penalty, self.time_penalty) < 0:
            raise ValueError("Les coefficients de récompense doivent être positifs")
        self.comm.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldConfig":
        raw = dict(data)
        raw["comm"] = CommConfig(**raw.get("comm", {}))
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "WorldConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
