from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import WorldConfig


@dataclass(frozen=True, slots=True)
class MissionScenario:
    key: str
    name: str
    subtitle: str
    briefing: tuple[str, ...]
    objective: str = "coverage"
    site_count: int = 0
    communication: bool = True
    radio_range: int | None = None
    radio_delay: int | None = None
    radio_loss: float | None = None
    energy: float | None = None
    target_coverage: float | None = None


SCENARIOS: dict[str, MissionScenario] = {
    "survey": MissionScenario(
        key="survey",
        name="LEVÉ ARES",
        subtitle="Cartographier le secteur et maintenir la cohésion de l'essaim.",
        briefing=(
            "Objectif primaire : couvrir 90 % du terrain accessible.",
            "Liaison nominale : partage cartographique à courte portée.",
            "Priorité : préserver l'énergie et limiter les impacts.",
        ),
        target_coverage=0.90,
    ),
    "blackout": MissionScenario(
        key="blackout",
        name="ZONE DE SILENCE",
        subtitle="Explorer sous liaison instable et latence renforcée.",
        briefing=(
            "Objectif primaire : localiser et réactiver deux relais.",
            "La liaison démarre presque inutilisable puis se rétablit par étapes.",
            "Une couverture minimale de 72 % reste exigée.",
        ),
        objective="relays",
        site_count=2,
        radio_range=4,
        radio_delay=4,
        radio_loss=0.55,
        target_coverage=0.72,
    ),
    "endurance": MissionScenario(
        key="endurance",
        name="RÉSERVE CRITIQUE",
        subtitle="Atteindre la couverture cible avec une réserve d'énergie limitée.",
        briefing=(
            "Objectif primaire : inspecter deux anomalies géologiques.",
            "Énergie initiale réduite : chaque déplacement compte.",
            "Une couverture de 60 % et un rover encore actif sont exigés.",
        ),
        objective="samples",
        site_count=2,
        energy=145.0,
        target_coverage=0.60,
    ),
}


def apply_scenario(config: WorldConfig, key: str) -> MissionScenario:
    try:
        scenario = SCENARIOS[key]
    except KeyError as exc:
        raise ValueError(f"Scénario inconnu: {key}") from exc
    config.comm.enabled = scenario.communication
    if scenario.radio_range is not None:
        config.comm.range = scenario.radio_range
    if scenario.radio_delay is not None:
        config.comm.delay = scenario.radio_delay
    if scenario.radio_loss is not None:
        config.comm.loss_probability = scenario.radio_loss
    if scenario.energy is not None:
        config.initial_energy = scenario.energy
    if scenario.target_coverage is not None:
        # Les missions à sites possèdent leur propre condition de victoire.
        # Le moteur reste ouvert jusqu'à 100 % ou jusqu'à sa limite normale.
        config.target_coverage = scenario.target_coverage if scenario.objective == "coverage" else 1.0
    config.validate()
    return scenario


class MissionRuntime:
    def __init__(self, world: Any, scenario: MissionScenario):
        self.scenario = scenario
        self.sites = self._place_sites(world, scenario.site_count)
        self.completed: set[int] = set()

    @staticmethod
    def _place_sites(world: Any, count: int) -> tuple[tuple[int, int], ...]:
        if count == 0:
            return ()
        starts = list(world.positions.values())
        candidates = [
            tuple(map(int, cell)) for cell in np.argwhere(world.accessible)
            if min(abs(int(cell[0]) - y) + abs(int(cell[1]) - x) for y, x in starts) > world.config.sensor_radius + 2
        ]
        distance = lambda cell: min(abs(cell[0] - y) + abs(cell[1] - x) for y, x in starts)
        ordered = sorted(candidates, key=lambda cell: (distance(cell), cell))
        selected: list[tuple[int, int]] = []
        window = max(5, len(ordered) // 10)
        for index in range(count):
            # Les objectifs s'éloignent progressivement du déploiement, sans
            # être systématiquement cachés dans les extrémités de la carte.
            quantile = 0.75 * (index + 1) / (count + 1)
            center = round(quantile * (len(ordered) - 1))
            pool = ordered[max(0, center - window):min(len(ordered), center + window + 1)]
            pool = [cell for cell in pool if cell not in selected]
            anchors = selected or starts
            choice = max(pool, key=lambda cell: min(abs(cell[0] - y) + abs(cell[1] - x) for y, x in anchors))
            selected.append(choice)
        return tuple(selected)

    @property
    def objective_total(self) -> int:
        return len(self.sites)

    @property
    def objective_done(self) -> int:
        return len(self.completed)

    def update(self, world: Any) -> list[str]:
        events: list[str] = []
        for index, site in enumerate(self.sites):
            if index in self.completed:
                continue
            if any(abs(site[0] - y) + abs(site[1] - x) <= world.config.sensor_radius for y, x in world.positions.values()):
                self.completed.add(index)
                noun = "relais rétabli" if self.scenario.objective == "relays" else "échantillon validé"
                events.append(f"OBJ  {noun} {len(self.completed)}/{len(self.sites)}")
        if self.scenario.objective == "relays" and self.sites:
            progress = len(self.completed) / len(self.sites)
            world.config.comm.range = 4 + round(3 * progress)
            world.config.comm.delay = max(1, 4 - round(3 * progress))
            world.config.comm.loss_probability = max(0.05, 0.55 - 0.50 * progress)
        return events

    def success(self, world: Any) -> bool:
        coverage_target = float(self.scenario.target_coverage or 0.0)
        if self.scenario.objective == "coverage":
            return world.coverage >= coverage_target
        sites_done = self.objective_done == self.objective_total
        energy_ok = any(world.active.values()) if self.scenario.objective == "samples" else True
        return sites_done and world.coverage >= coverage_target and energy_ok

    def finished(self, world: Any) -> bool:
        return self.success(world) or not any(world.active.values()) or world.step_count >= world.config.max_steps

    def objective_label(self, world: Any) -> str:
        if self.scenario.objective == "coverage":
            return f"COUVERTURE {world.coverage:.1%} / {float(self.scenario.target_coverage):.0%}"
        noun = "RELAIS" if self.scenario.objective == "relays" else "SITES"
        return f"{noun} {self.objective_done}/{self.objective_total}  //  COUV. {world.coverage:.1%}"

    def progress(self, world: Any) -> float:
        if self.scenario.objective == "coverage":
            return min(1.0, world.coverage / float(self.scenario.target_coverage))
        site_progress = self.objective_done / max(1, self.objective_total)
        coverage_progress = min(1.0, world.coverage / float(self.scenario.target_coverage))
        return 0.7 * site_progress + 0.3 * coverage_progress


def mission_report(world: Any, runtime: MissionRuntime, seed: int) -> dict[str, Any]:
    scenario = runtime.scenario
    energies = {aid: round(float(world.energy[aid]), 3) for aid in world.agent_ids}
    success = runtime.success(world)
    return {
        "scenario": scenario.key,
        "scenario_name": scenario.name,
        "seed": seed,
        "success": success,
        "coverage": float(world.coverage),
        "target_coverage": float(scenario.target_coverage or 0.0),
        "objective_type": scenario.objective,
        "objectives_completed": runtime.objective_done,
        "objectives_total": runtime.objective_total,
        "world_steps": int(world.step_count),
        "active_rovers": sum(bool(world.active[aid]) for aid in world.agent_ids),
        "remaining_energy": energies,
        "metrics": {key: float(value) if isinstance(value, float) else int(value) for key, value in world.metrics.items()},
    }
