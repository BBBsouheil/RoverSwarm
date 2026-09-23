from __future__ import annotations

import argparse
import json
from statistics import mean

from roverswarm.config import WorldConfig
from roverswarm.missions import SCENARIOS, MissionRuntime, apply_scenario
from roverswarm.policies import MaskedPPOPolicy
from roverswarm.train import ROOT, load_profile
from roverswarm.world import RoverWorld


def main() -> None:
    parser = argparse.ArgumentParser(description="Playtest automatique des missions RoverSwarm")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=3101)
    parser.add_argument("--model", default=str(ROOT / "models" / "roverswarm_v5_seed97.zip"))
    args = parser.parse_args()

    data, _ = load_profile("benchmark_v5")
    policy = MaskedPPOPolicy(args.model, device="cpu", deterministic=False)
    summary: dict[str, object] = {}
    for key in SCENARIOS:
        rows = []
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            config = WorldConfig.from_dict(data["world"])
            config.num_agents = 3
            scenario = apply_scenario(config, key)
            world = RoverWorld(config)
            world.reset(seed=seed)
            runtime = MissionRuntime(world, scenario)
            policy.set_seed(seed)
            while not runtime.finished(world):
                world.step(policy.actions(world))
                runtime.update(world)
            rows.append({
                "seed": seed,
                "success": runtime.success(world),
                "coverage": world.coverage,
                "objectives": runtime.objective_done,
                "objectives_total": runtime.objective_total,
                "steps": world.step_count,
            })
        summary[key] = {
            "success_rate": mean(float(row["success"]) for row in rows),
            "coverage_mean": mean(float(row["coverage"]) for row in rows),
            "objectives_mean": mean(float(row["objectives"]) for row in rows),
            "episodes": rows,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
