from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import WorldConfig
from .policies import FrontierPolicy, MaskedPPOPolicy, PPOPolicy, RandomPolicy
from .train import ROOT, load_profile
from .world import RoverWorld


def run_episode(config: WorldConfig, policy: Any, seed: int) -> dict[str, Any]:
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)
    world = RoverWorld(config)
    world.reset(seed=seed)
    threshold_step: int | None = 0 if world.coverage >= config.target_coverage else None
    while True:
        result = world.step(policy.actions(world))
        if threshold_step is None and world.coverage >= config.target_coverage:
            threshold_step = world.step_count
        if all(result.terminations.values()) or all(result.truncations.values()):
            break
    return {
        "seed": seed, "coverage": world.coverage,
        "success": bool(world.coverage >= config.target_coverage),
        "steps": world.step_count, "steps_to_threshold": threshold_step,
        **world.metrics,
    }


def evaluate(
    profile: str = "smoke", model_path: str | None = None,
    output_dir: str | Path | None = None, include_ppo: bool = True,
    communication: bool = False, perturbed_channel: bool = False,
    stochastic: bool = False,
    masked_ppo: bool | None = None,
    split: str = "evaluation",
    episodes: int | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    data, profile_path = load_profile(profile)
    config = WorldConfig.from_dict(data["world"])
    config.num_agents = 3
    config.comm.enabled = communication
    channel = "none"
    if communication:
        channel = "perturbed" if perturbed_channel else "normal"
        if perturbed_channel:
            config.comm.loss_probability = min(1.0, max(0.5, config.comm.loss_probability))
            config.comm.delay = max(3, config.comm.delay)
    policies: list[tuple[str, Any]] = [
        ("random", RandomPolicy(seed=991)), ("frontier", FrontierPolicy()),
    ]
    if include_ppo:
        if not model_path:
            raise ValueError("--model est requis pour évaluer PPO; aucune référence ne sera maquillée en PPO")
        if masked_ppo is None:
            masked_ppo = data.get("training", {}).get("algorithm") == "maskable_ppo"
        label = "ppo_masked" if masked_ppo else "ppo"
        if stochastic:
            label += "_stochastic"
        policy_class = MaskedPPOPolicy if masked_ppo else PPOPolicy
        policies.append((label, policy_class(model_path, device="cpu", deterministic=not stochastic)))
    if split not in {"validation", "evaluation"}:
        raise ValueError("split doit valoir 'validation' ou 'evaluation'")
    split_data = data[split]
    limit = int(episodes or split_data.get("max_episodes", len(split_data["seeds"])))
    seeds = [int(seed) for seed in split_data["seeds"][:limit]]
    rows: list[dict[str, Any]] = []
    for label, policy in policies:
        for seed in seeds:
            row = run_episode(config, policy, seed)
            row.update({"policy": label, "channel": channel, "profile": Path(profile_path).stem})
            rows.append(row)
    output = Path(output_dir) if output_dir else ROOT / "artifacts" / "results"
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{channel}" if channel != "none" else ""
    if stochastic:
        suffix += "_stochastic"
    if masked_ppo:
        suffix += "_masked"
    split_suffix = "_validation" if split == "validation" else ""
    tag_suffix = f"_{tag}" if tag else ""
    stem = f"evaluation_{Path(profile_path).stem}{split_suffix}{suffix}{tag_suffix}"
    csv_path, json_path, plot_path = output / f"{stem}.csv", output / f"{stem}.json", output / f"{stem}.png"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {}
    for label, _ in policies:
        selected = [row for row in rows if row["policy"] == label]
        coverages = [float(row["coverage"]) for row in selected]
        summary[label] = {
            "episodes": len(selected), "coverage_mean": mean(coverages),
            "coverage_std": pstdev(coverages) if len(coverages) > 1 else 0.0,
            "success_rate": mean(float(row["success"]) for row in selected),
            "collisions_mean": mean(float(row["collisions"]) for row in selected),
            "energy_mean": mean(float(row["energy_consumed"]) for row in selected),
            "communication_bytes_mean": mean(float(row["communication_bytes"]) for row in selected),
        }
    payload = {"experiment": {"profile": str(profile_path), "split": split, "channel": channel, "seeds": seeds}, "summary": summary, "episodes": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    labels = list(summary)
    means = [summary[label]["coverage_mean"] for label in labels]
    errors = [summary[label]["coverage_std"] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, yerr=errors, capsize=5, color=["#78909c", "#26a69a", "#42a5f5"][:len(labels)])
    ax.axhline(config.target_coverage, color="#ef5350", linestyle="--", label="seuil")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Couverture accessible finale"); ax.set_title("RoverSwarm — moyenne ± écart-type")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_path, dpi=160); plt.close(fig)
    return {"csv": str(csv_path), "json": str(json_path), "plot": str(plot_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reproductible RoverSwarm")
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--model")
    parser.add_argument("--without-ppo", action="store_true")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--perturbed-channel", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Échantillonne la distribution PPO au lieu de prendre l'argmax")
    parser.add_argument("--masked-ppo", action="store_true", help="Force le chargeur MaskablePPO")
    parser.add_argument("--split", choices=("validation", "evaluation"), default="evaluation")
    parser.add_argument("--episodes", type=int, help="Limite le nombre de cartes du split")
    parser.add_argument("--tag", help="Identifiant ajouté aux fichiers de sortie (ex. seed71)")
    args = parser.parse_args()
    result = evaluate(
        args.profile, args.model,
        include_ppo=not args.without_ppo,
        communication=args.communication,
        perturbed_channel=args.perturbed_channel,
        stochastic=args.stochastic,
        masked_ppo=True if args.masked_ppo else None,
        split=args.split,
        episodes=args.episodes,
        tag=args.tag,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
