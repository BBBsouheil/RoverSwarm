from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from .config import WorldConfig
from .envs import SingleRoverEnv
from .features import SpatialFeatureExtractor
from .shared_vec_env import SharedSwarmVecEnv


ROOT = Path(__file__).resolve().parents[2]


class WallClockLimit(BaseCallback):
    def __init__(self, max_seconds: float):
        super().__init__()
        self.max_seconds = max_seconds
        self.started = 0.0

    def _on_training_start(self) -> None:
        self.started = time.monotonic()

    def _on_step(self) -> bool:
        return time.monotonic() - self.started < self.max_seconds


def load_profile(name_or_path: str) -> tuple[dict[str, Any], Path]:
    path = Path(name_or_path)
    if not path.exists():
        path = ROOT / "configs" / f"{name_or_path}.json"
    if not path.exists():
        raise FileNotFoundError(f"Profil introuvable: {name_or_path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def train(
    profile: str = "smoke",
    mode: str = "shared",
    device: str = "cpu",
    resume: str | None = None,
    output_dir: str | Path | None = None,
    total_timesteps: int | None = None,
    enable_communication: bool = False,
    seed_override: int | None = None,
    tag: str | None = None,
    initialize_from: str | None = None,
) -> dict[str, Any]:
    data, profile_path = load_profile(profile)
    world_cfg = WorldConfig.from_dict(data["world"])
    world_cfg.num_agents = 1 if mode == "single" else 3
    world_cfg.comm.enabled = bool(enable_communication)
    training = data["training"]
    algorithm_name = str(training.get("algorithm", "ppo"))
    if algorithm_name == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        algorithm_class: Any = MaskablePPO
    elif algorithm_name == "ppo":
        algorithm_class = PPO
    else:
        raise ValueError(f"Algorithme inconnu: {algorithm_name}")
    seed = int(training["seed"] if seed_override is None else seed_override)
    budget = int(total_timesteps or training[f"total_timesteps_{mode}"])
    max_seconds = float(training["max_seconds"])
    output = Path(output_dir) if output_dir else ROOT / "artifacts"
    model_dir, log_dir = output / "models", output / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_comm" if enable_communication else ""
    seed_suffix = f"_seed{seed}" if seed_override is not None else ""
    tag_suffix = f"_{tag}" if tag else ""
    stem = f"ppo_{mode}_{Path(profile_path).stem}{suffix}{seed_suffix}{tag_suffix}"
    model_path = model_dir / stem

    if mode == "single":
        env: Any = SingleRoverEnv(world_cfg)
    elif mode == "shared":
        env = SharedSwarmVecEnv(world_cfg, seed=seed)
    else:
        raise ValueError("mode doit valoir 'single' ou 'shared'")

    if training.get("extractor") == "spatial_cnn":
        spatial_channels = 6 if world_cfg.observation_version == 3 else 5
        policy_kwargs = {
            "features_extractor_class": SpatialFeatureExtractor,
            "features_extractor_kwargs": {
                "grid_shape": (world_cfg.height, world_cfg.width), "channels": spatial_channels,
                "scalar_size": 9, "features_dim": int(training.get("features_dim", 256)),
            },
            "net_arch": {"pi": [64], "vf": [64]},
            "activation_fn": torch.nn.ReLU,
        }
    else:
        policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation_fn": torch.nn.Tanh}
    if resume and initialize_from:
        raise ValueError("--resume et --initialize-from sont incompatibles")
    initialization_report = {"copied_tensors": 0, "expanded_tensors": 0}
    if resume:
        model = algorithm_class.load(resume, env=env, device=device)
    else:
        model = algorithm_class(
            "MlpPolicy", env, device=device, seed=seed, verbose=1,
            n_steps=int(training["n_steps"]), batch_size=int(training["batch_size"]),
            n_epochs=int(training["n_epochs"]), learning_rate=float(training["learning_rate"]),
            gamma=float(training.get("gamma", 0.99)), gae_lambda=float(training.get("gae_lambda", 0.95)),
            ent_coef=float(training.get("ent_coef", 0.0)), clip_range=float(training.get("clip_range", 0.2)),
            policy_kwargs=policy_kwargs,
        )
        if initialize_from:
            source = algorithm_class.load(initialize_from, device=device)
            source_state = source.policy.state_dict()
            target_state = model.policy.state_dict()
            for key, target_value in target_state.items():
                source_value = source_state.get(key)
                if source_value is None:
                    continue
                if source_value.shape == target_value.shape:
                    target_state[key] = source_value.detach().to(target_value.device).clone()
                    initialization_report["copied_tensors"] += 1
                elif (
                    source_value.ndim == 4 and target_value.ndim == 4
                    and target_value.shape[0] == source_value.shape[0]
                    and target_value.shape[1] == source_value.shape[1] + 1
                    and target_value.shape[2:] == source_value.shape[2:]
                ):
                    expanded = target_value.detach().clone()
                    expanded[:, :source_value.shape[1]] = source_value.to(expanded.device)
                    expanded[:, source_value.shape[1]:] = 0.0
                    target_state[key] = expanded
                    initialization_report["expanded_tensors"] += 1
            if initialization_report["copied_tensors"] == 0:
                raise RuntimeError("Aucun paramètre compatible trouvé dans le modèle d'initialisation")
            model.policy.load_state_dict(target_state)
    model.set_logger(configure(str(log_dir / stem), ["stdout", "csv"]))
    before = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
    transitions_before = int(model.num_timesteps)
    started = time.monotonic()
    model.learn(total_timesteps=budget, callback=WallClockLimit(max_seconds), reset_num_timesteps=not bool(resume))
    elapsed = time.monotonic() - started
    changed = any(not torch.equal(before[key], value.detach().cpu()) for key, value in model.policy.state_dict().items())
    model.save(model_path)
    reloaded = algorithm_class.load(model_path, device=device)
    if mode == "single":
        sample, _ = env.reset(seed=seed + 1)
    else:
        sample = env.reset()
    predict_kwargs: dict[str, Any] = {"deterministic": True}
    if algorithm_name == "maskable_ppo":
        predict_kwargs["action_masks"] = env.world.action_masks() if mode == "shared" else env.action_masks()[None, :]
    predicted, _ = reloaded.predict(sample, **predict_kwargs)
    if not np.asarray(predicted).size or np.any(np.asarray(predicted) < 0) or np.any(np.asarray(predicted) >= 5):
        raise RuntimeError("Le modèle rechargé a produit une action invalide")
    if not changed:
        raise RuntimeError("Aucun paramètre du réseau n'a changé")
    actual_transitions = int(model.num_timesteps) - transitions_before
    world_steps = int(getattr(env, "world_steps", actual_transitions))
    metadata = {
        "profile": str(profile_path), "mode": mode, "communication": enable_communication,
        "algorithm": algorithm_name,
        "seed": seed, "requested_transition_budget": budget,
        "actual_individual_transitions": actual_transitions,
        "model_total_individual_transitions": int(model.num_timesteps),
        "actual_world_steps": world_steps,
        "elapsed_seconds": elapsed, "max_seconds": max_seconds,
        "optimization_verified_by_parameter_change": changed,
        "initialized_from": initialize_from,
        "initialization": initialization_report,
        "model": str(model_path.with_suffix(".zip")), "device": str(model.device),
        "config": world_cfg.to_dict(),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "sb3_contrib": importlib.metadata.version("sb3-contrib"),
            "pettingzoo": importlib.metadata.version("pettingzoo"),
            "numpy": np.__version__,
        },
    }
    metadata_path = log_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    env.close()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîner PPO pour RoverSwarm")
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--mode", choices=("single", "shared"), default="shared")
    parser.add_argument("--device", default="cpu", help="cpu, cuda ou auto")
    parser.add_argument("--resume", help="Chemin d'un modèle .zip à reprendre")
    parser.add_argument("--timesteps", type=int, help="Remplace le budget du profil")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--seed", type=int, help="Remplace la graine et crée un modèle distinct")
    parser.add_argument("--tag", help="Identifiant ajouté au modèle et aux journaux")
    parser.add_argument("--initialize-from", help="Transfère les poids compatibles d'un modèle vers une nouvelle architecture")
    args = parser.parse_args()
    result = train(
        args.profile, args.mode, args.device, args.resume,
        total_timesteps=args.timesteps,
        enable_communication=args.communication,
        seed_override=args.seed,
        tag=args.tag,
        initialize_from=args.initialize_from,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
