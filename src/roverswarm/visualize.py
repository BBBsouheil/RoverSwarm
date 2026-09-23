from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from .config import WorldConfig
from .policies import FrontierPolicy, PPOPolicy, RandomPolicy
from .train import load_profile
from .world import FREE, OBSTACLE, UNKNOWN, RoverWorld


PALETTE = [(72, 214, 154), (79, 157, 255), (255, 177, 66), (224, 94, 135)]


def make_policy(name: str, model: str | None, seed: int):
    if name == "random":
        return RandomPolicy(seed)
    if name == "frontier":
        return FrontierPolicy()
    if not model or not Path(model).exists():
        raise FileNotFoundError("PPO demandé, mais aucun --model .zip valide n'a été fourni. Utilisez --policy frontier ou random.")
    return PPOPolicy(model)


def run_headless(world: RoverWorld, policy, steps: int) -> None:
    for _ in range(steps):
        result = world.step(policy.actions(world))
        if all(result.terminations.values()) or all(result.truncations.values()):
            break
    print(f"{policy.label}: couverture={world.coverage:.3f}, pas={world.step_count}, collisions={world.metrics['collisions']}")


def run_window(world: RoverWorld, policy, seed: int) -> None:
    import pygame
    pygame.init()
    cell = max(16, min(30, 700 // max(world.grid.shape)))
    grid_w, grid_h = world.grid.shape[1] * cell, world.grid.shape[0] * cell
    panel_w = 330
    screen = pygame.display.set_mode((grid_w + panel_w, max(grid_h, 620)))
    pygame.display.set_caption("RoverSwarm — Mission Control")
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    paused, one_step, global_view, show_sensor, show_paths = False, False, True, True, True
    selected, fps, running = 0, 8, True
    last_update = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_n: one_step = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): fps = min(60, fps + 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS): fps = max(1, fps - 2)
                elif event.key == pygame.K_TAB: selected = (selected + 1) % len(world.agent_ids)
                elif event.key == pygame.K_g: global_view = not global_view
                elif event.key == pygame.K_v: show_sensor = not show_sensor
                elif event.key == pygame.K_t: show_paths = not show_paths
                elif event.key == pygame.K_r:
                    seed += 1; world.reset(seed=seed); paused = False
        now = time.monotonic()
        if (not paused or one_step) and now - last_update >= 1.0 / fps:
            result = world.step(policy.actions(world))
            last_update, one_step = now, False
            if all(result.terminations.values()) or all(result.truncations.values()):
                paused = True
        screen.fill((8, 12, 20))
        selected_aid = world.agent_ids[selected]
        memory = world.memories[selected_aid]
        for y in range(world.grid.shape[0]):
            for x in range(world.grid.shape[1]):
                value = int(world.grid[y, x]) if global_view else int(memory[y, x])
                color = (20, 27, 38) if value == UNKNOWN else ((45, 57, 72) if value == FREE else (12, 16, 24))
                pygame.draw.rect(screen, color, (x*cell, y*cell, cell-1, cell-1))
        if show_sensor:
            overlay = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
            for y, x in map(tuple, np.argwhere(world.visible[selected_aid])):
                pygame.draw.rect(overlay, (81, 170, 255, 35), (x*cell, y*cell, cell-1, cell-1))
            screen.blit(overlay, (0, 0))
        if show_paths:
            for i, aid in enumerate(world.agent_ids):
                points = [(x*cell+cell//2, y*cell+cell//2) for y, x in world.trajectories[aid]]
                if len(points) > 1: pygame.draw.lines(screen, PALETTE[i], False, points, 2)
        if world.config.comm.enabled:
            for i, aid in enumerate(world.agent_ids):
                sender = world.last_sender[aid]
                if sender and world.step_count - world.last_message_step[aid] <= 3:
                    ry, rx = world.positions[aid]; sy, sx = sender
                    pygame.draw.line(screen, (180, 110, 255), (sx*cell+cell//2, sy*cell+cell//2), (rx*cell+cell//2, ry*cell+cell//2), 2)
        for i, aid in enumerate(world.agent_ids):
            y, x = world.positions[aid]
            pygame.draw.circle(screen, PALETTE[i], (x*cell+cell//2, y*cell+cell//2), cell//3)
            if i == selected: pygame.draw.circle(screen, (255,255,255), (x*cell+cell//2, y*cell+cell//2), cell//2-1, 2)
        px = grid_w + 22
        lines = [
            ("ROVERSWARM / MISSION", (225,235,245), font),
            (policy.label, (72,214,154), font),
            (f"pas monde   {world.step_count:4d}", (180,195,210), small),
            (f"couverture  {world.coverage:6.1%}", (180,195,210), small),
            (f"collisions  {world.metrics['collisions']:4d}", (180,195,210), small),
            (f"canal       {'ACTIF' if world.config.comm.enabled else 'COUPÉ'}", (180,195,210), small),
            ("", (0,0,0), small),
        ]
        for i, aid in enumerate(world.agent_ids):
            lines.append((f"R{i}  {'ON ' if world.active[aid] else 'OFF'} {world.energy[aid]:6.1f}", PALETTE[i], small))
        lines += [
            ("", (0,0,0), small), ("ESPACE  pause", (145,160,175), small),
            ("N       un pas", (145,160,175), small), ("+/-     vitesse", (145,160,175), small),
            ("R       nouvelle carte", (145,160,175), small), ("TAB     robot sélectionné", (145,160,175), small),
            ("G       vue globale/mémoire", (145,160,175), small), ("V/T     capteur/trajectoires", (145,160,175), small),
        ]
        y_text = 24
        for text, color, used_font in lines:
            screen.blit(used_font.render(text, True, color), (px, y_text)); y_text += 26
        pygame.display.flip(); clock.tick(60)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tableau de mission 2D RoverSwarm")
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--policy", choices=("random", "frontier", "ppo"), default="frontier")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    data, _ = load_profile(args.profile)
    config = WorldConfig.from_dict(data["world"]); config.num_agents = 3; config.comm.enabled = args.communication
    world = RoverWorld(config); world.reset(seed=args.seed)
    policy = make_policy(args.policy, args.model, args.seed)
    run_headless(world, policy, args.steps) if args.headless else run_window(world, policy, args.seed)


if __name__ == "__main__":
    main()

