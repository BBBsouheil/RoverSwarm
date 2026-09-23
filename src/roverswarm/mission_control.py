from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import WorldConfig
from .missions import SCENARIOS, MissionRuntime, MissionScenario, apply_scenario, mission_report
from .policies import FrontierPolicy, MaskedPPOPolicy, PPOPolicy, RandomPolicy
from .train import ROOT, load_profile
from .world import FREE, OBSTACLE, UNKNOWN, RoverWorld, StepResult


WIDTH, HEIGHT = 1420, 900
MAP_X, MAP_Y, MAP_SIZE = 34, 116, 744
HUD_X = 812
COLORS = ((64, 224, 170), (78, 157, 255), (255, 181, 71), (225, 98, 145))
DEFAULT_MODEL = ROOT / "models" / "roverswarm_v5_seed97.zip"
DEFAULT_SCENARIO_SEEDS = {"survey": 3105, "blackout": 3108, "endurance": 3105}


def _policy(name: str, model: str | None, seed: int, stochastic: bool):
    if name == "random":
        return RandomPolicy(seed)
    if name == "frontier":
        return FrontierPolicy()
    if not model:
        raise FileNotFoundError("Un fichier --model est obligatoire pour une politique PPO")
    if name == "masked-ppo":
        return MaskedPPOPolicy(model, device="cpu", deterministic=not stochastic)
    return PPOPolicy(model, device="cpu", deterministic=not stochastic)


def _font(pygame: Any, size: int, bold: bool = False):
    return pygame.font.SysFont("segoeui", size, bold=bold)


def _text(surface: Any, font: Any, value: str, position: tuple[int, int], color=(202, 214, 225)) -> None:
    surface.blit(font.render(value, True, color), position)


def _panel(pygame: Any, surface: Any, rect: Any, title: str | None = None) -> None:
    pygame.draw.rect(surface, (15, 22, 32), rect, border_radius=10)
    pygame.draw.rect(surface, (42, 58, 74), rect, width=1, border_radius=10)
    if title:
        _text(surface, _font(pygame, 13, True), title.upper(), (rect.x + 16, rect.y + 11), (119, 147, 168))


def _draw_briefing(pygame: Any, screen: Any, selected: int) -> None:
    screen.fill((6, 10, 16))
    for x in range(0, WIDTH, 70):
        pygame.draw.line(screen, (9, 16, 23), (x, 0), (x, HEIGHT), 1)
    _text(screen, _font(pygame, 30, True), "ARES EXPEDITION // CENTRE DE MISSION", (70, 58), (220, 231, 237))
    _text(screen, _font(pygame, 14), "ROVERSWARM 1.0  —  SÉLECTION DU SCÉNARIO", (72, 102), (91, 126, 145))
    keys = list(SCENARIOS)
    for index, key in enumerate(keys):
        scenario = SCENARIOS[key]
        rect = pygame.Rect(72, 170 + index * 190, 1275, 158)
        _panel(pygame, screen, rect)
        if index == selected:
            pygame.draw.rect(screen, (64, 224, 170), rect, 2, border_radius=10)
        color = (64, 224, 170) if index == selected else (126, 148, 162)
        _text(screen, _font(pygame, 18, True), f"0{index + 1}", (rect.x + 24, rect.y + 24), color)
        _text(screen, _font(pygame, 23, True), scenario.name, (rect.x + 86, rect.y + 20), (220, 231, 237))
        _text(screen, _font(pygame, 15), scenario.subtitle, (rect.x + 86, rect.y + 55), (142, 163, 176))
        for line_index, line in enumerate(scenario.briefing):
            _text(screen, _font(pygame, 13), f"— {line}", (rect.x + 86, rect.y + 86 + line_index * 19), (116, 139, 153))
    _text(screen, _font(pygame, 14), "↑/↓ ou 1–3 sélectionner    ENTRÉE déployer l'essaim    ÉCHAP quitter", (72, 805), (114, 145, 162))


def _briefing_loop(pygame: Any, screen: Any, initial_key: str) -> str | None:
    keys = list(SCENARIOS)
    selected = keys.index(initial_key)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return None
            if event.key in (pygame.K_UP, pygame.K_w):
                selected = (selected - 1) % len(keys)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                selected = (selected + 1) % len(keys)
            elif pygame.K_1 <= event.key <= pygame.K_3:
                selected = event.key - pygame.K_1
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return keys[selected]
        _draw_briefing(pygame, screen, selected)
        pygame.display.flip()
        clock.tick(30)


def _save_report(world: RoverWorld, runtime: MissionRuntime, seed: int) -> Path:
    output = ROOT / "artifacts" / "results" / "last_mission.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = mission_report(world, runtime, seed)
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _draw_debrief(pygame: Any, screen: Any, world: RoverWorld, runtime: MissionRuntime, report_path: Path | None) -> None:
    scenario = runtime.scenario
    success = runtime.success(world)
    color = (64, 224, 170) if success else (245, 91, 91)
    shade = pygame.Surface((MAP_SIZE, MAP_SIZE), pygame.SRCALPHA)
    shade.fill((4, 8, 13, 188))
    screen.blit(shade, (MAP_X, MAP_Y))
    rect = pygame.Rect(MAP_X + 105, MAP_Y + 155, MAP_SIZE - 210, 405)
    pygame.draw.rect(screen, (10, 17, 25), rect, border_radius=12)
    pygame.draw.rect(screen, color, rect, 2, border_radius=12)
    heading = "MISSION RÉUSSIE" if success else "MISSION TERMINÉE"
    rendered = _font(pygame, 30, True).render(heading, True, color)
    screen.blit(rendered, (rect.centerx - rendered.get_width() // 2, rect.y + 31))
    subtitle = _font(pygame, 14).render(scenario.name, True, (130, 153, 168))
    screen.blit(subtitle, (rect.centerx - subtitle.get_width() // 2, rect.y + 76))
    values = (
        ("OBJECTIF", runtime.objective_label(world)),
        ("DURÉE MISSION", f"{world.step_count} pas terrain"),
        ("COLLISIONS", f"{int(world.metrics['collisions'])}"),
        ("ÉNERGIE CONSOMMÉE", f"{float(world.metrics['energy_consumed']):.1f} u"),
        ("LIAISONS REÇUES", f"{int(world.metrics['messages_received'])}"),
    )
    for index, (label, value) in enumerate(values):
        y = rect.y + 120 + index * 42
        _text(screen, _font(pygame, 13), label, (rect.x + 48, y), (113, 141, 158))
        rendered_value = _font(pygame, 17, True).render(value, True, (211, 223, 230))
        screen.blit(rendered_value, (rect.right - 48 - rendered_value.get_width(), y - 3))
    saved = report_path.name if report_path else "non enregistré"
    _text(screen, _font(pygame, 12), f"RAPPORT  {saved}", (rect.x + 48, rect.bottom - 66), (99, 126, 142))
    prompt = _font(pygame, 13, True).render("R REJOUER    B BRIEFING    ÉCHAP QUITTER", True, color)
    screen.blit(prompt, (rect.centerx - prompt.get_width() // 2, rect.bottom - 35))


def _bar(pygame: Any, surface: Any, rect: Any, ratio: float, color: tuple[int, int, int]) -> None:
    ratio = max(0.0, min(1.0, ratio))
    pygame.draw.rect(surface, (28, 39, 51), rect, border_radius=4)
    if ratio > 0:
        fill = rect.copy(); fill.width = max(3, int(rect.width * ratio))
        pygame.draw.rect(surface, color, fill, border_radius=4)
    pygame.draw.rect(surface, (56, 72, 87), rect, width=1, border_radius=4)


def _mission_state(world: RoverWorld, paused: bool, runtime: MissionRuntime) -> tuple[str, tuple[int, int, int], str | None]:
    if runtime.success(world):
        return "MISSION RÉUSSIE", (64, 224, 170), "OBJECTIFS DE MISSION ATTEINTS"
    if not any(world.active.values()):
        return "MISSION ÉCHOUÉE", (245, 91, 91), "ESSAIM IMMOBILISÉ — OBJECTIFS INCOMPLETS"
    if world.step_count >= world.config.max_steps:
        return "MISSION ÉCHOUÉE", (245, 91, 91), "FENÊTRE OPÉRATIONNELLE — OBJECTIFS INCOMPLETS"
    if paused:
        return "PAUSE", (255, 188, 71), None
    return "EN COURS", (64, 224, 170), None


def _cell_hash(x: int, y: int, seed: int) -> int:
    value = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
    return value & 0xFFFFFFFF


def _combined_memory(world: RoverWorld) -> np.ndarray:
    memory = np.full(world.grid.shape, UNKNOWN, dtype=np.int8)
    for aid in world.agent_ids:
        known = world.memories[aid] != UNKNOWN
        memory[known] = world.memories[aid][known]
    return memory


def _draw_map(pygame: Any, screen: Any, world: RoverWorld, selected: int, view_mode: str,
              scan_phase: float, runtime: MissionRuntime) -> tuple[int, int, int]:
    rows, cols = world.grid.shape
    cell = max(8, min(MAP_SIZE // cols, MAP_SIZE // rows))
    draw_w, draw_h = cols * cell, rows * cell
    origin_x = MAP_X + (MAP_SIZE - draw_w) // 2
    origin_y = MAP_Y + (MAP_SIZE - draw_h) // 2
    pygame.draw.rect(screen, (7, 12, 18), (MAP_X - 8, MAP_Y - 8, MAP_SIZE + 16, MAP_SIZE + 16), border_radius=12)
    pygame.draw.rect(screen, (40, 58, 72), (MAP_X - 8, MAP_Y - 8, MAP_SIZE + 16, MAP_SIZE + 16), 1, border_radius=12)
    selected_aid = world.agent_ids[selected]
    if view_mode == "global":
        displayed = world.grid
    elif view_mode == "selected":
        displayed = world.memories[selected_aid]
    else:
        displayed = _combined_memory(world)
    seed = int(world.seed_value or 0)
    for y in range(rows):
        for x in range(cols):
            value = int(displayed[y, x])
            rect = pygame.Rect(origin_x + x * cell, origin_y + y * cell, cell + 1, cell + 1)
            noise = _cell_hash(x, y, seed) % 17
            if value == UNKNOWN:
                base = 9 + noise // 5
                color = (base, base + 5, base + 9)
                pygame.draw.rect(screen, color, rect)
                if cell >= 20 and (x + y) % 3 == 0:
                    pygame.draw.line(screen, (18, 27, 36), rect.topleft, rect.bottomright, 1)
            elif value == FREE:
                color = (36 + noise, 42 + noise, 43 + noise // 2)
                pygame.draw.rect(screen, color, rect)
                if cell >= 18 and noise % 5 == 0:
                    px = rect.x + 4 + noise % max(5, cell - 8)
                    py = rect.y + 3 + (noise * 3) % max(5, cell - 7)
                    pygame.draw.circle(screen, (69, 70, 63), (px, py), 1)
            else:
                pygame.draw.rect(screen, (24, 28, 30), rect)
                inset = max(2, cell // 7)
                rock = rect.inflate(-2 * inset, -2 * inset)
                points = [
                    (rock.left, rock.centery), (rock.left + rock.width // 4, rock.top),
                    (rock.right - rock.width // 5, rock.top + 1), (rock.right, rock.centery),
                    (rock.right - rock.width // 4, rock.bottom), (rock.left + 2, rock.bottom - 2),
                ]
                pygame.draw.polygon(screen, (67 + noise, 65 + noise, 59 + noise // 2), points)
                pygame.draw.line(screen, (100, 95, 83), points[1], points[2], 1)
            pygame.draw.rect(screen, (26, 34, 40), rect, 1)

    for index, (y, x) in enumerate(runtime.sites):
        center = (origin_x + x * cell + cell // 2, origin_y + y * cell + cell // 2)
        completed = index in runtime.completed
        color = (64, 224, 170) if completed else ((185, 105, 255) if runtime.scenario.objective == "relays" else (255, 181, 71))
        pulse = max(9, cell // 3 + int(2 * (0.5 + 0.5 * math.sin(scan_phase + index))))
        pygame.draw.circle(screen, color, center, pulse, 2)
        if runtime.scenario.objective == "relays":
            pygame.draw.line(screen, color, (center[0], center[1] - 6), (center[0], center[1] + 6), 2)
            pygame.draw.arc(screen, color, (center[0] - 8, center[1] - 8, 16, 12), math.pi, 2 * math.pi, 2)
        else:
            points = ((center[0], center[1] - 7), (center[0] + 7, center[1] + 6), (center[0] - 7, center[1] + 6))
            pygame.draw.polygon(screen, color, points, 2)
        marker = _font(pygame, max(10, cell // 3), True).render(str(index + 1), True, color)
        screen.blit(marker, (center[0] + 8, center[1] - 10))

    sensor = pygame.Surface((draw_w, draw_h), pygame.SRCALPHA)
    pulse_alpha = int(22 + 15 * (0.5 + 0.5 * math.sin(scan_phase)))
    for y, x in map(tuple, np.argwhere(world.visible[selected_aid])):
        pygame.draw.rect(sensor, (*COLORS[selected], pulse_alpha), (x * cell, y * cell, cell, cell))
    screen.blit(sensor, (origin_x, origin_y))

    trail_layer = pygame.Surface((draw_w, draw_h), pygame.SRCALPHA)
    for i, aid in enumerate(world.agent_ids):
        trail = world.trajectories[aid][-100:]
        if len(trail) > 1:
            points = [(x * cell + cell // 2, y * cell + cell // 2) for y, x in trail]
            pygame.draw.lines(trail_layer, (*COLORS[i], 115), False, points, max(1, cell // 9))
    screen.blit(trail_layer, (origin_x, origin_y))

    if world.config.comm.enabled:
        comm_layer = pygame.Surface((draw_w, draw_h), pygame.SRCALPHA)
        for i, aid in enumerate(world.agent_ids):
            sender = world.last_sender[aid]
            if sender and world.step_count - world.last_message_step[aid] <= 3:
                ry, rx = world.positions[aid]; sy, sx = sender
                pygame.draw.line(comm_layer, (186, 105, 255, 190),
                                 (sx * cell + cell // 2, sy * cell + cell // 2),
                                 (rx * cell + cell // 2, ry * cell + cell // 2), 2)
        screen.blit(comm_layer, (origin_x, origin_y))

    for i, aid in enumerate(world.agent_ids):
        y, x = world.positions[aid]
        center = (origin_x + x * cell + cell // 2, origin_y + y * cell + cell // 2)
        trail = world.trajectories[aid]
        dy, dx = 0, 1
        for previous in reversed(trail[:-1]):
            if previous != trail[-1]:
                dy, dx = trail[-1][0] - previous[0], trail[-1][1] - previous[1]
                break
        angle = math.degrees(math.atan2(-dy, dx))
        size = max(18, int(cell * 0.82))
        rover = pygame.Surface((size, size), pygame.SRCALPHA)
        cy = size // 2
        body = pygame.Rect(size // 5, size // 3, 3 * size // 5, size // 3)
        pygame.draw.rect(rover, (24, 31, 36), (body.x - 2, body.y - 3, body.width + 4, 3), border_radius=2)
        pygame.draw.rect(rover, (24, 31, 36), (body.x - 2, body.bottom, body.width + 4, 3), border_radius=2)
        pygame.draw.rect(rover, COLORS[i], body, border_radius=3)
        pygame.draw.rect(rover, (16, 24, 29), body, 2, border_radius=3)
        pygame.draw.rect(rover, (61, 87, 111), (body.x + 3, body.y + 3, body.width // 3, body.height - 6))
        pygame.draw.line(rover, (220, 230, 226), (body.right - 3, cy), (size - 2, cy), 2)
        pygame.draw.circle(rover, (220, 230, 226), (size - 3, cy), 2)
        rover = pygame.transform.rotate(rover, angle)
        screen.blit(rover, rover.get_rect(center=center))
        if i == selected:
            pygame.draw.circle(screen, (225, 242, 248), center, max(10, cell // 2), 2)
        if not world.active[aid]:
            pygame.draw.line(screen, (245, 91, 91), (center[0]-7, center[1]-7), (center[0]+7, center[1]+7), 2)
    return origin_x, origin_y, cell


def _draw_hud(pygame: Any, screen: Any, world: RoverWorld, policy: Any, selected: int,
              view_mode: str, fps: int, paused: bool, events: list[str], runtime: MissionRuntime,
              show_end_banner: bool = True) -> None:
    scenario = runtime.scenario
    title_font, large, normal, small = _font(pygame, 18, True), _font(pygame, 28, True), _font(pygame, 15), _font(pygame, 12)
    _text(screen, title_font, "ARES EXPEDITION // ROVERSWARM", (34, 26), (220, 231, 237))
    _text(screen, small, scenario.name, (35, 53), (91, 126, 145))
    status, status_color, end_reason = _mission_state(world, paused, runtime)
    pygame.draw.circle(screen, status_color, (699, 39), 5)
    _text(screen, small, status, (712, 31), status_color)
    _text(screen, small, f"VUE {view_mode.upper()}  //  {fps:02d} Hz", (MAP_X, 86), (113, 141, 158))

    _panel(pygame, screen, pygame.Rect(HUD_X, 24, 574, 112), "Mission")
    if scenario.objective == "coverage":
        primary, caption = f"{world.coverage:05.1%}", "COUVERTURE ACCESSIBLE"
    else:
        primary = f"{runtime.objective_done}/{runtime.objective_total}"
        caption = "RELAIS RÉTABLIS" if scenario.objective == "relays" else "SITES INSPECTÉS"
    _text(screen, large, primary, (HUD_X + 18, 53), (72, 218, 174))
    _text(screen, small, caption, (HUD_X + 20, 88), (126, 148, 162))
    _bar(pygame, screen, pygame.Rect(HUD_X + 177, 59, 365, 12), runtime.progress(world), (53, 185, 145))
    _text(screen, small, f"couverture {world.coverage:.1%} / {float(scenario.target_coverage):.0%}", (HUD_X + 177, 82), (126, 148, 162))
    _text(screen, small, f"SOL {world.step_count:04d} / {world.config.max_steps:04d}", (HUD_X + 410, 82), (126, 148, 162))

    _panel(pygame, screen, pygame.Rect(HUD_X, 150, 574, 82), "Politique active")
    _text(screen, title_font, policy.label, (HUD_X + 18, 178), (86, 205, 255))
    _text(screen, small, "RÉSEAU CHARGÉ" if "PPO" in policy.label else "RÉFÉRENCE PROGRAMMÉE", (HUD_X + 360, 182), (113, 141, 158))

    y0 = 247
    for i, aid in enumerate(world.agent_ids):
        rect = pygame.Rect(HUD_X, y0 + i * 96, 574, 82)
        _panel(pygame, screen, rect)
        if i == selected:
            pygame.draw.rect(screen, COLORS[i], rect, 2, border_radius=10)
        _text(screen, title_font, f"R{i}", (rect.x + 16, rect.y + 13), COLORS[i])
        state = "ACTIF" if world.active[aid] else "HORS LIGNE"
        _text(screen, small, state, (rect.x + 57, rect.y + 18), (150, 170, 180) if world.active[aid] else (244, 99, 99))
        ratio = world.energy[aid] / world.config.initial_energy
        _bar(pygame, screen, pygame.Rect(rect.x + 57, rect.y + 43, 285, 9), ratio, COLORS[i])
        _text(screen, small, f"ÉNERGIE {world.energy[aid]:6.1f} u", (rect.x + 57, rect.y + 57), (126, 148, 162))
        py, px = world.positions[aid]
        _text(screen, normal, f"Y {py:02d}  X {px:02d}", (rect.x + 380, rect.y + 20), (205, 215, 220))
        _text(screen, small, f"visites {len(set(world.trajectories[aid])):03d}", (rect.x + 380, rect.y + 50), (126, 148, 162))

    metrics_y = y0 + len(world.agent_ids) * 96 + 3
    _panel(pygame, screen, pygame.Rect(HUD_X, metrics_y, 277, 112), "Télémétrie")
    _text(screen, normal, f"COLLISIONS   {int(world.metrics['collisions']):04d}", (HUD_X + 16, metrics_y + 37), (218, 151, 87))
    _text(screen, normal, f"ÉNERGIE      {float(world.metrics['energy_consumed']):06.1f}", (HUD_X + 16, metrics_y + 63), (190, 204, 213))
    _text(screen, normal, f"REDONDANCE   {int(world.metrics['redundant_moves']):04d}", (HUD_X + 16, metrics_y + 87), (190, 204, 213))
    _panel(pygame, screen, pygame.Rect(HUD_X + 297, metrics_y, 277, 112), "Liaison")
    channel = "ONLINE" if world.config.comm.enabled else "OFFLINE"
    _text(screen, normal, f"CANAL    {channel}", (HUD_X + 313, metrics_y + 37), (177, 111, 245) if world.config.comm.enabled else (120, 139, 151))
    _text(screen, normal, f"RX       {int(world.metrics['messages_received']):04d}", (HUD_X + 313, metrics_y + 63), (190, 204, 213))
    loss = int(round(world.config.comm.loss_probability * 100)) if world.config.comm.enabled else 0
    _text(screen, normal, f"PERTES   {loss:02d}%  /  {int(world.metrics['communication_bytes']):05d} o", (HUD_X + 313, metrics_y + 87), (190, 204, 213))

    events_y = metrics_y + 127
    _panel(pygame, screen, pygame.Rect(HUD_X, events_y, 574, 116), "Journal mission")
    for index, event in enumerate(events[-4:]):
        _text(screen, small, event, (HUD_X + 16, events_y + 34 + index * 19), (145, 164, 175))
    _text(screen, small, "ESPACE pause  N pas  +/- vitesse  R carte  B missions  TAB rover  G vue  F12 capture", (HUD_X, 873), (92, 119, 135))

    if end_reason and show_end_banner:
        overlay = pygame.Surface((MAP_SIZE - 32, 86), pygame.SRCALPHA)
        overlay.fill((8, 13, 20, 226))
        pygame.draw.rect(overlay, (*status_color, 255), overlay.get_rect(), 2, border_radius=8)
        screen.blit(overlay, (MAP_X + 16, MAP_Y + MAP_SIZE - 111))
        banner = _font(pygame, 22, True)
        detail = _font(pygame, 13)
        rendered = banner.render(status, True, status_color)
        screen.blit(rendered, (MAP_X + (MAP_SIZE - rendered.get_width()) // 2, MAP_Y + MAP_SIZE - 96))
        rendered_detail = detail.render(f"{end_reason}  //  R POUR RELANCER", True, (196, 210, 219))
        screen.blit(rendered_detail, (MAP_X + (MAP_SIZE - rendered_detail.get_width()) // 2, MAP_Y + MAP_SIZE - 62))


def render_frame(pygame: Any, screen: Any, world: RoverWorld, policy: Any, selected: int,
                 view_mode: str, fps: int, paused: bool, events: list[str], phase: float,
                 runtime: MissionRuntime, debrief: bool = False, report_path: Path | None = None) -> None:
    screen.fill((6, 10, 16))
    for x in range(0, WIDTH, 70):
        pygame.draw.line(screen, (9, 16, 23), (x, 0), (x, HEIGHT), 1)
    _draw_map(pygame, screen, world, selected, view_mode, phase, runtime)
    _draw_hud(pygame, screen, world, policy, selected, view_mode, fps, paused, events, runtime, not debrief)
    if debrief:
        _draw_debrief(pygame, screen, world, runtime, report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="RoverSwarm Mission Control")
    parser.add_argument("--profile", default="benchmark_v5")
    parser.add_argument("--policy", choices=("random", "frontier", "ppo", "masked-ppo"), default="masked-ppo")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--communication", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="survey")
    parser.add_argument("--briefing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60, help="Pas exécutés avant une capture automatisée")
    parser.add_argument("--screenshot", help="Enregistre une image puis quitte, sans ouvrir de fenêtre")
    args = parser.parse_args()
    seed_overridden = args.seed is not None
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT)) if args.screenshot else pygame.display.set_mode((WIDTH, HEIGHT))
    if not args.screenshot:
        pygame.display.set_caption("RoverSwarm — Mission Control")
    if not args.screenshot and args.briefing:
        chosen = _briefing_loop(pygame, screen, args.scenario)
        if chosen is None:
            pygame.quit()
            return
        args.scenario = chosen
    if args.seed is None:
        args.seed = DEFAULT_SCENARIO_SEEDS[args.scenario]
    data, _ = load_profile(args.profile)

    def build_world(scenario_key: str) -> tuple[RoverWorld, MissionRuntime]:
        config = WorldConfig.from_dict(data["world"])
        config.num_agents = 3
        scenario = apply_scenario(config, scenario_key)
        if args.communication is not None:
            config.comm.enabled = args.communication
        world = RoverWorld(config)
        world.reset(seed=args.seed)
        return world, MissionRuntime(world, scenario)

    world, runtime = build_world(args.scenario)
    scenario = runtime.scenario
    policy = _policy(args.policy, args.model, args.seed, args.stochastic)
    if hasattr(policy, "set_seed"):
        policy.set_seed(args.seed)
    events = [f"OPS  {scenario.name}", f"MAP  graine {args.seed}", f"AI   {policy.label}"]
    selected, view_mode, fps, paused, running = 0, "team", max(1, args.fps), False, True
    clock, last_step, phase = pygame.time.Clock(), 0.0, 0.0
    step_requested, debrief, report_path = False, False, None
    last_lost, low_energy_warned, offline_warned = 0, set(), set()
    target_steps = args.steps if args.screenshot else None
    while running:
        if not args.screenshot:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_SPACE and not debrief: paused = not paused
                    elif event.key == pygame.K_n and not debrief: step_requested = True; paused = True
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): fps = min(30, fps + 1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS): fps = max(1, fps - 1)
                    elif event.key == pygame.K_TAB: selected = (selected + 1) % len(world.agent_ids)
                    elif event.key == pygame.K_g:
                        view_mode = {"team": "selected", "selected": "global", "global": "team"}[view_mode]
                    elif event.key == pygame.K_r:
                        args.seed += 1
                        world, runtime = build_world(scenario.key)
                        scenario = runtime.scenario
                        if hasattr(policy, "set_seed"):
                            policy.set_seed(args.seed)
                        paused, debrief, report_path = False, False, None
                        last_lost, low_energy_warned, offline_warned = 0, set(), set()
                        events = [f"OPS  {scenario.name}", f"MAP  nouvelle graine {args.seed}", f"AI   {policy.label}"]
                    elif event.key == pygame.K_b:
                        chosen = _briefing_loop(pygame, screen, scenario.key)
                        if chosen is None:
                            running = False
                        else:
                            if not seed_overridden:
                                args.seed = DEFAULT_SCENARIO_SEEDS[chosen]
                            world, runtime = build_world(chosen)
                            scenario = runtime.scenario
                            if hasattr(policy, "set_seed"):
                                policy.set_seed(args.seed)
                            paused, debrief, report_path = False, False, None
                            last_lost, low_energy_warned, offline_warned = 0, set(), set()
                            events = [f"OPS  {scenario.name}", f"MAP  graine {args.seed}", f"AI   {policy.label}"]
                    elif event.key == pygame.K_F12:
                        path = ROOT / "artifacts" / "results" / "mission_control_v2.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pygame.image.save(screen, path)
                        events.append(f"IMG  {path.name}")
        now = time.monotonic()
        should_step = not debrief and (target_steps is not None or step_requested or (not paused and now - last_step >= 1.0 / fps))
        if should_step and (target_steps is None or world.step_count < target_steps):
            result: StepResult = world.step(policy.actions(world))
            step_requested = False
            last_step = now
            collisions = sum(bool(result.infos[aid]["collision"]) for aid in world.agent_ids)
            new_cells = int(result.infos[world.agent_ids[0]]["new_accessible_cells"])
            if collisions: events.append(f"WARN {collisions} tentative(s) bloquée(s) au sol {world.step_count:03d}")
            elif new_cells: events.append(f"SCAN +{new_cells:02d} cellules au sol {world.step_count:03d}")
            lost = int(world.metrics["messages_lost"])
            if lost > last_lost:
                events.append(f"COM  {lost - last_lost} paquet(s) perdu(s) au sol {world.step_count:03d}")
            last_lost = lost
            for index, aid in enumerate(world.agent_ids):
                if world.active[aid] and world.energy[aid] <= world.config.initial_energy * 0.25 and aid not in low_energy_warned:
                    events.append(f"PWR  R{index} réserve sous 25 %")
                    low_energy_warned.add(aid)
                if not world.active[aid] and aid not in offline_warned:
                    events.append(f"PWR  R{index} hors ligne")
                    offline_warned.add(aid)
            events.extend(runtime.update(world))
            if runtime.finished(world):
                paused, debrief = True, True
                target_steps = world.step_count if args.screenshot else None
                events.append("END  objectifs atteints" if runtime.success(world) else "END  mission terminée")
                report_path = _save_report(world, runtime, args.seed)
        phase += 0.06
        render_frame(pygame, screen, world, policy, selected, view_mode, fps, paused, events, phase, runtime, debrief, report_path)
        if args.screenshot and (world.step_count >= args.steps or paused):
            output = Path(args.screenshot)
            output.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, output)
            print(f"Capture Mission Control: {output.resolve()}")
            running = False
        elif not args.screenshot:
            pygame.display.flip(); clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
