import pytest

from roverswarm.config import WorldConfig
from roverswarm.mission_control import DEFAULT_MODEL
from roverswarm.missions import SCENARIOS, MissionRuntime, apply_scenario, mission_report
from roverswarm.world import RoverWorld


def test_mission_scenarios_are_distinct_and_valid():
    survey = WorldConfig()
    blackout = WorldConfig()
    endurance = WorldConfig()

    apply_scenario(survey, "survey")
    apply_scenario(blackout, "blackout")
    apply_scenario(endurance, "endurance")

    assert set(SCENARIOS) == {"survey", "blackout", "endurance"}
    assert survey.comm.enabled
    assert blackout.comm.loss_probability > survey.comm.loss_probability
    assert blackout.comm.delay > survey.comm.delay
    assert endurance.initial_energy < survey.initial_energy
    assert len({scenario.target_coverage for scenario in SCENARIOS.values()}) == 3
    assert blackout.target_coverage == endurance.target_coverage == 1.0


def test_final_game_model_is_packaged():
    assert DEFAULT_MODEL.is_file()
    assert DEFAULT_MODEL.suffix == ".zip"


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Scénario inconnu"):
        apply_scenario(WorldConfig(), "missing")


def test_mission_report_contains_only_measured_state():
    config = WorldConfig(width=9, height=9, num_agents=3, max_steps=5)
    scenario = apply_scenario(config, "survey")
    world = RoverWorld(config, map_name="open7")
    world.reset(seed=41)
    runtime = MissionRuntime(world, scenario)
    world.step({aid: 4 for aid in world.agent_ids})

    report = mission_report(world, runtime, 41)

    assert report["seed"] == 41
    assert report["world_steps"] == world.step_count
    assert report["coverage"] == world.coverage
    assert report["metrics"]["collisions"] == world.metrics["collisions"]
    assert set(report["remaining_energy"]) == set(world.agent_ids)


def test_relay_activation_restores_radio_progressively():
    config = WorldConfig(width=9, height=9, num_agents=3, max_steps=20)
    scenario = apply_scenario(config, "blackout")
    world = RoverWorld(config, map_name="open7")
    world.reset(seed=51)
    runtime = MissionRuntime(world, scenario)
    initial_loss = world.config.comm.loss_probability

    site = runtime.sites[0]
    world.set_positions_for_test([site, world.positions["rover_1"], world.positions["rover_2"]])
    events = runtime.update(world)

    assert runtime.objective_done >= 1
    assert world.config.comm.loss_probability < initial_loss
    assert any("relais rétabli" in event for event in events)


def test_sample_mission_requires_sites_coverage_and_survivor():
    config = WorldConfig(width=9, height=9, num_agents=3, max_steps=20)
    scenario = apply_scenario(config, "endurance")
    world = RoverWorld(config, map_name="open7")
    world.reset(seed=61)
    runtime = MissionRuntime(world, scenario)

    runtime.completed.update(range(runtime.objective_total))
    world.team_known_accessible[:] = False
    assert not runtime.success(world)
    world.team_known_accessible[:] = world.accessible
    assert runtime.success(world)
    world.active = {aid: False for aid in world.agent_ids}
    assert not runtime.success(world)
