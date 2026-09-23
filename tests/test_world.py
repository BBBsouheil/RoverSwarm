import numpy as np

from roverswarm.config import CommConfig, WorldConfig
from roverswarm.world import ACTION_DELTAS, FREE, OBSTACLE, UNKNOWN, RoverWorld


def cfg(n=3, **kwargs):
    return WorldConfig(width=9, height=9, num_agents=n, sensor_radius=1, max_steps=20, **kwargs)


def test_seed_reproducibility_and_valid_starts():
    a, b = RoverWorld(cfg()), RoverWorld(cfg())
    oa, ob = a.reset(42), b.reset(42)
    assert np.array_equal(a.grid, b.grid)
    assert a.positions == b.positions
    assert all(np.array_equal(oa[key], ob[key]) for key in oa)
    assert all(a.grid[pos] == FREE and a.accessible[pos] for pos in a.positions.values())


def test_same_target_and_swap_are_blocked_independent_of_action_order():
    world = RoverWorld(cfg(n=2), map_name="open7"); world.reset(1)
    world.set_positions_for_test([(3, 2), (3, 4)])
    result = world.step({"rover_1": 2, "rover_0": 3})
    assert world.positions == {"rover_0": (3, 2), "rover_1": (3, 4)}
    assert result.infos["rover_0"]["collision"] and result.infos["rover_1"]["collision"]
    world.reset(1); world.set_positions_for_test([(3, 2), (3, 3)])
    world.step({"rover_0": 3, "rover_1": 2})
    assert world.positions == {"rover_0": (3, 2), "rover_1": (3, 3)}


def test_wall_collision_energy_and_inactive_slot():
    world = RoverWorld(cfg(n=1, initial_energy=1.0), map_name="rooms9"); world.reset(2)
    world.set_positions_for_test([(0, 0)])
    result = world.step({"rover_0": 0})
    assert result.infos["rover_0"]["collision"]
    assert not world.active["rover_0"] and world.energy["rover_0"] == 0
    assert result.terminations["rover_0"] and not result.truncations["rover_0"]


def test_unobserved_change_does_not_change_observation():
    world = RoverWorld(cfg(n=1), map_name="open7"); world.reset(3)
    world.set_positions_for_test([(1, 1)])
    before = world.observation("rover_0").copy()
    unknown_cells = np.argwhere(world.memories["rover_0"] == UNKNOWN)
    assert len(unknown_cells) > 0
    y, x = map(int, unknown_cells[0])
    world.grid[y, x] = OBSTACLE
    after = world.observation("rover_0")
    assert np.array_equal(before, after)


def test_novelty_reward_counts_team_union_once():
    world = RoverWorld(cfg(n=2), map_name="open7"); world.reset(4)
    world.set_positions_for_test([(3, 2), (3, 4)])
    before = world.team_known_accessible.copy()
    result = world.step({"rover_0": 4, "rover_1": 4})
    reported = result.infos["rover_0"]["new_accessible_cells"]
    actual = int((world.team_known_accessible & ~before).sum())
    assert reported == actual
    second = world.step({"rover_0": 4, "rover_1": 4})
    assert second.infos["rover_0"]["new_accessible_cells"] == 0


def test_time_limit_is_truncation_not_success():
    world = RoverWorld(WorldConfig(width=9, height=9, num_agents=1, sensor_radius=0, max_steps=1, target_coverage=1.0), map_name="open7")
    world.reset(5)
    result = world.step({"rover_0": 4})
    assert not result.terminations["rover_0"]
    assert result.truncations["rover_0"]
    assert result.infos["rover_0"]["termination_reason"] is None


def test_communication_delivery_and_total_loss_are_reproducible():
    delivered_cfg = cfg(n=2)
    delivered_cfg.comm = CommConfig(enabled=True, range=99, frequency=1, capacity=49, delay=0, loss_probability=0.0)
    delivered = RoverWorld(delivered_cfg, map_name="open7"); delivered.reset(7)
    delivered.set_positions_for_test([(1, 1), (5, 5)])
    delivered.step({"rover_0": 4, "rover_1": 4})
    assert delivered.metrics["messages_received"] == 2
    assert delivered.metrics["communication_bytes"] > 0
    assert delivered.last_sender["rover_1"] == (1, 1)
    lost_cfg = cfg(n=2)
    lost_cfg.comm = CommConfig(enabled=True, range=99, frequency=1, capacity=4, delay=0, loss_probability=1.0)
    lost = RoverWorld(lost_cfg, map_name="open7"); lost.reset(7)
    lost.step({"rover_0": 4, "rover_1": 4})
    assert lost.metrics["messages_lost"] == 2 and lost.metrics["messages_received"] == 0


def test_radio_receiver_does_not_steal_local_discovery_credit():
    config = cfg(n=2)
    config.observation_version = 2
    config.novelty_reward = 0.0
    config.individual_novelty_reward = 1.0
    config.wait_penalty = 0.0
    config.time_penalty = 0.0
    config.comm = CommConfig(enabled=True, range=99, frequency=1, capacity=49, delay=0, loss_probability=0.0)
    world = RoverWorld(config, map_name="open7")
    world.reset(17)
    world.set_positions_for_test([(1, 1), (5, 5)])

    result = world.step({"rover_0": 3, "rover_1": 4})

    assert result.infos["rover_0"]["credited_new_cells"] > 0
    assert result.infos["rover_1"]["credited_new_cells"] == 0
    assert result.infos["rover_1"]["communication"]["received"] > 0


def test_v3_observation_marks_radio_only_cells_without_map_leakage():
    config = cfg(n=2)
    config.observation_version = 3
    config.comm = CommConfig(enabled=True, range=99, frequency=1, capacity=49, delay=0, loss_probability=0.0)
    world = RoverWorld(config, map_name="open7")
    observations = world.reset(23)
    assert observations["rover_0"].shape == (6 * 49 + 9,)
    world.set_positions_for_test([(1, 1), (5, 5)])

    result = world.step({"rover_0": 3, "rover_1": 4})
    received_only = (world.memories["rover_1"] != UNKNOWN) & ~world.local_known["rover_1"]
    channel = result.observations["rover_1"][5 * 49:6 * 49].reshape(7, 7)

    assert received_only.any()
    assert np.array_equal(channel, received_only.astype(np.float32))
    assert not np.any(channel[world.memories["rover_1"] == UNKNOWN])


def test_v2_observation_and_reward_credit_are_local_and_bounded():
    config = cfg(n=2)
    config.observation_version = 2
    config.novelty_reward = 0.3
    config.individual_novelty_reward = 0.7
    config.collision_penalty = 1.2
    config.revisit_penalty = 0.08
    world = RoverWorld(config, map_name="open7")
    observations = world.reset(12)
    assert observations["rover_0"].shape == (5 * 49 + 9,)
    assert observations["rover_0"].dtype == np.float32
    result = world.step({"rover_0": 4, "rover_1": 4})
    credited = sum(float(result.infos[aid]["credited_new_cells"]) for aid in world.agent_ids)
    assert credited <= float(result.infos["rover_0"]["new_accessible_cells"])
    assert all(np.isfinite(value) for value in result.rewards.values())


def test_action_mask_uses_only_known_obstacles():
    config = cfg(n=1)
    config.observation_version = 2
    world = RoverWorld(config, map_name="rooms9")
    world.reset(2)
    world.set_positions_for_test([(1, 4)])
    mask = world.action_mask("rover_0")
    assert mask.shape == (5,) and mask.dtype == bool and mask[4]
    for action in range(4):
        dy, dx = ACTION_DELTAS[action]
        y, x = world.positions["rover_0"]
        target = (y + dy, x + dx)
        if world.memories["rover_0"][target] == OBSTACLE:
            assert not mask[action]
