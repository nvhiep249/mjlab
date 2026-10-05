"""Tests for the velocity command's initial-velocity injection.

The init_velocity_prob path runs inside the reset pipeline, after reset
events wrote the new pose to qpos but before sim.forward(), so it must not
read (or write back) derived kinematics.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from conftest import get_test_device, load_fixture_xml, make_scene_and_sim

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
  sample_frontier_velocity_x,
  select_target_speed_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_frontier_sampler_assigns_explicit_mass_without_changing_yaw_config():
  torch.manual_seed(7)
  samples = sample_frontier_velocity_x(
    torch.empty(20_000), (-2.0, 3.0), 0.5, (-2.0, 2.0, 3.0)
  )

  frontier = torch.isin(samples, torch.tensor([-2.0, 2.0, 3.0]))
  assert 0.45 < float(frontier.float().mean()) < 0.55
  assert torch.all((samples >= -2.0) & (samples <= 3.0))


def test_target_speed_grid_resample_uses_only_qualified_commands():
  term = object.__new__(UniformVelocityCommand)
  term.cfg = cast(
    UniformVelocityCommandCfg, SimpleNamespace(target_speed_command_grid=True)
  )
  term._env = cast("ManagerBasedRlEnv", SimpleNamespace(device="cpu"))
  term.vel_command_b = torch.zeros(5, 3)
  term.vel_command_w = torch.zeros(5, 3)
  term.is_heading_env = torch.ones(5, dtype=torch.bool)
  term.is_standing_env = torch.ones(5, dtype=torch.bool)
  term.is_world_env = torch.ones(5, dtype=torch.bool)
  term.is_forward_env = torch.ones(5, dtype=torch.bool)

  term._resample_command(torch.arange(5))

  torch.testing.assert_close(
    term.vel_command_b,
    torch.tensor(
      [
        [1.2, 0.0, 0.0],
        [1.35, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.35, 0.0, 0.0],
      ]
    ),
  )
  assert not term.is_heading_env.any()
  assert not term.is_standing_env.any()
  assert not term.is_world_env.any()
  assert not term.is_forward_env.any()


def test_target_speed_curriculum_selects_stage_from_common_step_counter():
  stages = (
    (0, (0.3, 0.5, 0.8)),
    (4_800, (0.6, 0.9, 1.2)),
    (12_000, (1.0, 1.1, 1.2)),
  )

  assert select_target_speed_values(0, stages) == (0.3, 0.5, 0.8)
  assert select_target_speed_values(4_799, stages) == (0.3, 0.5, 0.8)
  assert select_target_speed_values(4_800, stages) == (0.6, 0.9, 1.2)
  assert select_target_speed_values(12_000, stages) == (1.0, 1.1, 1.2)


def _cpu_velocity_term():
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_b=torch.zeros(3, 3),
      root_link_ang_vel_b=torch.zeros(3, 3),
    )
  )
  env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.05, scene={"robot": robot})
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(0.05, 0.05),
    ranges=UniformVelocityCommandCfg.Ranges((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
  )
  return UniformVelocityCommand(cfg, cast("ManagerBasedRlEnv", env))


def test_pinned_commands_survive_timer_resampling_and_scoped_reset_cpu():
  term = _cpu_velocity_term()
  commands = torch.tensor([[1.2, 0.0, 0.0], [0.8, 0.1, -0.2], [0.5, -0.1, 0.2]])
  term.pin_commands(commands)
  commands.zero_()  # The pin owns its immutable copy.
  expected = term.command.clone()
  term.compute(dt=0.1)
  torch.testing.assert_close(term.command, expected)
  assert term.command_counter.tolist() == [1, 1, 1]
  term.reset(torch.tensor([1]))
  term.compute(dt=0.0, env_ids=torch.tensor([1]))
  torch.testing.assert_close(term.command, expected)
  torch.testing.assert_close(term.vel_command_w, expected)
  term.pin_commands(None)
  term.compute(dt=0.1)
  assert not term.command.any()


@pytest.mark.parametrize("commands", [torch.zeros(3, 2), torch.full((3, 3), torch.nan)])
def test_pinned_commands_reject_invalid_shape_and_nonfinite_cpu(commands):
  term = _cpu_velocity_term()
  with pytest.raises(ValueError, match="shape|finite"):
    term.pin_commands(commands)


def test_init_velocity_preserves_fresh_reset_pose(device):
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene=scene, sim=sim, num_envs=2, device=device),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    init_velocity_prob=1.0,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.5, 0.5), lin_vel_y=(0.2, 0.2), ang_vel_z=(0.0, 0.0)
    ),
  )
  term = cfg.build(env)
  robot = scene["robot"]
  env_ids = torch.arange(2, device=device)

  # Derived kinematics now hold the spawn pose (the "previous episode" state).
  sim.forward()

  # Emulate a reset event: write a fresh pose to qpos, no forward yet.
  pose = torch.tensor(
    [
      [1.0, 2.0, 1.5, 1.0, 0.0, 0.0, 0.0],
      [3.0, -1.0, 1.5, 1.0, 0.0, 0.0, 0.0],
    ],
    device=device,
  )
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)

  term.reset(env_ids=env_ids)
  sim.forward()

  # The fresh reset pose survives. Before the fix, the init-velocity path
  # wrote the stale pre-reset pose back into the sim.
  assert torch.allclose(robot.data.root_link_pos_w, pose[:, :3], atol=1e-6)

  # Planar velocity matches the sampled command (identity orientation, so
  # body frame equals world frame).
  assert torch.allclose(
    robot.data.root_link_lin_vel_b[:, :2],
    term.vel_command_b[:, :2],
    atol=1e-5,
  )


def test_mid_episode_resample_does_not_write_velocity(device):
  """Init velocity applies on reset only; a timer-expiry resample runs after
  step()'s forward and must not write sim state."""
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene=scene, sim=sim, num_envs=2, device=device, step_dt=0.02),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(0.001, 0.001),  # Expires on the first compute.
    init_velocity_prob=1.0,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.7, 0.7), lin_vel_y=(0.3, 0.3), ang_vel_z=(0.0, 0.0)
    ),
  )
  term = cfg.build(env)
  robot = scene["robot"]
  env_ids = torch.arange(2, device=device)

  term.reset(env_ids=env_ids)
  sim.forward()

  # Mid-episode the robot has decelerated to rest.
  robot.write_root_link_velocity_to_sim(
    torch.zeros(2, 6, device=device), env_ids=env_ids
  )
  sim.forward()

  # Step's command compute: the 1 ms timer expires and resamples.
  counter = term.command_counter.clone()
  term.compute(dt=1.0)
  assert (term.command_counter > counter).all()

  # Only the command changed; sim velocity is untouched.
  qvel_lin = sim.data.qvel[:, robot.indexing.free_joint_v_adr[:3]]
  assert torch.allclose(qvel_lin, torch.zeros_like(qvel_lin), atol=1e-6)
  assert torch.allclose(
    robot.data.root_link_lin_vel_b,
    torch.zeros_like(robot.data.root_link_lin_vel_b),
    atol=1e-6,
  )
