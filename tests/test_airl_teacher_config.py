from dataclasses import asdict

import mjlab.tasks  # noqa: F401
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_teacher_command_and_reward_contract():
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-AIRL-Teacher")
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (1.2, 1.2)
  assert command.ranges.lin_vel_y == (0.0, 0.0)
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert not command.heading_command
  assert command.ranges.heading is None
  assert not command.target_speed_command_grid
  assert not command.target_speed_curriculum
  assert command.rel_standing_envs == command.rel_heading_envs == 0
  assert cfg.curriculum == {}
  assert asdict(cfg)["rewards"] == asdict(unitree_g1_flat_env_cfg())["rewards"]
  agent = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1-AIRL-Teacher")
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  assert not agent.gail.enabled
  assert not agent.airl.enabled


def test_airl_mvp_uses_same_fixed_command_environment():
  teacher = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-AIRL-Teacher")
  mvp = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-AIRL-MVP")
  for field in ("commands", "curriculum", "rewards", "observations", "terminations"):
    assert asdict(mvp)[field] == asdict(teacher)[field]
