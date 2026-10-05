from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_coverage_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import GailVelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def test_g1_ablation_defaults_match_legacy_ppo_baseline():
  agent = unitree_g1_ppo_runner_cfg()
  env = make_velocity_env_cfg()

  assert agent.gail.enabled is False
  assert agent.algorithm.entropy_coef == 0.01
  assert agent.algorithm.num_learning_epochs == 5
  assert agent.algorithm.num_mini_batches == 4
  assert agent.algorithm.learning_rate == 1.0e-3
  assert agent.algorithm.desired_kl == 0.01
  assert env.rewards["pose"].weight == 1.0
  assert env.rewards["action_rate_l2"].weight == -0.1


def test_g1_ablation_uses_slow_velocity_curriculum():
  env = make_velocity_env_cfg()
  stages = env.curriculum["command_vel"].params["velocity_stages"]

  assert [stage["step"] for stage in stages] == [0, 5000 * 24, 10000 * 24]


def test_coverage_task_opt_in_changes_only_frontier_sampler():
  coverage = unitree_g1_flat_coverage_env_cfg()
  stages = coverage.curriculum["command_vel"].params["velocity_stages"]
  assert stages[-1]["frontier_velocity_prob"] == 0.5


def test_g1_tasks_use_opt_in_gail_runner():
  assert load_runner_cls("Mjlab-Velocity-Flat-Unitree-G1") is GailVelocityOnPolicyRunner
  assert (
    load_runner_cls("Mjlab-Velocity-Rough-Unitree-G1") is GailVelocityOnPolicyRunner
  )
