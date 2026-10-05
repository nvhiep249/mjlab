from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import (
  AirlVelocityOnPolicyRunner,
  GailVelocityOnPolicyRunner,
)

from .env_cfgs import (
  unitree_g1_flat_airl_teacher_env_cfg,
  unitree_g1_flat_coverage_env_cfg,
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=GailVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Coverage",
  env_cfg=unitree_g1_flat_coverage_env_cfg(),
  play_env_cfg=unitree_g1_flat_coverage_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=GailVelocityOnPolicyRunner,
)


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=GailVelocityOnPolicyRunner,
)

_airl_rl_cfg = unitree_g1_ppo_runner_cfg()
_airl_rl_cfg.experiment_name = "g1_velocity_airl"
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AIRL",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=_airl_rl_cfg,
  runner_cls=AirlVelocityOnPolicyRunner,
)

_airl_mvp_rl_cfg = unitree_g1_ppo_runner_cfg()
_airl_mvp_rl_cfg.experiment_name = "g1_velocity_airl_1p2"
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AIRL-MVP",
  env_cfg=unitree_g1_flat_airl_teacher_env_cfg(),
  play_env_cfg=unitree_g1_flat_airl_teacher_env_cfg(play=True),
  rl_cfg=_airl_mvp_rl_cfg,
  runner_cls=AirlVelocityOnPolicyRunner,
)

_teacher_rl_cfg = unitree_g1_ppo_runner_cfg()
_teacher_rl_cfg.experiment_name = "g1_airl_teacher_1p2"
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AIRL-Teacher",
  env_cfg=unitree_g1_flat_airl_teacher_env_cfg(),
  play_env_cfg=unitree_g1_flat_airl_teacher_env_cfg(play=True),
  rl_cfg=_teacher_rl_cfg,
  runner_cls=GailVelocityOnPolicyRunner,
)
