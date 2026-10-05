from pathlib import Path

from mjlab.scripts.gail_ablation import (
  AblationConfig,
  LearningPoint,
  build_train_command,
  choose_num_envs,
  first_learning_target,
)


def test_ablation_commands_only_differ_in_gail_settings_and_run_name():
  cfg = AblationConfig(
    dataset_path=Path("expert.pt"), num_envs=(512,), iterations=1000, seed=7
  )

  ppo = build_train_command(cfg, num_envs=512, use_gail=False)
  gail = build_train_command(cfg, num_envs=512, use_gail=True)

  ignored = {
    "--agent.run-name",
    "--agent.gail.enabled",
    "--agent.gail.dataset-path",
  }

  def common_args(command: list[str]) -> list[str]:
    result = command[:4]
    for index in range(4, len(command), 2):
      if command[index] not in ignored:
        result.extend(command[index : index + 2])
    return result

  assert common_args(ppo) == common_args(gail)
  assert "--agent.gail.enabled" in ppo
  assert ppo[ppo.index("--agent.gail.enabled") + 1] == "False"
  assert gail[gail.index("--agent.gail.enabled") + 1] == "True"


def test_choose_num_envs_returns_measurement_closest_to_target():
  measurements = {256: 51.0, 512: 76.0, 768: 91.0}

  assert choose_num_envs(measurements, target=80.0) == 512


def test_first_learning_target_reports_iteration_transitions_and_wall_time():
  points = [
    LearningPoint(0, 0.2, 100.0, 10.0),
    LearningPoint(100, 0.85, 700.0, 30.0),
    LearningPoint(200, 0.9, 920.0, 55.0),
  ]

  result = first_learning_target(points, num_envs=512, num_steps_per_env=24)

  assert result == {"iteration": 200, "transitions": 2_469_888, "wall_time_s": 45.0}
