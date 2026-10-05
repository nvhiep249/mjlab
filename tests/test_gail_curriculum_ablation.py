from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from mjlab.rl import VELOCITY_GAIL_FEATURE_SCHEMA
from mjlab.scripts.gail_curriculum_ablation import (
  CurriculumAblationConfig,
  ScalarPoint,
  StageConfig,
  _validate_stage,
  build_stage_train_command,
  summarize_post_transition,
)


def _stage() -> StageConfig:
  return StageConfig(
    name="stage2",
    checkpoint_path=Path("model_4950.pt"),
    dataset_path=Path("stage2.pt"),
    curriculum_step=120_000,
    warmup_iterations=50,
    post_iterations=500,
    num_steps_per_env=24,
    lin_vel_x=(-1.5, 2.0),
    ang_vel_z=(-0.7, 0.7),
  )


def _schema_v2_data(curriculum_step: int = 120_000) -> dict[str, object]:
  count = 4
  return {
    "observations": torch.zeros(count, 68),
    "actions": torch.zeros(count, 29),
    "commands": torch.zeros(count, 3),
    "dones": torch.zeros(count, dtype=torch.bool),
    "terminated": torch.zeros(count, dtype=torch.bool),
    "environment_ids": torch.arange(count),
    "episode_ids": torch.zeros(count, dtype=torch.long),
    "metadata": {
      "feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA,
      "schema_version": 2,
      "num_transitions": count,
      "checkpoint_sha256": "abc123",
      "curriculum_start_step": curriculum_step,
      "effective_command_ranges": {
        "lin_vel_x": (-1.5, 2.0),
        "ang_vel_z": (-0.7, 0.7),
      },
    },
  }


def test_stage_commands_only_differ_in_gail_settings_and_run_name():
  cfg = CurriculumAblationConfig(
    stage2_checkpoint=Path("model_4950.pt"),
    stage2_dataset=Path("stage2.pt"),
    stage3_checkpoint=Path("model_14550.pt"),
    stage3_dataset=Path("stage3.pt"),
  )

  ppo = build_stage_train_command(cfg, _stage(), False, "stage2-ppo")
  gail = build_stage_train_command(cfg, _stage(), True, "stage2-gail")

  ignored = {"--agent.run-name", "--agent.gail.enabled", "--agent.gail.dataset-path"}

  def common_args(command: list[str]) -> list[str]:
    result = command[:4]
    for index in range(4, len(command), 2):
      if command[index] not in ignored:
        result.extend(command[index : index + 2])
    return result

  assert common_args(ppo) == common_args(gail)
  assert ppo[ppo.index("--checkpoint-file") + 1].endswith("model_4950.pt")
  assert ppo[ppo.index("--agent.num-steps-per-env") + 1] == "24"
  assert ppo[ppo.index("--agent.max-iterations") + 1] == "550"


def test_summary_uses_only_post_transition_points_and_real_transition_count():
  linear = [
    ScalarPoint(step, float(step - 99), float(step)) for step in range(100, 130)
  ]
  episode = [ScalarPoint(step, 1000.0, float(step)) for step in range(100, 130)]

  summary = summarize_post_transition(
    linear,
    episode,
    transition_iteration=105,
    num_envs=10,
    num_steps_per_env=24,
    auc_windows=(10,),
    rolling_window=3,
    thresholds=(9.0,),
  )

  assert summary["post_transition_points"] == 25
  auc = cast(dict[str, object], summary["auc"])
  assert cast(dict[str, float], auc["10"])["sum"] == 105.0
  thresholds = cast(dict[str, object], summary["thresholds"])
  assert thresholds["9.0"] == {
    "relative_iteration": 4,
    "transitions": 1_200,
    "wall_time_s": 4.0,
  }


def test_stage_validation_rejects_legacy_world_frame_dataset(tmp_path):
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  dataset = tmp_path / "expert.pt"
  data = _schema_v2_data()
  metadata = cast(dict[str, object], data["metadata"])
  metadata.pop("feature_schema")
  torch.save(data, dataset)
  stage = StageConfig(
    name="stage2",
    checkpoint_path=checkpoint,
    dataset_path=dataset,
    curriculum_step=120_000,
    warmup_iterations=50,
    post_iterations=500,
    num_steps_per_env=24,
    lin_vel_x=(-1.5, 2.0),
    ang_vel_z=(-0.7, 0.7),
  )

  with pytest.raises(ValueError, match="feature_schema"):
    _validate_stage(stage)


def test_stage_validation_accepts_body_local_dataset(tmp_path):
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  dataset = tmp_path / "expert.pt"
  torch.save(_schema_v2_data(), dataset)
  stage = StageConfig(
    name="stage2",
    checkpoint_path=checkpoint,
    dataset_path=dataset,
    curriculum_step=120_000,
    warmup_iterations=50,
    post_iterations=500,
    num_steps_per_env=24,
    lin_vel_x=(-1.5, 2.0),
    ang_vel_z=(-0.7, 0.7),
  )

  _validate_stage(stage)


def test_stage_validation_rejects_inconsistent_transition_lengths(tmp_path):
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  dataset = tmp_path / "expert.pt"
  data = _schema_v2_data()
  data["episode_ids"] = torch.zeros(3, dtype=torch.long)
  torch.save(data, dataset)
  stage = _stage()
  stage = replace(stage, checkpoint_path=checkpoint, dataset_path=dataset)

  with pytest.raises(ValueError, match="non-zero length"):
    _validate_stage(stage)
