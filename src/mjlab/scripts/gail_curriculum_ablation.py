"""Run PPO versus PPO+GAIL branches at velocity curriculum boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

from mjlab.rl import VELOCITY_GAIL_FEATURE_SCHEMA
from mjlab.scripts.gail_ablation import RunResult, _run_and_measure


@dataclass(frozen=True)
class CurriculumAblationConfig:
  stage2_checkpoint: Path
  stage2_dataset: Path
  stage3_checkpoint: Path
  stage3_dataset: Path
  stage1_report: Path | None = Path("report/gail_ablation_20260924_034726.json")
  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  num_envs: int = 16_384
  seed: int = 42
  post_iterations: int = 500
  stage2_warmup_iterations: int = 50
  stage3_warmup_iterations: int = 50
  sample_interval_s: float = 1.0
  output_dir: Path = Path("report")
  log_root: Path = Path("logs/rsl_rl/g1_velocity")


@dataclass(frozen=True)
class StageConfig:
  name: str
  checkpoint_path: Path
  dataset_path: Path
  curriculum_step: int
  warmup_iterations: int
  post_iterations: int
  num_steps_per_env: int
  lin_vel_x: tuple[float, float]
  ang_vel_z: tuple[float, float]


@dataclass(frozen=True)
class ScalarPoint:
  step: int
  value: float
  wall_time: float


def build_stage_train_command(
  cfg: CurriculumAblationConfig,
  stage: StageConfig,
  use_gail: bool,
  run_name: str,
) -> list[str]:
  command = [
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    cfg.task,
    "--checkpoint-file",
    str(stage.checkpoint_path.resolve()),
    "--env.scene.num-envs",
    str(cfg.num_envs),
    "--agent.seed",
    str(cfg.seed),
    "--agent.max-iterations",
    str(stage.warmup_iterations + stage.post_iterations),
    "--agent.num-steps-per-env",
    str(stage.num_steps_per_env),
    "--agent.logger",
    "tensorboard",
    "--agent.run-name",
    run_name,
    "--agent.gail.enabled",
    str(use_gail),
  ]
  if use_gail:
    command.extend(("--agent.gail.dataset-path", str(stage.dataset_path.resolve())))
  return command


def summarize_post_transition(
  linear: list[ScalarPoint],
  episode_length: list[ScalarPoint],
  *,
  transition_iteration: int,
  num_envs: int,
  num_steps_per_env: int,
  auc_windows: tuple[int, ...] = (100, 300, 500),
  rolling_window: int = 25,
  thresholds: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4),
  max_post_iterations: int | None = None,
) -> dict[str, object]:
  post = [point for point in linear if point.step >= transition_iteration]
  if max_post_iterations is not None:
    post = post[:max_post_iterations]
  episodes = {
    point.step: point.value
    for point in episode_length
    if point.step >= transition_iteration
  }
  if not post:
    raise ValueError("No task metrics were logged after the curriculum transition")
  auc: dict[str, object] = {}
  for window in auc_windows:
    if len(post) < window:
      auc[str(window)] = None
      continue
    values = [point.value for point in post[:window]]
    auc[str(window)] = {"sum": sum(values), "mean": sum(values) / window}

  threshold_results: dict[str, object] = {}
  for threshold in thresholds:
    hit = None
    for index in range(rolling_window - 1, len(post)):
      window = post[index - rolling_window + 1 : index + 1]
      reward_mean = sum(point.value for point in window) / rolling_window
      episode_values = [episodes.get(point.step, 0.0) for point in window]
      episode_mean = sum(episode_values) / rolling_window
      if reward_mean >= threshold and episode_mean >= 900.0:
        relative_iteration = index
        hit = {
          "relative_iteration": relative_iteration,
          "transitions": (relative_iteration + 1) * num_envs * num_steps_per_env,
          "wall_time_s": post[index].wall_time - post[0].wall_time,
        }
        break
    threshold_results[str(threshold)] = hit

  return {
    "transition_iteration": transition_iteration,
    "post_transition_points": len(post),
    "auc": auc,
    "thresholds": threshold_results,
  }


def _stage_configs(cfg: CurriculumAblationConfig) -> tuple[StageConfig, StageConfig]:
  return (
    StageConfig(
      name="stage2",
      checkpoint_path=cfg.stage2_checkpoint,
      dataset_path=cfg.stage2_dataset,
      curriculum_step=120_000,
      warmup_iterations=cfg.stage2_warmup_iterations,
      post_iterations=cfg.post_iterations,
      num_steps_per_env=24,
      lin_vel_x=(-1.5, 2.0),
      ang_vel_z=(-0.7, 0.7),
    ),
    StageConfig(
      name="stage3",
      checkpoint_path=cfg.stage3_checkpoint,
      dataset_path=cfg.stage3_dataset,
      curriculum_step=240_000,
      warmup_iterations=cfg.stage3_warmup_iterations,
      post_iterations=cfg.post_iterations,
      num_steps_per_env=12,
      lin_vel_x=(-2.0, 3.0),
      ang_vel_z=(-0.7, 0.7),
    ),
  )


def _validate_stage(stage: StageConfig) -> None:
  if not stage.checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {stage.checkpoint_path}")
  if not stage.dataset_path.is_file():
    raise FileNotFoundError(f"Expert dataset not found: {stage.dataset_path}")
  loaded = torch.load(stage.dataset_path, map_location="cpu", weights_only=True)
  if not isinstance(loaded, dict):
    raise ValueError(f"Dataset must contain a dictionary: {stage.dataset_path}")
  expected_dims = {"observations": 68, "actions": 29, "commands": 3}
  transition_fields = {
    "dones",
    "terminated",
    "environment_ids",
    "episode_ids",
  }
  for name, width in expected_dims.items():
    value = loaded.get(name)
    if (
      not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width
    ):
      raise ValueError(f"{stage.name} dataset requires {name} with width {width}")
  missing = transition_fields - loaded.keys()
  if missing:
    raise ValueError(f"{stage.name} dataset is missing fields: {sorted(missing)}")
  tensors = {
    name: value
    for name, value in loaded.items()
    if name in expected_dims or name in transition_fields
  }
  if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
    raise ValueError(f"{stage.name} dataset fields must be tensors")
  lengths = {len(value) for value in tensors.values()}
  if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
    raise ValueError(f"{stage.name} dataset fields must have one non-zero length")
  if any(not torch.isfinite(value).all() for value in tensors.values()):
    raise ValueError(f"{stage.name} dataset contains NaN or Inf")
  metadata = loaded.get("metadata")
  if not isinstance(metadata, dict):
    raise ValueError(f"{stage.name} dataset is missing metadata")
  if metadata.get("feature_schema") != VELOCITY_GAIL_FEATURE_SCHEMA:
    raise ValueError(
      f"{stage.name} dataset must use feature_schema '{VELOCITY_GAIL_FEATURE_SCHEMA}'"
    )
  if metadata.get("schema_version") != 2:
    raise ValueError(f"{stage.name} dataset must use schema_version 2")
  if metadata.get("num_transitions") != next(iter(lengths)):
    raise ValueError(f"{stage.name} dataset has inconsistent num_transitions")
  if not metadata.get("checkpoint_sha256"):
    raise ValueError(f"{stage.name} dataset is missing checkpoint_sha256")
  if metadata.get("curriculum_start_step") != stage.curriculum_step:
    raise ValueError(f"{stage.name} dataset has the wrong curriculum_start_step")
  ranges = metadata.get("effective_command_ranges")
  if not isinstance(ranges, dict):
    raise ValueError(f"{stage.name} dataset is missing effective command ranges")
  if tuple(ranges.get("lin_vel_x", ())) != stage.lin_vel_x:
    raise ValueError(f"{stage.name} dataset has the wrong lin_vel_x range")
  if tuple(ranges.get("ang_vel_z", ())) != stage.ang_vel_z:
    raise ValueError(f"{stage.name} dataset has the wrong ang_vel_z range")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _scalars(events, tag: str) -> list[ScalarPoint]:
  if tag not in events.Tags().get("scalars", []):
    return []
  return [
    ScalarPoint(item.step, item.value, item.wall_time) for item in events.Scalars(tag)
  ]


def _find_transition(events, stage: StageConfig) -> int:
  minimum = _scalars(events, "Curriculum/command_vel/lin_vel_x_min")
  maximum = {
    point.step: point
    for point in _scalars(events, "Curriculum/command_vel/lin_vel_x_max")
  }
  for point in minimum:
    other = maximum.get(point.step)
    if (
      other is not None
      and math.isclose(point.value, stage.lin_vel_x[0], abs_tol=1e-5)
      and math.isclose(other.value, stage.lin_vel_x[1], abs_tol=1e-5)
    ):
      return point.step
  raise RuntimeError(f"Could not find the {stage.name} curriculum transition")


def _ranges_at_iteration(events, iteration: int) -> dict[str, tuple[float, float]]:
  ranges = {}
  for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    values = []
    for bound in ("min", "max"):
      tag = f"Curriculum/command_vel/{name}_{bound}"
      point = next((p for p in _scalars(events, tag) if p.step == iteration), None)
      if point is None:
        raise RuntimeError(f"Missing {tag} at curriculum iteration {iteration}")
      values.append(point.value)
    ranges[name] = (values[0], values[1])
  return ranges


def _metric_summary(
  events, tag: str, transition: int, limit: int
) -> dict[str, float] | None:
  points = [point for point in _scalars(events, tag) if point.step >= transition]
  points = points[:limit]
  if not points:
    return None
  values = [point.value for point in points]
  final_values = values[-100:]
  return {
    "mean": sum(values) / len(values),
    "final": values[-1],
    "last100_mean": sum(final_values) / len(final_values),
  }


def _analyze_run(result: RunResult, stage: StageConfig) -> dict[str, object]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  events = EventAccumulator(result.log_dir, size_guidance={"scalars": 0}).Reload()
  transition = _find_transition(events, stage)
  counter = next(
    (
      point.value
      for point in _scalars(events, "Curriculum/command_vel/common_step_counter")
      if point.step == transition
    ),
    None,
  )
  if counter is None:
    raise RuntimeError(f"Missing curriculum counter at iteration {transition}")
  learning = summarize_post_transition(
    _scalars(events, "Episode_Reward/track_linear_velocity"),
    _scalars(events, "Train/mean_episode_length"),
    transition_iteration=transition,
    num_envs=result.num_envs,
    num_steps_per_env=stage.num_steps_per_env,
    max_post_iterations=stage.post_iterations,
  )
  metric_tags = {
    "linear_velocity": "Episode_Reward/track_linear_velocity",
    "angular_velocity": "Episode_Reward/track_angular_velocity",
    "episode_length": "Train/mean_episode_length",
    "upright": "Episode_Reward/upright",
    "foot_slip": "Episode_Reward/foot_slip",
    "action_rate": "Episode_Reward/action_rate_l2",
    "discriminator_loss": "GAIL/discriminator_loss",
    "discriminator_accuracy": "GAIL/discriminator_accuracy",
    "weighted_gail_reward": "GAIL/weighted_reward_mean",
  }
  return {
    **asdict(result),
    "transition_counter": int(counter),
    "effective_command_ranges": _ranges_at_iteration(events, transition),
    "learning": learning,
    "metrics": {
      name: summary
      for name, tag in metric_tags.items()
      if (summary := _metric_summary(events, tag, transition, stage.post_iterations))
      is not None
    },
  }


def run_curriculum_ablation(cfg: CurriculumAblationConfig) -> None:
  if cfg.num_envs < 1 or cfg.post_iterations < 1:
    raise ValueError("num_envs and post_iterations must be positive")
  if cfg.stage2_warmup_iterations < 0 or cfg.stage3_warmup_iterations < 0:
    raise ValueError("warmup iterations must be non-negative")
  stages = _stage_configs(cfg)
  for stage in stages:
    _validate_stage(stage)
  stage1 = None
  if cfg.stage1_report is not None:
    if not cfg.stage1_report.is_file():
      raise FileNotFoundError(f"Stage 1 report not found: {cfg.stage1_report}")
    stage1 = json.loads(cfg.stage1_report.read_text(encoding="utf-8"))

  stage_runs: dict[str, list[tuple[StageConfig, RunResult]]] = {}
  for stage in stages:
    order = (False, True) if stage.name == "stage2" else (True, False)
    stage_runs[stage.name] = []
    for use_gail in order:
      arm = "gail" if use_gail else "ppo"
      name = f"curriculum-{stage.name}-seed{cfg.seed}-{cfg.num_envs}env-{arm}"
      iterations = stage.warmup_iterations + stage.post_iterations
      result = _run_and_measure(
        build_stage_train_command(cfg, stage, use_gail, name),
        name,
        cfg.num_envs,
        iterations,
        cfg.sample_interval_s,
        cfg.log_root,
      )
      stage_runs[stage.name].append((stage, result))

  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  output = cfg.output_dir / f"gail_curriculum_{datetime.now():%Y%m%d_%H%M%S}.json"
  output.write_text(
    json.dumps(
      {
        "config": {
          **asdict(cfg),
          "stage2_checkpoint": str(cfg.stage2_checkpoint),
          "stage2_dataset": str(cfg.stage2_dataset),
          "stage3_checkpoint": str(cfg.stage3_checkpoint),
          "stage3_dataset": str(cfg.stage3_dataset),
          "output_dir": str(cfg.output_dir),
          "log_root": str(cfg.log_root),
        },
        "stage1_existing_ablation": stage1,
        "stages": {
          name: {
            "stage": {
              **asdict(entries[0][0]),
              "checkpoint_path": str(entries[0][0].checkpoint_path),
              "dataset_path": str(entries[0][0].dataset_path),
              "checkpoint_sha256": _sha256(entries[0][0].checkpoint_path),
              "dataset_sha256": _sha256(entries[0][0].dataset_path),
            },
            "runs": [_analyze_run(result, stage) for stage, result in entries],
          }
          for name, entries in stage_runs.items()
        },
      },
      indent=2,
      default=str,
    ),
    encoding="utf-8",
  )
  print(f"[INFO] Curriculum ablation report: {output}")


def main() -> None:
  run_curriculum_ablation(tyro.cli(CurriculumAblationConfig))


if __name__ == "__main__":
  main()
