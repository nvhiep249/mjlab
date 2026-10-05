"""Run controlled PPO versus PPO+GAIL training experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import tyro


@dataclass(frozen=True)
class AblationConfig:
  dataset_path: Path
  """Expert transition dataset used only by the GAIL arm."""

  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  num_envs: tuple[int, ...] = (8192, 12288, 16384)
  calibration_iterations: int = 10
  calibrate_only: bool = False
  iterations: int = 1000
  seed: int = 42
  target_gpu_utilization: float = 80.0
  gpu_tolerance: float = 10.0
  sample_interval_s: float = 1.0
  output_dir: Path = Path("report")
  log_root: Path = Path("logs/rsl_rl/g1_velocity")


@dataclass(frozen=True)
class LearningPoint:
  iteration: int
  linear_velocity_reward: float
  episode_length: float
  wall_time: float


@dataclass(frozen=True)
class RunResult:
  name: str
  num_envs: int
  iterations: int
  duration_s: float
  mean_gpu_utilization: float
  samples: int
  log_dir: str


def build_train_command(
  cfg: AblationConfig,
  num_envs: int,
  use_gail: bool,
  *,
  iterations: int | None = None,
  run_name: str | None = None,
) -> list[str]:
  """Build one arm while keeping all non-GAIL arguments identical."""
  arm = "gail" if use_gail else "ppo"
  command = [
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    cfg.task,
    "--env.scene.num-envs",
    str(num_envs),
    "--agent.seed",
    str(cfg.seed),
    "--agent.max-iterations",
    str(iterations if iterations is not None else cfg.iterations),
    "--agent.logger",
    "tensorboard",
    "--agent.run-name",
    run_name or f"ablation-seed{cfg.seed}-{arm}",
    "--agent.gail.enabled",
    str(use_gail),
  ]
  if use_gail:
    command.extend(("--agent.gail.dataset-path", str(cfg.dataset_path.resolve())))
  return command


def choose_num_envs(measurements: dict[int, float], target: float) -> int:
  if not measurements:
    raise ValueError("At least one GPU utilization measurement is required")
  return min(measurements, key=lambda value: abs(measurements[value] - target))


def first_learning_target(
  points: list[LearningPoint], num_envs: int, num_steps_per_env: int
) -> dict[str, float | int] | None:
  """Find the first stable tracking point using task-only metrics."""
  if not points:
    return None
  start_time = points[0].wall_time
  for point in points:
    if point.linear_velocity_reward >= 0.8 and point.episode_length >= 900:
      return {
        "iteration": point.iteration,
        "transitions": (point.iteration + 1) * num_envs * num_steps_per_env,
        "wall_time_s": point.wall_time - start_time,
      }
  return None


def _read_gpu_utilization() -> float:
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-gpu=utilization.gpu",
      "--format=csv,noheader,nounits",
      "--id=0",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  return float(result.stdout.strip().splitlines()[0])


def _run_and_measure(
  command: list[str],
  name: str,
  num_envs: int,
  iterations: int,
  interval: float,
  log_root: Path,
) -> RunResult:
  env = os.environ.copy()
  env["PYTHONUTF8"] = "1"
  start = time.perf_counter()
  previous_runs = set(log_root.glob(f"*_{name}"))
  process = subprocess.Popen(command, env=env)
  samples: list[float] = []
  active_log_dir: Path | None = None
  while process.poll() is None:
    time.sleep(interval)
    if active_log_dir is None:
      new_runs = set(log_root.glob(f"*_{name}")) - previous_runs
      if len(new_runs) == 1:
        candidate = next(iter(new_runs))
        if next(candidate.glob("events.out.tfevents.*"), None) is not None:
          active_log_dir = candidate
    if active_log_dir is None:
      continue
    try:
      samples.append(_read_gpu_utilization())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
      pass
  if process.returncode:
    raise subprocess.CalledProcessError(process.returncode, command)
  if not samples:
    raise RuntimeError("nvidia-smi returned no GPU utilization samples")
  if active_log_dir is None:
    raise RuntimeError(f"No TensorBoard log directory was created for {name}")
  return RunResult(
    name=name,
    num_envs=num_envs,
    iterations=iterations,
    duration_s=time.perf_counter() - start,
    mean_gpu_utilization=sum(samples) / len(samples),
    samples=len(samples),
    log_dir=str(active_log_dir),
  )


def _learning_summary(
  log_dir: str, num_envs: int, num_steps_per_env: int = 24
) -> dict[str, object]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  events = EventAccumulator(log_dir, size_guidance={"scalars": 0}).Reload()
  linear = {
    item.step: item for item in events.Scalars("Episode_Reward/track_linear_velocity")
  }
  episode_length = {
    item.step: item for item in events.Scalars("Train/mean_episode_length")
  }
  points = [
    LearningPoint(
      iteration=step,
      linear_velocity_reward=linear[step].value,
      episode_length=episode_length[step].value,
      wall_time=max(linear[step].wall_time, episode_length[step].wall_time),
    )
    for step in sorted(linear.keys() & episode_length.keys())
  ]
  final = points[-1]
  return {
    "target": first_learning_target(points, num_envs, num_steps_per_env),
    "final_iteration": final.iteration,
    "final_linear_velocity_reward": final.linear_velocity_reward,
    "final_episode_length": final.episode_length,
  }


def run_ablation(cfg: AblationConfig) -> None:
  if not cfg.dataset_path.is_file():
    raise FileNotFoundError(f"Expert dataset not found: {cfg.dataset_path}")
  if not cfg.num_envs or any(value < 1 for value in cfg.num_envs):
    raise ValueError("num_envs must contain positive values")

  calibration: list[RunResult] = []
  for num_envs in cfg.num_envs:
    name = f"ablation-calibration-{num_envs}env"
    command = build_train_command(
      cfg,
      num_envs,
      use_gail=False,
      iterations=cfg.calibration_iterations,
      run_name=name,
    )
    calibration.append(
      _run_and_measure(
        command,
        name,
        num_envs,
        cfg.calibration_iterations,
        cfg.sample_interval_s,
        cfg.log_root,
      )
    )

  measurements = {
    result.num_envs: result.mean_gpu_utilization for result in calibration
  }
  selected = choose_num_envs(measurements, cfg.target_gpu_utilization)
  selected_utilization = measurements[selected]
  if (
    not cfg.calibrate_only
    and abs(selected_utilization - cfg.target_gpu_utilization) > cfg.gpu_tolerance
  ):
    raise RuntimeError(
      f"Closest calibration was {selected_utilization:.1f}% at {selected} envs; "
      f"target is {cfg.target_gpu_utilization:.1f}% ± {cfg.gpu_tolerance:.1f}%. "
      "Provide a better --num-envs candidate list."
    )

  runs: list[RunResult] = []
  if not cfg.calibrate_only:
    for use_gail in (False, True):
      arm = "gail" if use_gail else "ppo"
      name = f"ablation-seed{cfg.seed}-{selected}env-{arm}"
      runs.append(
        _run_and_measure(
          build_train_command(cfg, selected, use_gail=use_gail, run_name=name),
          name,
          selected,
          cfg.iterations,
          cfg.sample_interval_s,
          cfg.log_root,
        )
      )

  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  output = cfg.output_dir / f"gail_ablation_{datetime.now():%Y%m%d_%H%M%S}.json"
  output.write_text(
    json.dumps(
      {
        "config": {
          **asdict(cfg),
          "dataset_path": str(cfg.dataset_path),
          "output_dir": str(cfg.output_dir),
          "log_root": str(cfg.log_root),
        },
        "selected_num_envs": selected,
        "calibration": [asdict(result) for result in calibration],
        "runs": [
          {
            **asdict(result),
            "learning": _learning_summary(result.log_dir, result.num_envs),
          }
          for result in runs
        ],
      },
      indent=2,
    ),
    encoding="utf-8",
  )
  print(f"[INFO] Ablation report: {output}")


def main() -> None:
  run_ablation(tyro.cli(AblationConfig))


if __name__ == "__main__":
  main()
