"""Evaluate a velocity checkpoint on a fixed command grid and apply an expert gate."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExpertGate:
  min_success_rate: float = 0.95
  max_linear_rmse: float = 0.25
  max_yaw_rmse: float = 0.20
  min_upright_mean: float = 0.97


@dataclass(frozen=True)
class EvaluateExpertConfig:
  checkpoint_file: Path
  output_file: Path
  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  episodes_per_command: int = 16
  seed: int = 42
  device: str = "cuda:0"
  stochastic_policy: bool = False
  lin_vel_x: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
  lin_vel_y: tuple[float, ...] = (-1.0, 0.0, 1.0)
  ang_vel_z: tuple[float, ...] = (-0.7, 0.0, 0.7)
  gate: ExpertGate = ExpertGate()


def build_command_grid(
  lin_vel_x: tuple[float, ...],
  lin_vel_y: tuple[float, ...],
  ang_vel_z: tuple[float, ...],
) -> list[tuple[float, float, float]]:
  if not lin_vel_x or not lin_vel_y or not ang_vel_z:
    raise ValueError("command axes must not be empty")
  return list(itertools.product(lin_vel_x, lin_vel_y, ang_vel_z))


class GateResult(TypedDict):
  passed: bool
  failed_bins: list[int]
  failures: dict[str, list[str]]


def evaluate_gate(bins: list[dict[str, float]], gate: ExpertGate) -> GateResult:
  failures: dict[str, list[str]] = {}
  for index, metrics in enumerate(bins):
    failed = []
    if metrics["success_rate"] < gate.min_success_rate:
      failed.append("success_rate")
    if metrics["linear_rmse"] > gate.max_linear_rmse:
      failed.append("linear_rmse")
    if metrics["yaw_rmse"] > gate.max_yaw_rmse:
      failed.append("yaw_rmse")
    if metrics["upright_mean"] < gate.min_upright_mean:
      failed.append("upright_mean")
    if failed:
      failures[str(index)] = failed
  failed_bins = [int(index) for index in failures]
  return {"passed": not failed_bins, "failed_bins": failed_bins, "failures": failures}


def classify_successful_episodes(
  timed_out: torch.Tensor,
  linear_rmse: torch.Tensor,
  yaw_rmse: torch.Tensor,
  upright_mean: torch.Tensor,
  gate: ExpertGate,
) -> torch.Tensor:
  """Require both survival and task tracking before calling an episode successful."""
  return (
    timed_out
    & (linear_rmse <= gate.max_linear_rmse)
    & (yaw_rmse <= gate.max_yaw_rmse)
    & (upright_mean >= gate.min_upright_mean)
  )


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def set_fixed_commands(term: UniformVelocityCommand, commands: torch.Tensor) -> None:
  term.pin_commands(commands)


def refresh_fixed_command_observation(env: Any) -> Any:
  """Recompute observations after a caller restores fixed commands."""
  return env.get_observations()


def failure_breakdown(
  timed_out: torch.Tensor,
  linear_rmse: torch.Tensor,
  yaw_rmse: torch.Tensor,
  upright_mean: torch.Tensor,
  gate: ExpertGate,
) -> dict[str, int]:
  """Count overlapping episode failure predicates for diagnosis."""
  success = classify_successful_episodes(
    timed_out, linear_rmse, yaw_rmse, upright_mean, gate
  )
  return {
    "success": int(success.sum()),
    "fall": int((~timed_out).sum()),
    "linear_rmse": int((linear_rmse > gate.max_linear_rmse).sum()),
    "yaw_rmse": int((yaw_rmse > gate.max_yaw_rmse).sum()),
    "upright": int((upright_mean < gate.min_upright_mean).sum()),
  }


def run_evaluation(cfg: EvaluateExpertConfig) -> dict[str, object]:
  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_file}")
  if cfg.output_file.exists():
    raise FileExistsError(f"Output already exists: {cfg.output_file}")
  if cfg.episodes_per_command < 1:
    raise ValueError("episodes_per_command must be positive")

  configure_torch_backends()
  grid = build_command_grid(cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z)
  commands = torch.tensor(grid, device=cfg.device).repeat_interleave(
    cfg.episodes_per_command, dim=0
  )
  bin_ids = torch.arange(len(grid), device=cfg.device).repeat_interleave(
    cfg.episodes_per_command
  )
  env_cfg = load_env_cfg(cfg.task)
  env_cfg.scene.num_envs = len(commands)
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  agent_cfg = load_rl_cfg(cfg.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device),
    clip_actions=agent_cfg.clip_actions,
  )
  try:
    term = cast(UniformVelocityCommand, env.unwrapped.command_manager.get_term("twist"))
    set_fixed_commands(term, commands)
    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(cfg.checkpoint_file),
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)
    if cfg.stochastic_policy and runner.alg.actor.distribution is None:
      raise ValueError("Stochastic evaluation requires an actor distribution")
    obs = env.reset()[0].to(cfg.device)
    robot = env.unwrapped.scene["robot"]
    active = torch.ones(len(commands), dtype=torch.bool, device=cfg.device)
    timed_out = torch.zeros_like(active)
    fell = torch.zeros_like(active)
    samples = torch.zeros(len(commands), device=cfg.device)
    linear_sq = torch.zeros_like(samples)
    yaw_sq = torch.zeros_like(samples)
    upright_sum = torch.zeros_like(samples)
    actual_vx_sum = torch.zeros_like(samples)

    with torch.inference_mode():
      for _ in range(env.max_episode_length):
        actual_linear = robot.data.root_link_lin_vel_b[:, :2]
        actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
        actual_vx_sum += active * actual_linear[:, 0]
        linear_sq += active * torch.sum((commands[:, :2] - actual_linear) ** 2, dim=1)
        yaw_sq += active * (commands[:, 2] - actual_yaw) ** 2
        gravity_b = quat_apply_inverse(
          robot.data.root_link_quat_w, robot.data.gravity_vec_w
        )
        upright_sum += active * torch.exp(
          -torch.sum(gravity_b[:, :2] ** 2, dim=1) / 0.2
        )
        samples += active
        actions = (
          runner.alg.actor(obs, stochastic_output=True)
          if cfg.stochastic_policy
          else policy(obs)
        )
        obs, _, dones, extras = env.step(actions)
        newly_done = dones.bool() & active
        if newly_done.any():
          time_outs = extras.get("time_outs", torch.zeros_like(newly_done))
          timed_out |= newly_done & time_outs.bool()
          fell |= newly_done & ~time_outs.bool()
          active &= ~newly_done
        if not active.any():
          break

    per_env_linear = torch.sqrt(linear_sq / samples.clamp(min=1))
    per_env_yaw = torch.sqrt(yaw_sq / samples.clamp(min=1))
    per_env_upright = upright_sum / samples.clamp(min=1)
    per_env_actual_vx = actual_vx_sum / samples.clamp(min=1)
    success = classify_successful_episodes(
      timed_out,
      per_env_linear,
      per_env_yaw,
      per_env_upright,
      cfg.gate,
    )
    bins = []
    for index, command in enumerate(grid):
      selected = bin_ids == index
      success_count = int(success[selected].sum())
      episode_count = int(selected.sum())
      bins.append(
        {
          "command": command,
          "success_rate": success_count / episode_count,
          "success_count": success_count,
          "episode_count": episode_count,
          "timeout_rate": timed_out[selected].float().mean().item(),
          "fall_rate": fell[selected].float().mean().item(),
          "mean_actual_vx": per_env_actual_vx[selected].mean().item(),
          "linear_rmse": per_env_linear[selected].mean().item(),
          "yaw_rmse": per_env_yaw[selected].mean().item(),
          "upright_mean": per_env_upright[selected].mean().item(),
          "failure_breakdown": failure_breakdown(
            timed_out[selected],
            per_env_linear[selected],
            per_env_yaw[selected],
            per_env_upright[selected],
            cfg.gate,
          ),
        }
      )
    gate_metrics = [
      {
        name: value
        for name, value in item.items()
        if name
        not in (
          "command",
          "failure_breakdown",
          "success_count",
          "episode_count",
        )
      }
      for item in bins
    ]
    gate_result = evaluate_gate(gate_metrics, cfg.gate)
    result = {
      "expert_policy": "stochastic" if cfg.stochastic_policy else "deterministic",
      "checkpoint": str(cfg.checkpoint_file.resolve()),
      "checkpoint_sha256": _sha256(cfg.checkpoint_file),
      "config": asdict(cfg),
      "bins": bins,
      "gate": gate_result,
    }
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(
      json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"[INFO] Expert gate: {'PASS' if gate_result['passed'] else 'FAIL'}")
    print(f"[INFO] Report: {cfg.output_file}")
    return result
  finally:
    env.close()


def main() -> None:
  run_evaluation(tyro.cli(EvaluateExpertConfig))


if __name__ == "__main__":
  main()
