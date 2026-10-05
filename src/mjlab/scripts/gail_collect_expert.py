"""Collect command-conditioned GAIL transitions from a trained policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from tqdm import tqdm

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import (
  VELOCITY_GAIL_FEATURE_SCHEMA,
  MjlabOnPolicyRunner,
  RslRlVecEnvWrapper,
  velocity_gail_state,
)
from mjlab.scripts.gail_evaluate_expert import set_fixed_commands
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollectConfig:
  checkpoint_file: Path
  output_path: Path
  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  num_envs: int = 1024
  transitions: int = 100_000
  seed: int = 42
  device: str = "cuda:0"
  command_name: str = "twist"
  curriculum_start_step: int | None = None
  balanced_command_bins: tuple[int, int, int] | None = (7, 5, 5)
  gate_report: Path | None = None
  """Require a passing gate report for this exact checkpoint before collecting."""
  standard_command_grid: bool = False
  """Use the locked 4x3x3 Standard Expert Gate command grid."""
  target_speed_command_grid: bool = False
  """Use the qualified forward target grid: 1.2, 1.35, and 1.5 m/s."""
  forward_speeds: tuple[float, ...] = ()
  """Collect explicit forward speeds; requires a matching passing gate report."""
  include_images: bool = False
  """Store raw NHWC uint8 camera frames when the task has a camera sensor."""
  camera_sensor_name: str = "g1_vision"
  airl_compatible: bool = False
  """Save true successor and actor inputs for AIRL."""
  stochastic_expert: bool = True
  """For AIRL, sample Gaussian actions; False collects raw deterministic means."""


def _trim_and_merge(
  batches: list[dict[str, torch.Tensor]], count: int
) -> dict[str, torch.Tensor]:
  return {
    name: torch.cat([batch[name] for batch in batches])[:count] for name in batches[0]
  }


def _collection_batch(
  *,
  observations: torch.Tensor,
  next_observations: torch.Tensor,
  actions: torch.Tensor,
  commands: torch.Tensor,
  dones: torch.Tensor,
  time_outs: torch.Tensor,
  environment_ids: torch.Tensor,
  episode_ids: torch.Tensor,
  images: torch.Tensor | None = None,
  actor_observations: torch.Tensor | None = None,
  next_actor_observations: torch.Tensor | None = None,
  next_commands: torch.Tensor | None = None,
  terminated: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  """Build one aligned collector batch before episode counters advance."""
  return {
    "observations": observations.cpu(),
    "next_observations": next_observations.cpu(),
    "actions": actions.cpu(),
    "commands": commands.cpu(),
    "dones": dones.bool().cpu(),
    "terminated": (
      terminated.bool() if terminated is not None else dones.bool() & ~time_outs.bool()
    ).cpu(),
    "truncated": time_outs.bool().cpu(),
    "environment_ids": environment_ids.cpu(),
    "episode_ids": episode_ids.cpu(),
    **({"images": images.cpu()} if images is not None else {}),
    **{
      name: value.cpu()
      for name, value in {
        "actor_observations": actor_observations,
        "next_actor_observations": next_actor_observations,
        "next_commands": next_commands,
      }.items()
      if value is not None
    },
  }


def _capture_airl_transition(
  env: ManagerBasedRlEnv, command_name: str
) -> dict[str, torch.Tensor]:
  actor_obs = env.obs_buf["actor"]
  if not isinstance(actor_obs, torch.Tensor):
    raise ValueError("AIRL collection requires concatenated policy observations")
  commands = env.command_manager.get_command(command_name)
  if not isinstance(commands, torch.Tensor):
    raise ValueError(f"Command '{command_name}' is unavailable")
  return {
    "next_observations": velocity_gail_state(env.scene["robot"]),
    "next_actor_observations": actor_obs,
    "next_commands": commands,
  }


def _load_collection_env_cfg(task: str) -> ManagerBasedRlEnvCfg:
  """Keep expert commands aligned with the training distribution."""
  return load_env_cfg(task)


def _activate_curriculum_step(env: RslRlVecEnvWrapper, step: int) -> None:
  """Apply a curriculum boundary before collecting the first transition."""
  if step < 0:
    raise ValueError("curriculum_start_step must be non-negative")
  env.unwrapped.common_step_counter = step
  env.unwrapped.curriculum_manager.compute()
  env.reset()


def _checkpoint_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as checkpoint:
    for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _effective_command_ranges(
  env: RslRlVecEnvWrapper, command_name: str
) -> dict[str, tuple[float, float]]:
  term = env.unwrapped.command_manager.get_term(command_name)
  if term is None:
    raise RuntimeError(f"Command '{command_name}' is unavailable")
  cfg = cast(UniformVelocityCommandCfg, term.cfg)
  return {
    "lin_vel_x": cfg.ranges.lin_vel_x,
    "lin_vel_y": cfg.ranges.lin_vel_y,
    "ang_vel_z": cfg.ranges.ang_vel_z,
  }


def _balanced_commands(
  ranges: dict[str, tuple[float, float]],
  num_envs: int,
  bins: tuple[int, int, int],
  device: str,
) -> torch.Tensor:
  if any(count < 2 for count in bins):
    raise ValueError("balanced command bins must all be at least 2")
  axes = [
    torch.linspace(*ranges[name], bins[index], device=device)
    for index, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z"))
  ]
  grid = torch.cartesian_prod(*axes)
  repeats = (num_envs + len(grid) - 1) // len(grid)
  return grid.repeat((repeats, 1))[:num_envs]


def _standard_commands(num_envs: int, device: str) -> torch.Tensor:
  """Build the locked Standard Expert Gate command grid."""
  axes = [
    torch.tensor(values, dtype=torch.float32, device=device)
    for values in ((-0.5, 0.0, 0.5, 1.0), (-0.3, 0.0, 0.3), (-0.2, 0.0, 0.2))
  ]
  grid = torch.cartesian_prod(*axes)
  repeats = (num_envs + len(grid) - 1) // len(grid)
  return grid.repeat((repeats, 1))[:num_envs]


def _target_speed_commands(num_envs: int, device: str) -> torch.Tensor:
  """Build the fixed target-speed gate grid for forward velocity tracking."""
  grid = torch.tensor(
    ((1.2, 0.0, 0.0), (1.35, 0.0, 0.0), (1.5, 0.0, 0.0)),
    dtype=torch.float32,
    device=device,
  )
  repeats = (num_envs + len(grid) - 1) // len(grid)
  return grid.repeat((repeats, 1))[:num_envs]


def _validate_gate_report(
  report_path: Path, checkpoint_file: Path
) -> dict[str, object]:
  report = json.loads(report_path.read_text(encoding="utf-8"))
  if not isinstance(report, dict):
    raise ValueError(f"Gate report must contain a dictionary: {report_path}")
  gate = report.get("gate")
  if not isinstance(gate, dict) or gate.get("passed") is not True:
    raise ValueError(f"Gate report does not pass: {report_path}")
  report_checkpoint = Path(str(report.get("checkpoint", ""))).resolve()
  if report_checkpoint != checkpoint_file.resolve():
    raise ValueError(
      "Gate report checkpoint does not match collection checkpoint: "
      f"{report_checkpoint} != {checkpoint_file.resolve()}"
    )
  return report


def collect(cfg: CollectConfig) -> None:
  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_file}")
  gate_report: dict[str, object] | None = None
  if cfg.gate_report is not None:
    if not cfg.gate_report.is_file():
      raise FileNotFoundError(f"Gate report not found: {cfg.gate_report}")
    gate_report = _validate_gate_report(cfg.gate_report, cfg.checkpoint_file)
    if cfg.airl_compatible:
      expected_mode = "stochastic" if cfg.stochastic_expert else "deterministic"
      gate_config = gate_report.get("config")
      if not isinstance(gate_config, dict):
        raise ValueError("AIRL gate expert policy mode must match collection")
      gate_config = cast(dict[str, object], gate_config)
      if (
        gate_report.get("expert_policy") != expected_mode
        or gate_config.get("stochastic_policy") is not cfg.stochastic_expert
      ):
        raise ValueError("AIRL gate expert policy mode must match collection")
    if cfg.airl_compatible and gate_report.get(
      "checkpoint_sha256"
    ) != _checkpoint_sha256(cfg.checkpoint_file):
      raise ValueError("Gate checkpoint SHA-256 mismatch")
  if cfg.output_path.exists():
    raise FileExistsError(f"Output already exists: {cfg.output_path}")
  if cfg.num_envs < 1 or cfg.transitions < 1:
    raise ValueError("num_envs and transitions must be positive")
  if cfg.standard_command_grid and cfg.target_speed_command_grid:
    raise ValueError("Choose only one fixed command grid")
  if cfg.forward_speeds:
    if cfg.standard_command_grid or cfg.target_speed_command_grid:
      raise ValueError("Choose only one fixed command grid")
    if gate_report is None:
      raise ValueError("forward_speeds requires a passing gate report")
    if gate_report.get("checkpoint_sha256") != _checkpoint_sha256(cfg.checkpoint_file):
      raise ValueError("Gate checkpoint SHA-256 mismatch")
    bins = cast(list[dict], gate_report.get("bins", []))
    covered = {
      tuple(b["command"]) for b in bins if "fall_rate" in b and "timeout_rate" in b
    }
    if any((v, 0.0, 0.0) not in covered for v in cfg.forward_speeds):
      raise ValueError("Forward speeds not covered by the current evaluator gate")

  configure_torch_backends()
  env_cfg = _load_collection_env_cfg(cfg.task)
  agent_cfg = load_rl_cfg(cfg.task)
  if cfg.airl_compatible and agent_cfg.clip_actions is not None:
    raise ValueError("AIRL collection requires unclipped raw policy actions")
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device),
    clip_actions=agent_cfg.clip_actions,
  )
  try:
    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(cfg.checkpoint_file),
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    if cfg.curriculum_start_step is not None:
      _activate_curriculum_step(env, cfg.curriculum_start_step)
    effective_ranges = _effective_command_ranges(env, cfg.command_name)
    command_term = cast(
      UniformVelocityCommand,
      env.unwrapped.command_manager.get_term(cfg.command_name),
    )
    fixed_commands = None
    if cfg.forward_speeds:
      grid = torch.tensor(
        [(v, 0.0, 0.0) for v in cfg.forward_speeds], device=cfg.device
      )
      fixed_commands = grid[torch.arange(cfg.num_envs, device=cfg.device) % len(grid)]
      set_fixed_commands(command_term, fixed_commands)
    elif cfg.target_speed_command_grid:
      fixed_commands = _target_speed_commands(cfg.num_envs, cfg.device)
      set_fixed_commands(command_term, fixed_commands)
    elif cfg.standard_command_grid:
      fixed_commands = _standard_commands(cfg.num_envs, cfg.device)
      set_fixed_commands(command_term, fixed_commands)
    elif cfg.balanced_command_bins is not None:
      fixed_commands = _balanced_commands(
        effective_ranges, cfg.num_envs, cfg.balanced_command_bins, cfg.device
      )
      set_fixed_commands(command_term, fixed_commands)
    policy = runner.get_inference_policy(device=cfg.device)
    if cfg.airl_compatible:
      actor = runner.alg.actor
      if type(actor) is not MLPModel or tuple(actor.obs_groups) != ("actor",):
        raise ValueError("AIRL collection requires an MLP with only the actor group")
      if not isinstance(actor.distribution, GaussianDistribution):
        raise ValueError("AIRL collection requires a stochastic Gaussian actor")
      env.unwrapped.set_transition_capture(
        lambda inner: _capture_airl_transition(inner, cfg.command_name)
      )
    obs = (env.reset()[0] if fixed_commands is not None else env.get_observations()).to(
      cfg.device
    )
    batches: list[dict[str, torch.Tensor]] = []
    collection_start_step = env.unwrapped.common_step_counter
    collected = 0
    episode_ids = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
    environment_ids = torch.arange(cfg.num_envs, device=cfg.device)
    progress = tqdm(total=cfg.transitions, unit="transition")
    with torch.inference_mode():
      while collected < cfg.transitions:
        robot = env.unwrapped.scene["robot"]
        images = None
        if cfg.include_images:
          camera = env.unwrapped.scene[cfg.camera_sensor_name]
          images = camera.data.rgb
          if images is None:
            raise RuntimeError(f"Camera '{cfg.camera_sensor_name}' has no RGB data")
          if images.dtype != torch.uint8 or images.ndim != 4:
            raise RuntimeError("Collected camera frames must be NHWC uint8")
        actor_observations = obs["actor"].clone() if cfg.airl_compatible else None
        actions = (
          runner.alg.actor(obs, stochastic_output=True)
          if cfg.airl_compatible and cfg.stochastic_expert
          else policy(obs)
        )
        commands = env.unwrapped.command_manager.get_command(cfg.command_name)
        if not isinstance(commands, torch.Tensor):
          raise RuntimeError(f"Command '{cfg.command_name}' is unavailable")
        current_states = velocity_gail_state(robot)
        current_commands = commands.clone()
        current_episode_ids = episode_ids.clone()
        obs, _, dones, extras = env.step(actions.to(env.device))
        time_outs = env.unwrapped.reset_time_outs.bool()
        transition = extras["transition"] if cfg.airl_compatible else {}
        next_states = transition.get("next_observations", velocity_gail_state(robot))
        batches.append(
          _collection_batch(
            observations=current_states,
            next_observations=next_states,
            actions=actions,
            commands=current_commands,
            dones=dones,
            time_outs=time_outs,
            environment_ids=environment_ids,
            episode_ids=current_episode_ids,
            images=images,
            actor_observations=actor_observations,
            next_actor_observations=transition.get("next_actor_observations"),
            next_commands=transition.get("next_commands"),
            terminated=transition.get("terminated"),
          )
        )
        episode_ids += dones.bool()
        added = min(cfg.num_envs, cfg.transitions - collected)
        collected += added
        progress.update(added)
    progress.close()

    merged = _trim_and_merge(batches, cfg.transitions)
    data: dict[str, object] = {**merged}
    data["metadata"] = {
      "task": cfg.task,
      "checkpoint": str(cfg.checkpoint_file.resolve()),
      "checkpoint_sha256": _checkpoint_sha256(cfg.checkpoint_file),
      "seed": cfg.seed,
      "curriculum_start_step": cfg.curriculum_start_step,
      "effective_collection_start_step": collection_start_step,
      "effective_command_ranges": effective_ranges,
      "balanced_command_bins": cfg.balanced_command_bins,
      "standard_command_grid": cfg.standard_command_grid,
      "target_speed_command_grid": cfg.target_speed_command_grid,
      "forward_speeds": cfg.forward_speeds,
      "gate_report": str(cfg.gate_report.resolve()) if cfg.gate_report else None,
      "gate_passed": gate_report is not None,
      "feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA,
      "state_dim": 68,
      "num_transitions": cfg.transitions,
      "schema_version": 4 if cfg.airl_compatible else 3,
      "transition_contract": "pre_reset_v1" if cfg.airl_compatible else None,
      "expert_policy": (
        "stochastic"
        if cfg.airl_compatible and cfg.stochastic_expert
        else "deterministic"
      ),
      "action_contract": (
        ("raw_policy_sample" if cfg.stochastic_expert else "raw_policy_mean")
        if cfg.airl_compatible
        else None
      ),
      "actor_observation_contract": "policy_input_v1" if cfg.airl_compatible else None,
      "actor_observation_group": "actor" if cfg.airl_compatible else None,
      "actor_observation_terms": (
        list(env.unwrapped.observation_manager.active_terms["actor"])
        if cfg.airl_compatible
        else None
      ),
      "actor_observation_dim": (
        merged["actor_observations"].shape[-1] if cfg.airl_compatible else None
      ),
      "action_dim": merged["actions"].shape[-1],
      "image_layout": "NHWC" if cfg.include_images else None,
      "image_dtype": "uint8" if cfg.include_images else None,
      "camera_sensor_name": cfg.camera_sensor_name if cfg.include_images else None,
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cfg.output_path)
    print(
      f"[INFO] Wrote {cfg.output_path}: "
      + ", ".join(
        f"{name}={tuple(value.shape)}"
        for name, value in data.items()
        if isinstance(value, torch.Tensor)
      )
    )
  finally:
    env.close()


def main() -> None:
  collect(tyro.cli(CollectConfig))


if __name__ == "__main__":
  main()
