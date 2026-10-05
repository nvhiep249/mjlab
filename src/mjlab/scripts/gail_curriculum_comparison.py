"""Prepare or run the GAIL counterpart of the recorded September 29 PPO run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
import yaml

import mjlab.tasks  # noqa: F401
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.rl.gail import CommandMatchedSampler, GailTransitionDataset
from mjlab.scripts.gail_collect_expert import _checkpoint_sha256
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
DATASET = Path("experiments/g1_velocity/expert/g1_expert_curriculum_0p3_1p2_v2.pt")
GATE = Path("report/g1_teacher_curriculum_preparation_20260929.json")
PPO_RUNS = (
  Path(
    "logs/rsl_rl/g1_target_curriculum/ppo_curriculum_pilot/"
    "2026-09-29_09-47-53_seed42-stage1-200it"
  ),
  Path(
    "logs/rsl_rl/g1_target_curriculum/g1_velocity/"
    "2026-09-29_11-06-15_seed42-curriculum-resume-to-1000"
  ),
)
SPEEDS = (0.3, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2)


def _normalize_legacy_agent_config(expected: dict, current: dict) -> None:
  """An absent legacy AIRL block is equivalent only to explicitly disabled AIRL."""
  if "airl" not in expected and current.get("airl", {}).get("enabled") == "false":
    expected["airl"] = current["airl"]


def _phase2_iterations(checkpoint: dict[str, object]) -> int:
  iteration = checkpoint.get("iter")
  infos = checkpoint.get("infos")
  if not isinstance(infos, dict) or "gail_state_dict" not in infos:
    raise ValueError("Phase2 resume requires a GAIL checkpoint")
  if not isinstance(iteration, int) or not 0 <= iteration < 999:
    raise ValueError("Phase2 resume checkpoint must be before model_999")
  return 1000 - iteration


def build_config(
  phase: int,
  root: Path,
  checkpoint: Path | None = None,
  dataset: Path = DATASET,
) -> TrainConfig:
  if phase not in (1, 2) or (phase == 1 and checkpoint is not None):
    raise ValueError("Phase1 must start from scratch; phase must be 1 or 2")
  cfg = TrainConfig.from_task(TASK)
  cfg = replace(cfg, log_root=str(root), checkpoint_file=checkpoint)
  cfg.env.scene.num_envs = 16384
  cfg.env.seed = 42
  command = cast(UniformVelocityCommandCfg, cfg.env.commands["twist"])
  command.target_speed_command_grid = True
  command.target_speed_curriculum = True
  command.target_speed_curriculum_stages = (
    (0, (0.3, 0.5, 0.8)),
    (4800 if phase == 1 else 12000, (0.6, 0.9, 1.2)),
    (12000 if phase == 1 else 24000, (1.0, 1.1, 1.2)),
  )
  agent = cast(RslRlOnPolicyRunnerCfg, cfg.agent)
  agent.seed = 42
  agent.num_steps_per_env = 24
  agent.max_iterations = 200 if phase == 1 else 801
  agent.save_interval = 50 if phase == 1 else 100
  agent.experiment_name = f"gail_phase{phase}"
  agent.run_name = "seed42"
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.algorithm.num_learning_epochs = 5
  agent.algorithm.num_mini_batches = 4
  agent.gail.enabled = True
  agent.gail.dataset_path = str(dataset)
  agent.gail.match_expert_commands = True
  agent.gail.weight = 0.01
  agent.gail.learning_rate = 3e-4
  agent.gail.updates = 1
  return cfg


def preflight(dataset: Path = DATASET, gate: Path = GATE) -> None:
  report = json.loads(gate.read_text(encoding="utf-8"))
  if report["gate"]["passed"] is not True:
    raise ValueError("Teacher gate did not pass")
  expert = GailTransitionDataset.load(dataset)
  expert.require_feature_schema("g1_velocity_body_local_v1")
  teacher_hash = _checkpoint_sha256(Path(report["checkpoint"]))
  if not (
    expert.metadata.get("checkpoint_sha256")
    == report["checkpoint_sha256"]
    == teacher_hash
  ):
    raise ValueError("Teacher/dataset/gate provenance mismatch")
  if expert.metadata.get("gate_report") != str(gate.resolve()):
    raise ValueError("Dataset was not collected under the current teacher gate")
  commands = torch.tensor([(v, 0.0, 0.0) for v in SPEEDS])
  CommandMatchedSampler(expert._data["commands"]).sample(commands)
  # Read trusted saved YAML as basic values; never construct Python objects.
  for phase, ppo_run in enumerate(PPO_RUNS, 1):
    actual = asdict(build_config(phase, Path("unused")))
    for section in ("agent", "env"):
      expected = yaml.load(
        (ppo_run / "params" / f"{section}.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
      )
      current = yaml.load(yaml.dump(actual[section]), Loader=yaml.BaseLoader)
      if section == "env":
        # Scene initialization propagates num_envs into the terrain config.
        current["scene"]["terrain"]["num_envs"] = current["scene"]["num_envs"]
      if section == "agent":
        _normalize_legacy_agent_config(expected, current)
        for key in ("gail", "experiment_name", "run_name"):
          expected.pop(key, None)
          current.pop(key, None)
      if current != expected:
        different = [k for k in current if current[k] != expected.get(k)]
        raise ValueError(f"Phase {phase} {section} differs from PPO: {different}")
  print("Preflight PASS: expert coverage/provenance and saved PPO configs match.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("check", "train", "smoke", "phase1", "phase2"))
  parser.add_argument("--root", type=Path)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--dataset", type=Path, default=DATASET)
  parser.add_argument("--gate", type=Path, default=GATE)
  args = parser.parse_args()
  preflight(args.dataset, args.gate)
  if args.mode == "check":
    return
  root = args.root or Path("logs/rsl_rl/g1_gail_curriculum") / datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
  )
  if args.mode == "train":
    if root.exists():
      raise FileExistsError(root)
    # Separate processes reproduce the reference's interruption and seed reset.
    module = "mjlab.scripts.gail_curriculum_comparison"
    base = [sys.executable, "-m", module]
    subprocess.run(
      base
      + [
        "phase1",
        "--root",
        str(root),
        "--dataset",
        str(args.dataset),
        "--gate",
        str(args.gate),
      ],
      check=True,
    )
    checkpoints = list((root / "gail_phase1").glob("*/model_199.pt"))
    if len(checkpoints) != 1:
      raise RuntimeError("Expected exactly one GAIL phase1 checkpoint")
    subprocess.run(
      base
      + [
        "phase2",
        "--root",
        str(root),
        "--checkpoint",
        str(checkpoints[0]),
        "--dataset",
        str(args.dataset),
        "--gate",
        str(args.gate),
      ],
      check=True,
    )
    return
  phase = 2 if args.mode == "phase2" else 1
  remaining_iterations = None
  if phase == 2:
    if args.checkpoint is None:
      raise ValueError("phase2 requires its GAIL checkpoint")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    remaining_iterations = _phase2_iterations(ckpt)
  cfg = build_config(phase, root, args.checkpoint, args.dataset)
  if remaining_iterations is not None:
    cfg.agent.max_iterations = remaining_iterations
  if args.mode == "smoke":
    cfg.env.scene.num_envs = 64
    cfg.agent.max_iterations = 2
    cfg.agent.experiment_name = "smoke_not_benchmark"
  launch_training(TASK, cfg)


if __name__ == "__main__":
  main()
