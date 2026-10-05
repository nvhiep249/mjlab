"""Continue the seed-42 PPO and GAIL arms for 500 final-curriculum updates."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import torch
import yaml

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.scripts.gail_curriculum_comparison import DATASET, GATE, TASK, preflight
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

Arm = Literal["ppo", "gail"]
FINAL_COMMANDS = (1.0, 1.1, 1.2)
ITERATIONS = 500
PPO_CHECKPOINT = Path(
  "logs/rsl_rl/g1_target_curriculum/g1_velocity/"
  "2026-09-29_11-06-15_seed42-curriculum-resume-to-1000/model_999.pt"
)
GAIL_CHECKPOINT = Path(
  "logs/rsl_rl/g1_gail_curriculum/2026-09-29_15-05-36/gail_phase2/"
  "2026-09-29_20-09-23_seed42/model_999.pt"
)
DEFAULT_ROOT = Path(
  "logs/rsl_rl/g1_gail_curriculum/2026-09-29_15-05-36/final_extension"
)


def expected_final_iteration(start_iteration: int, updates: int) -> int:
  if updates <= 0:
    raise ValueError("Extension iterations must be positive")
  return start_iteration + updates - 1


def validate_extension_checkpoint(arm: Arm, checkpoint: dict[str, object]) -> None:
  if checkpoint.get("iter") != 999:
    raise ValueError("Extension must resume from model_999")
  infos = checkpoint.get("infos")
  has_gail = isinstance(infos, dict) and "gail_state_dict" in infos
  if arm == "ppo" and has_gail:
    raise ValueError("PPO extension received a GAIL checkpoint")
  if arm == "gail" and not has_gail:
    raise ValueError("GAIL extension requires discriminator state")


def build_extension_config(
  arm: Arm,
  checkpoint: Path,
  root: Path,
  iterations: int = ITERATIONS,
  dataset: Path = DATASET,
) -> TrainConfig:
  if arm not in ("ppo", "gail"):
    raise ValueError(f"Unknown extension arm: {arm}")
  cfg = TrainConfig.from_task(TASK)
  cfg = replace(cfg, log_root=str(root), checkpoint_file=checkpoint)
  cfg.env.scene.num_envs = 16384
  cfg.env.seed = 42
  command = cast(UniformVelocityCommandCfg, cfg.env.commands["twist"])
  command.target_speed_command_grid = True
  command.target_speed_curriculum = True
  # A resumed process has a fresh environment counter, so final speeds begin at zero.
  command.target_speed_curriculum_stages = ((0, FINAL_COMMANDS),)
  agent = cast(RslRlOnPolicyRunnerCfg, cfg.agent)
  agent.seed = 42
  agent.num_steps_per_env = 24
  if iterations <= 0:
    raise ValueError("Extension iterations must be positive")
  agent.max_iterations = iterations
  agent.save_interval = 100
  agent.experiment_name = f"{arm}_final_extension"
  agent.run_name = "seed42"
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.algorithm.num_learning_epochs = 5
  agent.algorithm.num_mini_batches = 4
  agent.gail.enabled = arm == "gail"
  if arm == "gail":
    agent.gail.dataset_path = str(dataset)
    agent.gail.match_expert_commands = True
    agent.gail.weight = 0.01
    agent.gail.learning_rate = 3e-4
    agent.gail.updates = 1
  return cfg


def validate_extension(
  ppo_checkpoint: Path,
  gail_checkpoint: Path,
  root: Path,
  dataset: Path = DATASET,
  gate: Path = GATE,
) -> None:
  preflight(dataset, gate)
  ppo = torch.load(ppo_checkpoint, map_location="cpu", weights_only=False)
  gail = torch.load(gail_checkpoint, map_location="cpu", weights_only=False)
  validate_extension_checkpoint("ppo", ppo)
  validate_extension_checkpoint("gail", gail)
  if "gail_optimizer_state_dict" not in gail["infos"]:
    raise ValueError("GAIL extension requires discriminator optimizer state")
  saved = yaml.load(
    (gail_checkpoint.parent / "params/agent.yaml").read_text(encoding="utf-8"),
    Loader=yaml.BaseLoader,
  )
  if Path(saved["gail"]["dataset_path"]).resolve() != dataset.resolve():
    raise ValueError("Extension dataset does not match source GAIL checkpoint config")
  ppo_cfg = build_extension_config("ppo", ppo_checkpoint, root)
  gail_cfg = build_extension_config("gail", gail_checkpoint, root, dataset=dataset)
  if ppo_cfg.env != gail_cfg.env:
    raise ValueError("PPO/GAIL extension environments differ")
  if ppo_cfg.agent.seed != gail_cfg.agent.seed:
    raise ValueError("PPO/GAIL extension seeds differ")
  if ppo_cfg.agent.max_iterations != gail_cfg.agent.max_iterations:
    raise ValueError("PPO/GAIL extension budgets differ")
  print(
    "Extension preflight PASS: paired final curriculum, 500 updates, "
    f"expected final label {expected_final_iteration(999, ITERATIONS)}."
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("check", "ppo", "gail"))
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--ppo-checkpoint", type=Path, default=PPO_CHECKPOINT)
  parser.add_argument("--gail-checkpoint", type=Path, default=GAIL_CHECKPOINT)
  parser.add_argument("--dataset", type=Path, default=DATASET)
  parser.add_argument("--gate", type=Path, default=GATE)
  args = parser.parse_args()
  validate_extension(
    args.ppo_checkpoint, args.gail_checkpoint, args.root, args.dataset, args.gate
  )
  if args.mode == "check":
    return
  arm = cast(Arm, args.mode)
  checkpoint = args.ppo_checkpoint if arm == "ppo" else args.gail_checkpoint
  launch_training(
    TASK, build_extension_config(arm, checkpoint, args.root, dataset=args.dataset)
  )


if __name__ == "__main__":
  main()
