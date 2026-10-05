"""Resume the same GAIL policy with GAIL on versus off for a short ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch

from mjlab.scripts.gail_curriculum_comparison import TASK, preflight
from mjlab.scripts.gail_curriculum_extension import (
  DEFAULT_ROOT,
  build_extension_config,
)
from mjlab.scripts.train import launch_training

CHECKPOINT = Path(
  "logs/rsl_rl/g1_gail_curriculum/2026-09-29_15-05-36/final_extension/"
  "gail_final_extension/2026-09-30_00-58-48_seed42/model_1498.pt"
)
ITERATIONS = 200


def build_ablation_config(
  mode: str, checkpoint: Path, root: Path, iterations: int = ITERATIONS
):
  if mode not in ("on", "off"):
    raise ValueError("Ablation mode must be 'on' or 'off'")
  arm: Literal["ppo", "gail"] = "gail" if mode == "on" else "ppo"
  cfg = build_extension_config(arm, checkpoint, root, iterations)
  cfg.agent.experiment_name = f"gail_resume_{mode}"
  return cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("on", "off"))
  parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT / "resume_ablation")
  args = parser.parse_args()
  preflight()
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  if checkpoint.get("iter") != 1498:
    raise ValueError("Resume ablation requires model_1498")
  launch_training(
    TASK,
    build_ablation_config(args.mode, args.checkpoint, args.root),
  )


if __name__ == "__main__":
  main()
