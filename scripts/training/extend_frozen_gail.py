"""Replay the recorded 801 + 500 update schedule with frozen D1200."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.scripts.gail_curriculum_comparison import TASK, build_config
from mjlab.scripts.gail_curriculum_extension import build_extension_config
from mjlab.scripts.train import launch_training
from mjlab.tasks.velocity.rl.gail_runner import GailVelocityOnPolicyRunner

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
  "logs/rsl_rl/frozen_gail/g1_frozen_d/"
  "2026-10-01_22-51-21_frozen_d1200_pilot200_retry/model_199.pt"
)
DATASET = ROOT / (
  "experiments/g1_velocity/expert/g1_expert_curriculum_0p3_1p2_v3_fixed_obs.pt"
)
OUTPUT = ROOT / "logs/rsl_rl/frozen_gail/extension_20261002"


def same(a, b):
  if isinstance(a, torch.Tensor):
    return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
  if isinstance(a, dict):
    return (
      isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    )
  if isinstance(a, (list, tuple)):
    return (
      isinstance(b, (list, tuple))
      and len(a) == len(b)
      and all(same(x, y) for x, y in zip(a, b, strict=True))
    )
  return a == b


def build_frozen_config(stage: str, checkpoint: Path, root: Path, donor: Path):
  if stage == "phase2":
    cfg = build_config(2, root, checkpoint, DATASET)
  elif stage == "final":
    cfg = build_extension_config("gail", checkpoint, root, dataset=DATASET)
  else:
    raise ValueError("Unknown frozen stage")
  cfg = replace(cfg, enable_nan_guard=True)
  assert isinstance(cfg.agent, RslRlOnPolicyRunnerCfg)
  cfg.agent.experiment_name = f"frozen_{stage}"
  cfg.agent.gail.frozen = True
  cfg.agent.gail.discriminator_checkpoint = str(donor)
  cfg.agent.gail.validate()
  return cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", choices=("all", "phase2", "final"), default="all")
  parser.add_argument("--checkpoint", type=Path, default=SOURCE)
  parser.add_argument("--root", type=Path, default=OUTPUT)
  parser.add_argument("--check-only", action="store_true")
  args = parser.parse_args()
  os.chdir(ROOT)
  source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  saved = yaml.load(
    (args.checkpoint.parent / "params/agent.yaml").read_text(encoding="utf-8"),
    Loader=yaml.BaseLoader,
  )
  if saved["gail"]["frozen"] != "true":
    raise ValueError("Only a frozen pilot checkpoint can be continued")
  if Path(saved["gail"]["dataset_path"]).resolve() != DATASET:
    raise ValueError("Frozen dataset mismatch")
  donor = Path(saved["gail"]["discriminator_checkpoint"])
  donor_state = torch.load(donor, map_location="cpu", weights_only=False)["infos"][
    "gail_state_dict"
  ]
  if not same(source["infos"]["gail_state_dict"], donor_state):
    raise ValueError("Frozen discriminator mismatch")
  stage = "phase2" if args.stage == "all" else args.stage
  expected_iter = 199 if stage == "phase2" else 999
  if source["iter"] != expected_iter:
    raise ValueError(f"{stage} requires saved iteration {expected_iter}")
  for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
    if key not in source:
      raise ValueError(f"Missing full PPO state: {key}")
  assert torch.cuda.is_available() and torch.cuda.device_count() == 1
  torch.ones(1, device="cuda")
  cfg = build_frozen_config(stage, args.checkpoint, args.root, donor)
  assert isinstance(cfg.agent, RslRlOnPolicyRunnerCfg)
  # Preserve the reward objective of the source trial across process resets.
  cfg.agent.gail.weight = float(saved["gail"]["weight"])
  saved_cap = saved["gail"].get("frozen_reward_cap")
  if saved_cap not in (None, "null", "None", ""):
    cfg.agent.gail.frozen_reward_cap = float(saved_cap)
  cfg.agent.gail.validate()
  print(
    f"Preflight PASS: {stage}, saved={expected_iter}, updates={cfg.agent.max_iterations}",
    flush=True,
  )
  if args.check_only:
    return
  if args.stage == "all":
    if args.root.exists():
      raise FileExistsError("Inspect existing extension before starting again")
    base = [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root)]
    subprocess.run(
      base + ["--stage", "phase2", "--checkpoint", str(args.checkpoint)], check=True
    )
    checkpoints = list((args.root / "frozen_phase2").glob("*/model_999.pt"))
    if len(checkpoints) != 1:
      raise RuntimeError("Expected exactly one completed frozen phase2")
    subprocess.run(
      base + ["--stage", "final", "--checkpoint", str(checkpoints[0])], check=True
    )
    return
  original_load = GailVelocityOnPolicyRunner.load
  original_save = GailVelocityOnPolicyRunner.save

  def checked_load(runner, path, load_cfg=None, strict=True, map_location=None):
    infos = original_load(
      runner, path, load_cfg, strict, map_location, allow_frozen_resume=True
    )
    assert same(runner.alg.actor.state_dict(), source["actor_state_dict"])
    assert same(runner.alg.critic.state_dict(), source["critic_state_dict"])
    assert same(runner.alg.optimizer.state_dict(), source["optimizer_state_dict"])
    assert runner.current_learning_iteration == expected_iter
    assert runner.gail is not None and runner.gail.frozen
    assert same(runner.gail.discriminator.state_dict(), donor_state)
    print("Full PPO restore verified; D1200 remains frozen.", flush=True)
    return infos

  def checked_save(runner, path, infos=None):
    assert runner.gail is not None and runner.gail.frozen
    assert same(runner.gail.discriminator.state_dict(), donor_state)
    original_save(runner, path, infos)
    Path(path).parent.joinpath("frozen_verification.json").write_text(
      json.dumps(
        {
          "iter": runner.current_learning_iteration,
          "D_equal": True,
          "full_PPO_restore_verified": True,
        }
      )
      + "\n",
      encoding="utf-8",
    )

  with (
    patch.object(GailVelocityOnPolicyRunner, "load", checked_load),
    patch.object(GailVelocityOnPolicyRunner, "save", checked_save),
  ):
    launch_training(TASK, cfg)


if __name__ == "__main__":
  main()
