"""Run paired fresh PPO/guarded frozen trials with the recorded 1501-update schedule."""

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from extend_frozen_gail import same

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.scripts.gail_curriculum_comparison import TASK, build_config
from mjlab.scripts.gail_curriculum_extension import build_extension_config
from mjlab.scripts.train import launch_training
from mjlab.tasks.velocity.rl.gail_runner import GailVelocityOnPolicyRunner

ROOT = Path(__file__).resolve().parents[2]
DONOR = (
  ROOT
  / "logs/rsl_rl/g1_gail_curriculum/fresh_v3_20260930/final_extension/gail_final_extension/2026-10-01_16-11-42_seed42/model_1200.pt"
)
DATASET = (
  ROOT / "experiments/g1_velocity/expert/g1_expert_curriculum_0p3_1p2_v3_fixed_obs.pt"
)
CAP = 3.9056143760681152


def configuration(arm, stage, root, checkpoint=None):
  if stage == "final":
    if checkpoint is None:
      raise ValueError("Final stage requires checkpoint")
    cfg = build_extension_config(
      "ppo" if arm == "ppo" else "gail", checkpoint, root, dataset=DATASET
    )
  else:
    cfg = build_config(1 if stage == "pilot" else 2, root, checkpoint, DATASET)
  cfg = replace(cfg, enable_nan_guard=True)
  assert isinstance(cfg.agent, RslRlOnPolicyRunnerCfg)
  cfg.agent.experiment_name = f"{arm}_{stage}"
  cfg.agent.gail.enabled = arm == "guarded"
  cfg.agent.gail.frozen = arm == "guarded"
  cfg.agent.gail.discriminator_checkpoint = str(DONOR) if arm == "guarded" else ""
  cfg.agent.gail.weight = 0.0002 if arm == "guarded" else 0.01
  cfg.agent.gail.frozen_reward_cap = CAP if arm == "guarded" else None
  cfg.agent.gail.validate()
  return cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--root", type=Path, default=ROOT / "logs/rsl_rl/guarded_frozen_comparison_20261002"
  )
  parser.add_argument("--arm", choices=("ppo", "guarded"))
  parser.add_argument("--stage", choices=("pilot", "phase2", "final"))
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--check-only", action="store_true")
  args = parser.parse_args()
  if bool(args.arm) != bool(args.stage):
    parser.error("arm and stage must be specified together")
  if args.arm is None:
    if args.check_only:
      for arm in ("ppo", "guarded"):
        for stage in ("pilot", "phase2", "final"):
          cfg = configuration(
            arm, stage, args.root, None if stage == "pilot" else Path("placeholder.pt")
          )
          assert isinstance(cfg.agent, RslRlOnPolicyRunnerCfg)
          print(arm, stage, cfg.agent.max_iterations, cfg.agent.gail)
      if not DONOR.is_file() or not DATASET.is_file():
        raise FileNotFoundError("Missing donor/dataset")
      return
    if args.root.exists():
      raise FileExistsError("Inspect existing comparison before starting another run")
    for arm in ("ppo", "guarded"):
      checkpoint = None
      for stage, endpoint in (("pilot", 199), ("phase2", 999), ("final", 1498)):
        command = [
          sys.executable,
          str(Path(__file__).resolve()),
          "--root",
          str(args.root),
          "--arm",
          arm,
          "--stage",
          stage,
        ]
        if checkpoint is not None:
          command += ["--checkpoint", str(checkpoint)]
        subprocess.run(command, check=True, cwd=ROOT)
        files = list((args.root / f"{arm}_{stage}").glob(f"*/model_{endpoint}.pt"))
        if len(files) != 1:
          raise RuntimeError("Expected exactly one completed stage checkpoint")
        checkpoint = files[0]
        if stage != "pilot":
          for seed in (42002, 42003):
            subprocess.run(
              [
                sys.executable,
                "-m",
                "mjlab.scripts.gail_evaluate_expert",
                "--checkpoint-file",
                str(checkpoint),
                "--output-file",
                str(args.root / "evaluations" / f"{arm}_{stage}_{seed}.json"),
                "--episodes-per-command",
                "100",
                "--seed",
                str(seed),
                "--lin-vel-x",
                "0",
                "0.8",
                "1.0",
                "1.2",
                "--lin-vel-y",
                "0",
                "--ang-vel-z",
                "0",
              ],
              check=True,
              cwd=ROOT,
            )
    return
  if (args.stage == "pilot") != (args.checkpoint is None):
    parser.error("pilot must be fresh; continuation requires checkpoint")
  cfg = configuration(args.arm, args.stage, args.root, args.checkpoint)
  source = None
  if args.checkpoint is not None:
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = 199 if args.stage == "phase2" else 999
    if source["iter"] != expected:
      raise ValueError("Wrong stage checkpoint iteration")
    has_d = "gail_state_dict" in (source.get("infos") or {})
    if has_d != (args.arm == "guarded"):
      raise ValueError("Checkpoint belongs to the other arm")
  if args.check_only:
    print(cfg.agent)
    return
  donor = torch.load(DONOR, map_location="cpu", weights_only=False)["infos"][
    "gail_state_dict"
  ]
  original_load = GailVelocityOnPolicyRunner.load
  original_save = GailVelocityOnPolicyRunner.save

  def checked_load(runner, path, **kwargs):
    infos = original_load(
      runner, path, allow_frozen_resume=args.arm == "guarded", **kwargs
    )
    assert source is not None
    for name in ("actor", "critic", "optimizer"):
      assert same(getattr(runner.alg, name).state_dict(), source[f"{name}_state_dict"])
    assert runner.current_learning_iteration == source["iter"]
    print("Full PPO restore verified", flush=True)
    return infos

  def checked_save(runner, path, infos=None):
    if args.arm == "guarded":
      assert same(runner.gail.discriminator.state_dict(), donor)
    original_save(runner, path, infos)

  with (
    patch.object(GailVelocityOnPolicyRunner, "load", checked_load),
    patch.object(GailVelocityOnPolicyRunner, "save", checked_save),
  ):
    launch_training(TASK, cfg)


if __name__ == "__main__":
  main()
