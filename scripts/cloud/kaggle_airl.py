"""Pack the current source; measure single-GPU AIRL; launch matched PPO arms.

Run with uv run --no-sync python scripts/cloud/kaggle_airl.py --help.
All sweep candidates run in separate processes, releasing Torch/Warp allocations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TASK = "Mjlab-Velocity-Flat-Unitree-G1-AIRL-MVP"
DATASET = "airl_expert_1p2_deterministic_v1.pt"
CHECKPOINT = "ppo_common_999.pt"
ENVS = (16384, 20480, 24576, 32768)
STEPS = 24


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def pack(output: Path, dataset: Path, checkpoint: Path) -> None:
  # Explicit source/artifact allowlist: never include .venv, logs, or credentials.
  if output.exists():
    raise FileExistsError(f"Choose a new bundle directory: {output}")
  for path in (dataset, checkpoint):
    if not path.is_file():
      raise FileNotFoundError(path)
  output.mkdir(parents=True)
  files = [
    ROOT / name for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE")
  ]
  files.extend(
    path
    for path in (ROOT / "src").rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix not in (".pyc", ".pem", ".key")
    and not path.name.startswith(".env")
  )
  files.append(Path(__file__).resolve())
  files.append(ROOT / "scripts/cloud/kaggle_airl.ipynb")
  files.append(ROOT / "docs/guides/airl_dataset_2026-10-01.md")
  files.append(ROOT / "docs/guides/kaggle_airl_2026-10-05.md")
  files.extend(
    (ROOT / "report/airl_smoke_20261001").glob("teacher35550_deterministic_*.json")
  )
  files.append(ROOT / "report/airl_smoke_20261001/deterministic_dataset_audit.json")
  source_hashes = {}
  with zipfile.ZipFile(
    output / "mjlab_source.zip", "w", zipfile.ZIP_DEFLATED
  ) as archive:
    for path in sorted(files):
      relative = path.relative_to(ROOT).as_posix()
      source_hashes[relative] = sha256(path)
      archive.write(path, relative)
  shutil.copyfile(dataset, output / DATASET)
  shutil.copyfile(checkpoint, output / CHECKPOINT)
  shutil.copyfile(
    ROOT / "scripts/cloud/kaggle_airl.ipynb", output / "kaggle_airl.ipynb"
  )
  write_json(
    output / "manifest.json",
    {
      "task": TASK,
      "created_unix": time.time(),
      "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
      ).strip(),
      "git_status": subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
      ),
      "artifact_sources": {
        DATASET: str(dataset.resolve()),
        CHECKPOINT: str(checkpoint.resolve()),
      },
      "source_files": source_hashes,
      "files": {
        name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
        for name in ("mjlab_source.zip", DATASET, CHECKPOINT, "kaggle_airl.ipynb")
      },
    },
  )
  print(f"Private Kaggle input prepared: {output.resolve()}")


def update_count(transitions: int, envs: int) -> int:
  rollout = envs * STEPS
  if transitions < rollout or transitions % rollout:
    raise ValueError(f"Transition budget must be a positive multiple of {rollout}")
  return transitions // rollout


def choose(results: list[dict], headroom: float = 0.15) -> dict:
  eligible = [
    row
    for row in results
    if row.get("ok")
    and row["envs"] > 16000
    and math.isfinite(row["transitions_per_second"])
    and row["transitions_per_second"] > 0
    and row["peak_device_mib"] <= (1 - headroom) * row["total_device_mib"]
  ]
  if not eligible:
    raise RuntimeError("No >16,000-env candidate passed with 15% VRAM headroom")
  fastest = max(row["transitions_per_second"] for row in eligible)
  # Within measurement noise, prefer more free VRAM, then fewer environments.
  close = [row for row in eligible if row["transitions_per_second"] >= 0.97 * fastest]
  return min(close, key=lambda row: (row["peak_device_mib"], row["envs"]))


def validate_selection_runtime(selected: dict, runtime: dict) -> None:
  for key in ("gpu", "cuda", "packages", "total_device_mib"):
    if selected.get(key) != runtime[key]:
      raise ValueError(f"Selection runtime changed ({key}); remeasure before training")


def worker(args: argparse.Namespace) -> None:
  started = time.perf_counter()
  # Set visibility before importing Torch/Warp; AIRL has no DDP synchronization.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
  import torch
  import warp as wp

  import mjlab.tasks  # noqa: F401
  from mjlab.rl.config import RslRlOnPolicyRunnerCfg
  from mjlab.scripts import train
  from mjlab.tasks.velocity.rl.airl_runner import AirlVelocityOnPolicyRunner

  if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("One working CUDA GPU is required")
  major, _ = torch.cuda.get_device_capability(0)
  if major < 7:
    raise RuntimeError("Pinned cu128 stack requires a newer GPU than P100/Pascal")
  # Real CUDA operations, not just nvidia-smi detection.
  assert (torch.ones(4, device="cuda:0") + 1).sum().item() == 8
  wp.init()
  probe = wp.zeros(4, device="cuda:0")
  wp.synchronize_device("cuda:0")
  del probe
  runtime = {
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda,
    "total_device_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
    "packages": {
      name: importlib.metadata.version(name)
      for name in ("torch", "warp-lang", "mujoco", "mujoco-warp", "rsl-rl-lib")
    },
  }
  if args.selected is not None:
    validate_selection_runtime(json.loads(args.selected.read_text()), runtime)
  samples: list[float] = []
  memory: list[float] = []
  stop = threading.Event()
  memory_errors: list[str] = []

  def monitor() -> None:
    while not stop.is_set():
      try:
        result = subprocess.run(
          [
            "nvidia-smi",
            f"--id={args.gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
          ],
          capture_output=True,
          text=True,
          check=True,
          timeout=5,
        )
        memory.append(float(result.stdout.strip()))
      except (OSError, ValueError, subprocess.SubprocessError) as error:
        memory_errors.append(str(error))
        return
      stop.wait(0.25)

  monitor_thread = threading.Thread(target=monitor, daemon=True)
  monitor_thread.start()
  config = train.TrainConfig.from_task(TASK)
  config = replace(config, checkpoint_file=args.checkpoint, video=False)
  if not isinstance(config.agent, RslRlOnPolicyRunnerCfg):
    raise TypeError("The selected task must use the PPO runner configuration")
  checkpoint_infos = (
    torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("infos")
    or {}
  )
  if args.arm == "ppo" and "airl_state_dict" in checkpoint_infos:
    raise ValueError("PPO control cannot load an AIRL treatment checkpoint")
  config.env.scene.num_envs = args.envs
  config.agent.seed = args.seed
  config.agent.resume = args.resume
  config.agent.logger = "tensorboard"
  config.agent.upload_model = False
  config.agent.num_steps_per_env = STEPS
  config.agent.algorithm.num_mini_batches = args.minibatches
  config.agent.gail.enabled = False
  config.agent.airl.enabled = args.arm == "airl"
  config.agent.airl.dataset_path = str(args.input / DATASET)
  config.agent.airl.match_expert_commands = True
  config.agent.max_iterations = args.updates
  config.agent.save_interval = 1_000_000 if args.benchmark else 50

  # Keep task defaults, loaded optimizer LR/std/normalizers, and all validations.
  # Benchmark suppresses checkpoint/ONNX output only; real runs save normally.
  class MeasuredRunner(AirlVelocityOnPolicyRunner):
    def load(self, *positional, **keywords):
      result = super().load(*positional, **keywords)
      if self.airl is None and args.resume:
        # RSL stores the last completed index; AIRL already restores next index.
        self.current_learning_iteration += 1
      return result

    def learn(self, *positional, **keywords):
      original = self.logger.log
      last = time.perf_counter()
      seen = 0

      def measured_log(*positional, **keywords):
        nonlocal last, seen
        torch.cuda.synchronize()
        now = time.perf_counter()
        if seen >= args.warmup:
          samples.append(now - last)
        seen += 1
        result = original(*positional, **keywords)
        last = time.perf_counter()
        return result

      with patch.object(self.logger, "log", measured_log):
        return super().learn(*positional, **keywords)

    def save(self, *positional, **keywords):
      if not args.benchmark:
        return super().save(*positional, **keywords)

  args.output.mkdir(parents=True, exist_ok=True)
  run_dir = (
    args.output / f"{args.arm}_n{args.envs}_mb{args.minibatches}_seed{args.seed}"
  )
  if run_dir.exists():
    raise FileExistsError(f"Choose a new output folder; run already exists: {run_dir}")
  try:
    with patch.object(train, "load_runner_cls", return_value=MeasuredRunner):
      train.run_train(TASK, config, run_dir)
  finally:
    stop.set()
    monitor_thread.join(timeout=6)
  if not samples or not memory or memory_errors:
    raise RuntimeError(f"Incomplete timing/device-memory measurements: {memory_errors}")
  total = torch.cuda.get_device_properties(0).total_memory / 2**20
  reserved = torch.cuda.max_memory_reserved() / 2**20
  median = statistics.median(samples)
  write_json(
    args.result,
    {
      "ok": True,
      "envs": args.envs,
      "minibatches": args.minibatches,
      "arm": args.arm,
      "gpu": runtime["gpu"],
      "torch": torch.__version__,
      "cuda": torch.version.cuda,
      "packages": runtime["packages"],
      "checkpoint_sha256": sha256(args.checkpoint),
      "seed": args.seed,
      "rollout_transitions": args.envs * STEPS,
      "ppo_minibatch_size": args.envs * STEPS // args.minibatches,
      "median_update_seconds": median,
      "update_seconds": samples,
      "transitions_per_second": args.envs * STEPS / median,
      "peak_device_mib": max(max(memory), reserved),
      "torch_peak_reserved_mib": reserved,
      "total_device_mib": total,
      "wall_seconds_including_startup": time.perf_counter() - started,
      "run_dir": str(run_dir),
      "additional_updates": args.updates,
      "additional_transitions": args.updates * args.envs * STEPS,
      "resume": args.resume,
    },
  )


def child_command(
  args: argparse.Namespace,
  envs: int,
  minibatches: int,
  output: Path,
  result: Path,
  updates: int,
  benchmark: bool,
) -> list[str]:
  command = [
    sys.executable,
    str(Path(__file__).resolve()),
    "worker",
    "--input",
    str(args.input),
    "--checkpoint",
    str(args.checkpoint),
    "--output",
    str(output),
    "--result",
    str(result),
    "--envs",
    str(envs),
    "--minibatches",
    str(minibatches),
    "--updates",
    str(updates),
    "--gpu",
    str(args.gpu),
    "--arm",
    args.arm,
    "--seed",
    str(args.seed),
    "--warmup",
    str(args.warmup if benchmark else 0),
  ]
  if benchmark:
    command.append("--benchmark")
  if args.resume:
    command.append("--resume")
  if not benchmark and getattr(args, "selected", None) is not None:
    command.extend(("--selected", str(args.selected)))
  return command


def sweep(args: argparse.Namespace) -> None:
  if args.output.exists():
    raise FileExistsError(f"Use a new sweep directory: {args.output}")
  args.output.mkdir(parents=True)
  results = []
  for envs in args.envs:
    for minibatches in args.minibatches:
      if envs * STEPS % minibatches:
        raise ValueError("PPO minibatches must divide the rollout exactly")
      name = f"n{envs}_mb{minibatches}"
      result = args.output / f"{name}.json"
      command = child_command(
        args,
        envs,
        minibatches,
        args.output / name,
        result,
        args.warmup + args.measure,
        True,
      )
      print(
        f"Measuring {name}: {args.warmup} warmup + {args.measure} full updates",
        flush=True,
      )
      with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
        try:
          completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
          )
          returncode = completed.returncode
        except subprocess.TimeoutExpired:
          returncode = -1
      row = (
        json.loads(result.read_text())
        if returncode == 0
        else {
          "ok": False,
          "envs": envs,
          "minibatches": minibatches,
          "returncode": returncode,
          "log": f"{name}.log",
        }
      )
      results.append(row)
      write_json(args.output / "results.json", results)
  selected = choose(results)
  write_json(args.output / "selected.json", selected)
  print(json.dumps(selected, indent=2))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  packing = commands.add_parser("pack")
  packing.add_argument("--output", type=Path, required=True)
  packing.add_argument(
    "--dataset", type=Path, default=ROOT / "experiments/g1_velocity" / DATASET
  )
  packing.add_argument(
    "--checkpoint",
    type=Path,
    default=ROOT
    / (
      "logs/rsl_rl/g1_target_curriculum/g1_velocity/"
      "2026-09-29_11-06-15_seed42-curriculum-resume-to-1000/model_999.pt"
    ),
  )
  for name in ("sweep", "train", "worker"):
    command = commands.add_parser(name)
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--arm", choices=("ppo", "airl"), default="airl")
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--warmup", type=int, default=3)
    if name == "sweep":
      command.add_argument("--envs", type=int, nargs="+", default=list(ENVS))
      command.add_argument("--minibatches", type=int, nargs="+", default=[4, 8])
      command.add_argument("--measure", type=int, default=8)
      command.add_argument("--timeout", type=int, default=1800)
    elif name == "train":
      command.add_argument("--selected", type=Path, required=True)
      command.add_argument("--transitions", type=int, default=23592960)
    else:
      command.add_argument("--selected", type=Path)
      command.add_argument("--envs", type=int, required=True)
      command.add_argument("--minibatches", type=int, required=True)
      command.add_argument("--updates", type=int, required=True)
      command.add_argument("--result", type=Path, required=True)
      command.add_argument("--benchmark", action="store_true")
  args = parser.parse_args()
  if args.command == "pack":
    pack(args.output, args.dataset, args.checkpoint)
    return
  args.checkpoint = args.checkpoint or args.input / CHECKPOINT
  if args.command == "worker":
    worker(args)
  elif args.command == "sweep":
    sweep(args)
  else:
    selected = json.loads(args.selected.read_text())
    if not selected.get("ok"):
      raise ValueError("A successful measured selection is required")
    updates = update_count(args.transitions, selected["envs"])
    command = child_command(
      args,
      selected["envs"],
      selected["minibatches"],
      args.output,
      args.output / "run_summary.json",
      updates,
      False,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
  main()
