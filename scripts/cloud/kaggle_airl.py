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
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TASK = "Mjlab-Velocity-Flat-Unitree-G1-AIRL-MVP"
DATASET = "airl_expert_1p2_deterministic_v1.pt"
CHECKPOINT = "ppo_common_999.pt"
ENVS = (16384, 20480, 24576, 32768)
STEPS = 24
PPO_RNG_KEYS = ("kaggle_torch_rng", "kaggle_cuda_rng")


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
  # Reject a reused run before initializing CUDA/Warp or starting the monitor.
  args.output.mkdir(parents=True, exist_ok=True)
  run_dir = (
    args.output / f"{args.arm}_n{args.envs}_mb{args.minibatches}_seed{args.seed}"
  )
  if run_dir.exists():
    raise FileExistsError(f"Choose a new output folder; run already exists: {run_dir}")
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
  if args.arm == "ppo" and args.resume and args.require_rng:
    if any(key not in checkpoint_infos for key in PPO_RNG_KEYS):
      raise ValueError("Paired PPO resume requires a checkpoint with Torch RNG")
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
        if all(key in checkpoint_infos for key in PPO_RNG_KEYS):
          restore_ppo_rng(checkpoint_infos, self.device)
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

    def save(self, path, infos=None):
      if not args.benchmark:
        if self.airl is None:
          infos = {
            **(infos or {}),
            "kaggle_torch_rng": torch.get_rng_state(),
            "kaggle_cuda_rng": torch.cuda.get_rng_state(self.device),
          }
        return super().save(path, infos)

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
      "dataset_sha256": sha256(args.input / DATASET) if args.arm == "airl" else None,
      "source_manifest_sha256": source_manifest_hash(),
      "final_checkpoint_sha256": sha256(latest_checkpoint(run_dir))
      if not args.benchmark
      else None,
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
  if getattr(args, "require_rng", False):
    command.append("--require-rng")
  if not benchmark and getattr(args, "selected", None) is not None:
    command.extend(("--selected", str(args.selected)))
  return command


def latest_checkpoint(run_dir: Path) -> Path:
  paths = [
    path
    for path in run_dir.glob("model_*.pt")
    if path.is_file() and re.fullmatch(r"model_\d+", path.stem)
  ]
  if not paths:
    raise FileNotFoundError(f"Completed run has no checkpoint: {run_dir}")
  return max(paths, key=lambda path: int(path.stem.split("_")[-1]))


def source_manifest_hash() -> str | None:
  manifest = ROOT / "scripts/cloud/kaggle_source_manifest.json"
  return sha256(manifest) if manifest.is_file() else None


def restore_ppo_rng(infos: dict, device: str) -> None:
  # Keep cloud orchestration/imports hardware-free until a worker needs Torch.
  import torch

  torch.set_rng_state(infos["kaggle_torch_rng"].cpu())
  if torch.device(device).type == "cuda":
    torch.cuda.set_rng_state(infos["kaggle_cuda_rng"].cpu(), device)


def train_arm(args: argparse.Namespace, timeout: float | None = None) -> None:
  selected = json.loads(args.selected.read_text())
  if not selected.get("ok"):
    raise ValueError("A successful measured selection is required")
  updates = update_count(args.transitions, selected["envs"])
  result = args.output / "run_summary.json"
  if result.exists():
    if not args.reuse_completed:
      raise FileExistsError(f"Summary already exists; use --reuse-completed: {result}")
    row = json.loads(result.read_text())
    expected = {
      "ok": True,
      "arm": args.arm,
      "envs": selected["envs"],
      "minibatches": selected["minibatches"],
      "seed": args.seed,
      "checkpoint_sha256": sha256(args.checkpoint),
      "additional_updates": updates,
      "additional_transitions": args.transitions,
      "resume": args.resume,
    }
    if not isinstance(row, dict):
      raise ValueError(f"Invalid completed-run summary; preserve it: {result}")
    for key, value in expected.items():
      if row.get(key) != value:
        raise ValueError(f"Completed run differs ({key}); choose a new output folder")
    validate_selection_runtime(row, selected)
    run_dir = Path(row["run_dir"]).resolve()
    if not run_dir.is_relative_to(args.output.resolve()):
      raise ValueError("Completed run directory must stay inside the requested output")
    checkpoint = latest_checkpoint(run_dir)
    provenance = {
      "source_manifest_sha256": source_manifest_hash(),
      "final_checkpoint_sha256": sha256(checkpoint),
      "dataset_sha256": sha256(args.input / DATASET) if args.arm == "airl" else None,
    }
    for key, value in provenance.items():
      if key in row and row[key] != value:
        raise ValueError(f"Completed run differs ({key}); choose a new output folder")
    if any(key not in row for key in provenance):
      print("[INFO] Legacy summary: source/dataset/final hashes not all recorded")
    print(f"[INFO] Reusing completed {args.arm}: {run_dir} ({updates} updates)")
    return
  output = args.output
  if output.exists() and any(output.iterdir()):
    if not args.restart_incomplete:
      raise FileExistsError(
        f"No completion summary in {output}; use --restart-incomplete to restart "
        "the full budget from the requested checkpoint, preserving old files"
      )
    attempt = 1
    while (output / f"attempt_{attempt:03d}").exists():
      attempt += 1
    output = output / f"attempt_{attempt:03d}"
    print(
      f"[INFO] Restarting incomplete {args.arm} from requested checkpoint: {output}"
    )
  command = child_command(
    args, selected["envs"], selected["minibatches"], output, result, updates, False
  )
  subprocess.run(command, check=True, timeout=timeout)


def atomic_json(path: Path, data: dict) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  write_json(temporary, data)
  temporary.replace(path)


def copy_verified(source: Path, target: Path, digest: str) -> None:
  if not source.is_file() or sha256(source) != digest:
    raise ValueError(f"Missing or changed checkpoint/artifact: {source}")
  if target.exists():
    if sha256(target) != digest:
      raise ValueError(f"Refusing to replace a different artifact: {target}")
    return
  temporary = target.with_suffix(target.suffix + ".tmp")
  shutil.copyfile(source, temporary)
  if sha256(temporary) != digest:
    raise ValueError(f"Checkpoint copy hash mismatch: {temporary}")
  temporary.replace(target)


def state_artifact(folder: Path, name: str, digest: str) -> Path:
  if not isinstance(name, str) or not name or Path(name).name != name:
    raise ValueError("Paired checkpoint references must be flat relative filenames")
  path = folder / name
  if not path.resolve().is_relative_to(folder.resolve()):
    raise ValueError("Paired artifact escapes its state directory")
  if not path.is_file() or sha256(path) != digest:
    raise ValueError(f"Missing or changed paired artifact: {path}")
  return path


@contextmanager
def paired_writer(folder: Path):
  folder.mkdir(parents=True, exist_ok=True)
  lock = folder / ".paired.lock"
  try:
    stream = lock.open("x", encoding="utf-8")
  except FileExistsError as error:
    raise FileExistsError(
      f"Paired writer active or stale lock: {lock}. After the old process stops, "
      "use --state paired_state.json with a new --output directory."
    ) from error
  try:
    with stream:
      stream.write(str(os.getpid()))
    yield
  finally:
    lock.unlink()


def paired(args: argparse.Namespace) -> None:
  if not math.isfinite(args.session_seconds) or args.session_seconds <= 0:
    raise ValueError("session_seconds must be finite and positive")
  started = time.perf_counter()
  destination = args.output / "paired_state.json"
  original_state_hash = sha256(destination) if destination.exists() else None
  if args.state is not None and destination.exists():
    if args.state.resolve() != destination.resolve():
      raise FileExistsError("Import --state into a new --output directory")
  source_state = args.state or (destination if destination.exists() else None)
  state = json.loads(source_state.read_text()) if source_state is not None else None
  if state is not None and state.get("format") != "mjlab_paired_train_v1":
    raise ValueError("Unsupported paired state format")
  source_folder = source_state.parent if source_state is not None else args.output
  selection_file = args.selected or source_folder / "selected.json"
  selected = json.loads(selection_file.read_text())
  if not selected.get("ok") or selected["envs"] <= 16000:
    raise ValueError("Paired training requires a successful >16000-env selection")
  if not math.isfinite(selected["median_update_seconds"]) or (
    selected["median_update_seconds"] <= 0
  ):
    raise ValueError("Selection timing must be finite and positive")
  update_count(args.transitions, selected["envs"])
  update_count(args.chunk_transitions, selected["envs"])
  manifest_hash = source_manifest_hash()
  if manifest_hash is None:
    raise ValueError("Paired training requires the verified GitHub source manifest")
  initializer = args.checkpoint
  if initializer is None:
    initializer = (
      source_folder / "initial.pt" if state is not None else args.input / CHECKPOINT
    )
  config = {
    "task": TASK,
    "envs": selected["envs"],
    "minibatches": selected["minibatches"],
    "steps": STEPS,
    "seed": args.seed,
    "target_transitions": args.transitions,
    "chunk_transitions": args.chunk_transitions,
    "initial_checkpoint_sha256": sha256(initializer),
    "dataset_sha256": sha256(args.input / DATASET),
    "source_manifest_sha256": manifest_hash,
    "runtime": {
      key: selected[key] for key in ("gpu", "cuda", "packages", "total_device_mib")
    },
  }
  with paired_writer(args.output):
    current_hash = sha256(destination) if destination.exists() else None
    if current_hash != original_state_hash:
      raise ValueError("Paired state changed before lock acquisition; rerun command")
    if state is None:
      if any(path.name != ".paired.lock" for path in args.output.iterdir()):
        raise FileExistsError("No paired state in existing output; choose a new output")
      state = {
        "format": "mjlab_paired_train_v1",
        "source_commit": subprocess.check_output(
          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config": config,
        "selected_sha256": sha256(selection_file),
        "complete": False,
        "arms": {
          arm: {
            "transitions": 0,
            "checkpoint": "initial.pt",
            "checkpoint_sha256": config["initial_checkpoint_sha256"],
            "wall_seconds": 0.0,
          }
          for arm in ("ppo", "airl")
        },
        "history": [],
        "failed_attempts": [],
      }
      copy_verified(initializer, args.output / "initial.pt", sha256(initializer))
      copy_verified(
        selection_file, args.output / "selected.json", sha256(selection_file)
      )
    else:
      if state["config"] != config:
        raise ValueError(
          "Paired config/data/source/runtime changed; restore saved config"
        )
      if sha256(selection_file) != state["selected_sha256"]:
        raise ValueError("Paired selection hash changed; use saved selected.json")
      if set(state["arms"]) != {"ppo", "airl"}:
        raise ValueError("Paired state must contain exactly PPO and AIRL")
      if any(item["arm"] not in ("ppo", "airl") for item in state["history"]):
        raise ValueError("Invalid paired history arm")
      cached = state_artifact(source_folder, "selected.json", state["selected_sha256"])
      copy_verified(cached, args.output / "selected.json", state["selected_sha256"])
      initial = state_artifact(
        source_folder, "initial.pt", config["initial_checkpoint_sha256"]
      )
      copy_verified(initial, args.output / "initial.pt", sha256(initial))
      for arm in ("ppo", "airl"):
        row = state["arms"][arm]
        if not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
          raise ValueError("Invalid paired wall time")
        count = row["transitions"]
        if (
          type(count) is not int
          or not 0 <= count <= args.transitions
          or (count % (selected["envs"] * STEPS))
        ):
          raise ValueError("Invalid paired transition counter")
        history = [item for item in state["history"] if item["arm"] == arm]
        if count == 0 and (
          row["checkpoint"] != "initial.pt"
          or row["checkpoint_sha256"] != config["initial_checkpoint_sha256"]
        ):
          raise ValueError("Zero-count arm must use the common initializer")
        if any(
          type(item["transitions"]) is not int
          or item["transitions"] <= 0
          or item["transitions"] > args.chunk_transitions
          or item["updates"] != update_count(item["transitions"], selected["envs"])
          or item["resume"] != (index > 0)
          for index, item in enumerate(history)
        ):
          raise ValueError("Invalid paired chunk history")
        if sum(item["transitions"] for item in history) != count:
          raise ValueError("Paired history/counter mismatch")
        if history and (
          history[-1]["checkpoint_sha256"] != row["checkpoint_sha256"]
          or history[-1]["checkpoint"] != row["checkpoint"]
        ):
          raise ValueError("Paired checkpoint/history mismatch")
        checkpoint = state_artifact(
          source_folder, row["checkpoint"], row["checkpoint_sha256"]
        )
        copy_verified(
          checkpoint, args.output / row["checkpoint"], row["checkpoint_sha256"]
        )
    counts = [state["arms"][arm]["transitions"] for arm in ("ppo", "airl")]
    if max(counts) - min(counts) > args.chunk_transitions:
      raise ValueError("Paired arms differ by more than one chunk")
    state["complete"] = all(count == args.transitions for count in counts)
    atomic_json(destination, state)
    while not state["complete"]:
      arm = min(("ppo", "airl"), key=lambda name: state["arms"][name]["transitions"])
      row = state["arms"][arm]
      amount = min(args.chunk_transitions, args.transitions - row["transitions"])
      updates = update_count(amount, selected["envs"])
      remaining = args.session_seconds - (time.perf_counter() - started)
      estimate = updates * selected["median_update_seconds"] * 1.2 + 120
      if remaining < estimate:
        print(f"[INFO] Session budget reached; next arm={arm}. Resume {destination}")
        break
      parent = args.output / "chunks" / f"t{row['transitions']:012d}" / arm
      attempt = 1
      while (parent / f"attempt_{attempt:03d}").exists():
        attempt += 1
      job_output = parent / f"attempt_{attempt:03d}"
      checkpoint = state_artifact(
        args.output, row["checkpoint"], row["checkpoint_sha256"]
      )
      job = argparse.Namespace(
        input=args.input,
        checkpoint=checkpoint,
        selected=args.output / "selected.json",
        output=job_output,
        arm=arm,
        gpu=args.gpu,
        seed=args.seed,
        transitions=amount,
        resume=row["transitions"] > 0,
        require_rng=True,
        reuse_completed=False,
        restart_incomplete=False,
      )
      print(
        f"[INFO] Long {arm}: +{updates} updates; committed={row['transitions']} transitions",
        flush=True,
      )
      job_started = time.perf_counter()
      try:
        train_arm(job, timeout=remaining)
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        state["failed_attempts"].append(
          {
            "arm": arm,
            "output": str(job_output),
            "error": type(error).__name__,
            "wall_seconds": time.perf_counter() - job_started,
          }
        )
        atomic_json(destination, state)
        if isinstance(error, subprocess.TimeoutExpired):
          print(f"[INFO] Chunk timed out; committed state preserved: {destination}")
          break
        raise
      result = json.loads((job_output / "run_summary.json").read_text())
      expected = {
        "ok": True,
        "arm": arm,
        "envs": selected["envs"],
        "minibatches": selected["minibatches"],
        "seed": args.seed,
        "resume": job.resume,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "additional_updates": updates,
        "additional_transitions": amount,
        "source_manifest_sha256": manifest_hash,
        "dataset_sha256": config["dataset_sha256"] if arm == "airl" else None,
      }
      if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Long chunk summary/config mismatch; state not advanced")
      timings = result.get("update_seconds", [])
      if len(timings) != updates or any(
        not math.isfinite(value) or value <= 0 for value in timings
      ):
        raise ValueError(
          "Chunk has incomplete/nonfinite update timing; state not advanced"
        )
      validate_selection_runtime(result, selected)
      run_dir = Path(result["run_dir"])
      if not run_dir.resolve().is_relative_to(job_output.resolve()):
        raise ValueError("Chunk checkpoint escaped its output")
      final = latest_checkpoint(run_dir)
      digest = result["final_checkpoint_sha256"]
      total = row["transitions"] + amount
      name = f"{arm}_{total:012d}_{digest[:12]}.pt"
      copy_verified(final, args.output / name, digest)
      elapsed = time.perf_counter() - job_started
      row.update(
        transitions=total,
        checkpoint=name,
        checkpoint_sha256=digest,
        wall_seconds=row["wall_seconds"] + elapsed,
      )
      state["history"].append(
        {
          "arm": arm,
          "transitions": amount,
          "updates": updates,
          "resume": job.resume,
          "checkpoint": name,
          "checkpoint_sha256": digest,
          "wall_seconds": elapsed,
        }
      )
      state["complete"] = all(
        r["transitions"] == args.transitions for r in state["arms"].values()
      )
      atomic_json(destination, state)
    print(f"[INFO] Long training complete={state['complete']}; state={destination}")


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
      command.add_argument("--reuse-completed", action="store_true")
      command.add_argument("--restart-incomplete", action="store_true")
    else:
      command.add_argument("--selected", type=Path)
      command.add_argument("--envs", type=int, required=True)
      command.add_argument("--minibatches", type=int, required=True)
      command.add_argument("--updates", type=int, required=True)
      command.add_argument("--result", type=Path, required=True)
      command.add_argument("--benchmark", action="store_true")
      command.add_argument("--require-rng", action="store_true")
  long_train = commands.add_parser("paired")
  long_train.add_argument("--input", type=Path, required=True)
  long_train.add_argument("--checkpoint", type=Path)
  long_train.add_argument("--selected", type=Path)
  long_train.add_argument("--output", type=Path, required=True)
  long_train.add_argument("--state", type=Path)
  long_train.add_argument("--transitions", type=int, default=589824000)
  long_train.add_argument("--chunk-transitions", type=int, default=23592960)
  long_train.add_argument("--session-seconds", type=float, default=28800)
  long_train.add_argument("--gpu", type=int, default=0)
  long_train.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.command == "pack":
    pack(args.output, args.dataset, args.checkpoint)
    return
  if args.command == "paired":
    paired(args)
    return
  args.checkpoint = args.checkpoint or args.input / CHECKPOINT
  if args.command == "worker":
    worker(args)
  elif args.command == "sweep":
    sweep(args)
  else:
    train_arm(args)


if __name__ == "__main__":
  main()
