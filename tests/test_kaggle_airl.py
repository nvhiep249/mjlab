"""Hardware-free checks for the cloud selection and matched sample budgets."""

import ast
import builtins
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
  "kaggle_airl", Path(__file__).parents[1] / "scripts/cloud/kaggle_airl.py"
)
assert spec is not None and spec.loader is not None
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


def test_selection_requires_large_envs_memory_headroom_and_finite_throughput():
  rows = [
    {
      "ok": True,
      "envs": n,
      "transitions_per_second": speed,
      "peak_device_mib": memory,
      "total_device_mib": 10000,
    }
    for n, speed, memory in (
      (8192, 2000, 4000),
      (16384, 970, 6000),
      (20480, 1000, 7000),
      (24576, 1500, 9000),
      (32768, math.inf, 8000),
    )
  ]
  assert cloud.choose(rows)["envs"] == 16384
  with pytest.raises(RuntimeError, match="No >16,000-env"):
    cloud.choose(rows[3:])


def test_all_candidate_budgets_match_in_environment_transitions():
  budget = 23592960
  assert [cloud.update_count(budget, n) for n in cloud.ENVS] == [60, 48, 40, 30]
  for n in cloud.ENVS:
    assert cloud.update_count(budget, n) * n * cloud.STEPS == budget
  with pytest.raises(ValueError, match="multiple"):
    cloud.update_count(budget + 1, 16384)


def test_selection_cannot_be_reused_on_a_different_gpu_or_package_stack():
  runtime = {
    "gpu": "T4",
    "cuda": "12.8",
    "packages": {"torch": "2.9.0+cu128"},
    "total_device_mib": 15360,
  }
  cloud.validate_selection_runtime(dict(runtime), runtime)
  for key, value in (
    ("gpu", "P100"),
    ("total_device_mib", 8192),
    ("packages", {"torch": "2.8.0"}),
  ):
    with pytest.raises(ValueError, match="remeasure"):
      cloud.validate_selection_runtime({**runtime, key: value}, runtime)


def notebook_cell(cell_id: str) -> str:
  notebook = json.loads(
    (cloud.ROOT / "scripts/cloud/kaggle_airl.ipynb").read_text(encoding="utf-8")
  )
  return "".join(
    next(cell["source"] for cell in notebook["cells"] if cell["id"] == cell_id)
  )


def test_baseline_fail_is_reported_without_rejecting_qualified_dataset(
  tmp_path, capsys
):
  common = tmp_path / "model_999.pt"
  report = {"gate": {"passed": False, "failures": {"0": ["success_rate"]}}, "bins": []}
  calls = []

  def run(command, **kwargs):
    calls.append(command)
    output = Path(command[command.index("--output-file") + 1])
    output.write_text(json.dumps(report))

  namespace: dict[str, Any] = {
    "subprocess": SimpleNamespace(run=run),
    "json": json,
    "os": os,
    "SOURCE": tmp_path,
    "COMMON": common,
    "OUTPUT": tmp_path,
    "SESSION": "test",
    "manifest": {"task": cloud.TASK},
    "GPU": 0,
  }
  exec(compile(notebook_cell("airl-07"), "baseline_evaluation", "exec"), namespace)
  assert calls[0][calls[0].index("--evaluation-role") + 1] == "baseline"
  assert namespace["baseline"]["gate"]["passed"] is False
  assert "Common PPO baseline gate" in capsys.readouterr().out
  namespace["evaluate"](common, tmp_path / "endpoint.json", 42006)
  assert calls[-1][calls[-1].index("--evaluation-role") + 1] == "endpoint"


@pytest.mark.parametrize(
  "pinned,tampered", ((False, False), (True, False), (False, True))
)
def test_notebook_clones_named_branch_verifies_hashes_and_can_pin_commit(
  tmp_path, pinned, tampered
):
  remote = tmp_path / "remote"
  remote.mkdir()
  branch = "feature/kaggle-airl-16k"

  def git(*args):
    return subprocess.check_output(
      ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
      cwd=remote,
      text=True,
    ).strip()

  git("init", "--initial-branch", branch)
  git("config", "core.autocrlf", "false")
  (remote / "src").mkdir()
  code = remote / "src/current.py"
  code.write_bytes(b"# current runtime\n")
  (remote / "scripts/cloud").mkdir(parents=True)
  (remote / "scripts/cloud/kaggle_source_manifest.json").write_text(
    json.dumps(
      {"task": cloud.TASK, "source_files": {"src/current.py": cloud.sha256(code)}}
    )
  )
  git("add", ".")
  git("commit", "-m", "runtime")
  first = git("rev-parse", "HEAD")
  (remote / "README.md").write_text("second commit")
  if tampered:
    code.write_text("# stale manifest\n")
  git("add", ".")
  git("commit", "-m", "branch tip")
  tip = git("rev-parse", "HEAD")
  output = tmp_path / "output"
  output.mkdir()
  namespace: dict[str, Any] = {
    "SOURCE": tmp_path / "checkout",
    "REPO_URL": remote.as_uri(),
    "REPO_BRANCH": branch,
    "REPO_COMMIT": first if pinned else None,
    "os": os,
    "subprocess": subprocess,
    "json": json,
    "digest": cloud.sha256,
    "DATA_FILES": {},
    "OUTPUT": output,
    "SESSION": "test",
  }
  cell = compile(notebook_cell("airl-github-clone"), "notebook_clone", "exec")
  if tampered:
    with pytest.raises(RuntimeError, match="Source hash mismatch"):
      exec(cell, namespace)
    return
  exec(cell, namespace)
  assert namespace["source_commit"] == (first if pinned else tip)
  assert (
    json.loads((output / "test_github_source.json").read_text())["commit"]
    == namespace["source_commit"]
  )
  with pytest.raises(FileExistsError):
    exec(cell, namespace)


@pytest.mark.parametrize("tampered", (False, True))
def test_notebook_rejects_wrong_kaggle_artifacts_before_clone(tmp_path, tampered):
  expert = tmp_path / "expert.pt"
  common = tmp_path / "common.pt"
  expert.write_bytes(b"expert")
  common.write_bytes(b"common")
  expected = {
    name: {"sha256": cloud.sha256(path), "bytes": path.stat().st_size}
    for name, path in (("expert", expert), ("common_ppo", common))
  }
  # Substitute only fixed dataset literals; execute the actual validation cell.
  tree = ast.parse(notebook_cell("airl-02"))
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(
      isinstance(target, ast.Name) and target.id == "DATA_FILES"
      for target in node.targets
    ):
      node.value = ast.parse(repr(expected), mode="eval").body
  ast.fix_missing_locations(tree)
  if tampered:
    expert.write_bytes(b"broken")
  namespace: dict[str, Any] = {
    "hashlib": hashlib,
    "json": json,
    "EXPERT": expert,
    "COMMON": common,
    "OUTPUT": tmp_path,
    "SESSION": "test",
  }
  cell = compile(tree, "notebook_artifacts", "exec")
  if tampered:
    with pytest.raises(RuntimeError, match="Wrong qualified input"):
      exec(cell, namespace)
  else:
    exec(cell, namespace)
    assert (tmp_path / "test_data_manifest.json").is_file()


@pytest.fixture
def pilot_request(tmp_path):
  input_dir = tmp_path / "input"
  input_dir.mkdir()
  (input_dir / cloud.DATASET).write_bytes(b"qualified expert")
  common = input_dir / cloud.CHECKPOINT
  common.write_bytes(b"common PPO checkpoint")
  selected = {
    "ok": True,
    "envs": 16384,
    "minibatches": 8,
    "gpu": "Tesla T4",
    "cuda": "12.8",
    "packages": {"torch": "2.9.0+cu128"},
    "total_device_mib": 14911.6875,
  }
  selected_file = tmp_path / "selected.json"
  selected_file.write_text(json.dumps(selected))
  args = SimpleNamespace(
    input=input_dir,
    checkpoint=common,
    selected=selected_file,
    output=tmp_path / "pilot_ppo",
    arm="ppo",
    gpu=0,
    seed=42,
    resume=False,
    transitions=23592960,
    reuse_completed=True,
    restart_incomplete=True,
  )
  return args, selected


def save_completed_pilot(args, selected):
  run_dir = args.output / f"{args.arm}_n16384_mb8_seed42"
  run_dir.mkdir(parents=True)
  (run_dir / "model_1058.pt").write_bytes(b"final checkpoint")
  row = {
    **selected,
    "arm": args.arm,
    "seed": args.seed,
    "checkpoint_sha256": cloud.sha256(args.checkpoint),
    "additional_updates": 60,
    "additional_transitions": args.transitions,
    "resume": False,
    "run_dir": str(run_dir),
  }
  cloud.write_json(args.output / "run_summary.json", row)
  return row, run_dir


@pytest.mark.parametrize("arm", ("ppo", "airl"))
def test_completed_legacy_pilot_is_reused_without_training(
  pilot_request, monkeypatch, arm
):
  args, selected = pilot_request
  args.arm = arm
  _, run_dir = save_completed_pilot(args, selected)
  summary = args.output / "run_summary.json"
  before = summary.read_bytes()

  def no_training(*args, **kwargs):
    pytest.fail("A completed arm must not start another worker")

  monkeypatch.setattr(cloud.subprocess, "run", no_training)
  cloud.train_arm(args)
  assert summary.read_bytes() == before
  assert (run_dir / "model_1058.pt").read_bytes() == b"final checkpoint"


@pytest.mark.parametrize(
  "fault", ("seed", "budget", "gpu", "checkpoint", "final", "source", "dataset", "path")
)
def test_completed_pilot_requires_matching_request_and_artifacts(pilot_request, fault):
  args, selected = pilot_request
  args.arm = "airl"
  row, run_dir = save_completed_pilot(args, selected)
  if fault == "seed":
    args.seed += 1
  elif fault == "budget":
    args.transitions *= 2
  elif fault == "gpu":
    row["gpu"] = "Different GPU"
  elif fault == "checkpoint":
    args.checkpoint.write_bytes(b"different initializer")
  elif fault == "final":
    row["final_checkpoint_sha256"] = "bad"
  elif fault == "source":
    row["source_manifest_sha256"] = "bad"
  elif fault == "dataset":
    row["dataset_sha256"] = "bad"
  else:
    row["run_dir"] = str(args.output.parent / "outside")
  cloud.write_json(args.output / "run_summary.json", row)
  with pytest.raises(ValueError):
    cloud.train_arm(args)
  assert (run_dir / "model_1058.pt").read_bytes() == b"final checkpoint"


def test_missing_final_checkpoint_is_not_treated_as_completed(pilot_request):
  args, selected = pilot_request
  _, run_dir = save_completed_pilot(args, selected)
  (run_dir / "model_1058.pt").unlink()
  with pytest.raises(FileNotFoundError, match="no checkpoint"):
    cloud.train_arm(args)


def test_incomplete_pilot_restarts_full_budget_and_preserves_old_run(
  pilot_request, monkeypatch
):
  args, selected = pilot_request
  old = args.output / "ppo_n16384_mb8_seed42/model_1000.pt"
  old.parent.mkdir(parents=True)
  old.write_bytes(b"partial checkpoint")
  (args.output / "attempt_001").mkdir()
  calls = []

  def run(command, **kwargs):
    calls.append(command)
    output = Path(command[command.index("--output") + 1])
    result = Path(command[command.index("--result") + 1])
    run_dir = output / "ppo_n16384_mb8_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "model_1058.pt").write_bytes(b"final")
    cloud.write_json(
      result,
      {
        **selected,
        "arm": "ppo",
        "seed": 42,
        "checkpoint_sha256": cloud.sha256(args.checkpoint),
        "additional_updates": 60,
        "additional_transitions": args.transitions,
        "resume": False,
        "run_dir": str(run_dir),
      },
    )

  monkeypatch.setattr(cloud.subprocess, "run", run)
  cloud.train_arm(args)
  assert len(calls) == 1
  command = calls[0]
  assert command[command.index("--output") + 1] == str(args.output / "attempt_002")
  assert command[command.index("--checkpoint") + 1] == str(args.checkpoint)
  assert command[command.index("--updates") + 1] == "60"
  assert "--resume" not in command
  assert old.read_bytes() == b"partial checkpoint"
  cloud.train_arm(args)  # Now the successful attempt is reused.
  assert len(calls) == 1


def test_incomplete_restart_and_completed_reuse_are_opt_in(pilot_request):
  args, selected = pilot_request
  args.output.mkdir()
  sentinel = args.output / "old.log"
  sentinel.write_bytes(b"old log")
  args.restart_incomplete = False
  with pytest.raises(FileExistsError, match="--restart-incomplete"):
    cloud.train_arm(args)
  assert sentinel.read_bytes() == b"old log"
  save_completed_pilot(args, selected)
  args.reuse_completed = False
  with pytest.raises(FileExistsError, match="--reuse-completed"):
    cloud.train_arm(args)


def test_worker_rejects_existing_run_before_importing_gpu_stack(tmp_path, monkeypatch):
  args = SimpleNamespace(output=tmp_path, arm="ppo", envs=16384, minibatches=8, seed=42)
  (tmp_path / "ppo_n16384_mb8_seed42").mkdir()
  original = builtins.__import__

  def forbid_gpu_import(name, *args, **kwargs):
    if name in ("torch", "warp"):
      pytest.fail("Output collision must be detected before CUDA/Warp initialization")
    return original(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", forbid_gpu_import)
  with pytest.raises(FileExistsError, match="run already exists"):
    cloud.worker(args)


def test_notebook_pilot_cell_enables_preserving_reruns(tmp_path):
  calls = []
  namespace = {
    "selected": {"median_update_seconds": 12, "rollout_transitions": 393216},
    "TRANSITIONS": 23592960,
    "OUTPUT": tmp_path,
    "SESSION": "pilot01",
    "INPUT": tmp_path / "input",
    "COMMON": tmp_path / "common.pt",
    "SELECTED": tmp_path / "selected.json",
    "GPU": 0,
    "cloud": lambda *args: calls.append(args),
  }
  exec(compile(notebook_cell("airl-11"), "pilot_cell", "exec"), namespace)
  assert len(calls) == 2
  for command in calls:
    assert "--reuse-completed" in command and "--restart-incomplete" in command
    assert command[command.index("--checkpoint") + 1] == namespace["COMMON"]
    assert command[command.index("--transitions") + 1] == namespace["TRANSITIONS"]
    assert "--resume" not in command


def test_train_cli_reuses_completed_arm_without_gpu_initialization(pilot_request):
  args, selected = pilot_request
  save_completed_pilot(args, selected)
  completed = subprocess.run(
    [
      sys.executable,
      str(cloud.ROOT / "scripts/cloud/kaggle_airl.py"),
      "train",
      "--input",
      str(args.input),
      "--checkpoint",
      str(args.checkpoint),
      "--selected",
      str(args.selected),
      "--output",
      str(args.output),
      "--arm",
      args.arm,
      "--reuse-completed",
      "--restart-incomplete",
    ],
    capture_output=True,
    text=True,
    check=True,
  )
  assert "Reusing completed ppo" in completed.stdout
  assert "Warp" not in completed.stdout
