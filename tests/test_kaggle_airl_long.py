"""Paired continuation accounting/recovery without simulator or GPU training."""

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
  "kaggle_airl_long", Path(__file__).parents[1] / "scripts/cloud/kaggle_airl.py"
)
assert spec is not None and spec.loader is not None
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)
ROLLOUT = 16384 * 24


@pytest.fixture
def setup(tmp_path, monkeypatch):
  inputs = tmp_path / "input"
  inputs.mkdir()
  (inputs / cloud.CHECKPOINT).write_bytes(b"common PPO999")
  (inputs / cloud.DATASET).write_bytes(b"expert")
  selected = tmp_path / "selected.json"
  runtime = {
    "gpu": "Tesla T4",
    "cuda": "12.8",
    "packages": {"torch": "2.9.0+cu128"},
    "total_device_mib": 14911.6875,
  }
  selection = {
    **runtime,
    "ok": True,
    "envs": 16384,
    "minibatches": 8,
    "median_update_seconds": 0.01,
  }
  selected.write_text(json.dumps(selection))
  monkeypatch.setattr(cloud, "source_manifest_hash", lambda: "source-hash")
  args = argparse.Namespace(
    input=inputs,
    checkpoint=inputs / cloud.CHECKPOINT,
    selected=selected,
    output=tmp_path / "long",
    state=None,
    transitions=2 * ROLLOUT,
    chunk_transitions=ROLLOUT,
    session_seconds=10000,
    gpu=0,
    seed=42,
  )
  calls = []

  def worker(job, timeout=None):
    calls.append(job)
    job.output.mkdir(parents=True)
    run = job.output / "run"
    run.mkdir()
    final = run / "model_1000.pt"
    final.write_bytes(job.checkpoint.read_bytes() + job.arm.encode())
    cloud.write_json(
      job.output / "run_summary.json",
      {
        **runtime,
        "ok": True,
        "arm": job.arm,
        "envs": 16384,
        "minibatches": 8,
        "seed": job.seed,
        "resume": job.resume,
        "checkpoint_sha256": cloud.sha256(job.checkpoint),
        "additional_updates": job.transitions // ROLLOUT,
        "additional_transitions": job.transitions,
        "source_manifest_sha256": "source-hash",
        "dataset_sha256": cloud.sha256(inputs / cloud.DATASET)
        if job.arm == "airl"
        else None,
        "run_dir": str(run),
        "final_checkpoint_sha256": cloud.sha256(final),
        "update_seconds": [0.01] * (job.transitions // ROLLOUT),
      },
    )

  monkeypatch.setattr(cloud, "train_arm", worker)
  return args, calls, worker


def saved(args):
  return json.loads((args.output / "paired_state.json").read_text())


def test_equal_budgets_resume_own_arm_and_completed_rerun(setup):
  args, calls, _ = setup
  cloud.paired(args)
  state = saved(args)
  assert state["complete"]
  assert [job.arm for job in calls] == ["ppo", "airl", "ppo", "airl"]
  assert [job.resume for job in calls] == [False, False, True, True]
  assert all(job.require_rng for job in calls)
  assert calls[0].checkpoint == calls[1].checkpoint
  assert calls[2].checkpoint.name.startswith("ppo_")
  assert calls[3].checkpoint.name.startswith("airl_")
  assert all(row["transitions"] == 2 * ROLLOUT for row in state["arms"].values())
  cloud.paired(args)
  assert len(calls) == 4


def test_failed_airl_preserves_ppo_and_next_run_catches_up(setup, monkeypatch):
  args, calls, worker = setup

  def fail(job, timeout=None):
    if job.arm == "airl":
      job.output.mkdir(parents=True)
      (job.output / "partial.pt").write_bytes(b"orphan")
      raise subprocess.CalledProcessError(1, ["worker"])
    worker(job, timeout)

  monkeypatch.setattr(cloud, "train_arm", fail)
  with pytest.raises(subprocess.CalledProcessError):
    cloud.paired(args)
  assert saved(args)["arms"]["ppo"]["transitions"] == ROLLOUT
  assert saved(args)["arms"]["airl"]["transitions"] == 0
  monkeypatch.setattr(cloud, "train_arm", worker)
  cloud.paired(args)
  assert [job.arm for job in calls] == ["ppo", "airl", "ppo", "airl"]
  assert calls[1].output.name == "attempt_002"
  assert saved(args)["complete"]


def test_state_portable_to_new_output_without_mutating_input(setup):
  args, calls, _ = setup
  args.session_seconds = 1
  cloud.paired(args)
  original = args.output
  hashes = {p.name: cloud.sha256(p) for p in original.iterdir() if p.is_file()}
  args.state = original / "paired_state.json"
  args.output = original.parent / "next_session"
  args.checkpoint = None
  args.selected = None
  args.session_seconds = 10000
  cloud.paired(args)
  assert saved(args)["complete"] and len(calls) == 4
  assert hashes == {p.name: cloud.sha256(p) for p in original.iterdir() if p.is_file()}
  assert all(
    Path(r["checkpoint"]).name == r["checkpoint"] for r in saved(args)["arms"].values()
  )


@pytest.mark.parametrize(
  "change",
  ["checkpoint", "dataset", "seed", "source", "history", "initializer", "selection"],
)
def test_changed_artifacts_or_contract_stop_before_new_training(
  setup, monkeypatch, change
):
  args, calls, _ = setup
  cloud.paired(args)
  state = saved(args)
  if change == "checkpoint":
    (args.output / state["arms"]["airl"]["checkpoint"]).write_bytes(b"changed")
  elif change == "dataset":
    (args.input / cloud.DATASET).write_bytes(b"changed")
  elif change == "seed":
    args.seed += 1
  elif change == "source":
    monkeypatch.setattr(cloud, "source_manifest_hash", lambda: "changed")
  elif change == "history":
    state["history"][0]["updates"] += 1
    cloud.write_json(args.output / "paired_state.json", state)
  elif change == "selection":
    selection = json.loads(args.selected.read_text())
    selection["median_update_seconds"] *= 2
    args.selected.write_text(json.dumps(selection))
  else:
    (args.output / "initial.pt").write_bytes(b"changed")
  with pytest.raises(ValueError):
    cloud.paired(args)
  assert len(calls) == 4


def test_single_writer_and_import_cannot_overwrite_existing_state(setup):
  args, calls, _ = setup
  args.output.mkdir()
  lock = args.output / ".paired.lock"
  lock.write_text("other writer")
  with pytest.raises(FileExistsError, match="writer"):
    cloud.paired(args)
  assert lock.read_text() == "other writer" and calls == []
  lock.unlink()
  cloud.paired(args)
  args.state = args.output.parent / "other_state.json"
  with pytest.raises(FileExistsError, match="new --output"):
    cloud.paired(args)


def test_timeout_does_not_advance_counter(setup, monkeypatch):
  args, _, _ = setup

  def timeout(job, timeout=None):
    assert timeout is not None
    raise subprocess.TimeoutExpired(["worker"], timeout)

  monkeypatch.setattr(cloud, "train_arm", timeout)
  cloud.paired(args)
  state = saved(args)
  assert not state["complete"]
  assert all(row["transitions"] == 0 for row in state["arms"].values())
  assert state["failed_attempts"][0]["error"] == "TimeoutExpired"


def test_invalid_chunk_summary_is_not_committed(setup, monkeypatch):
  args, _, worker = setup

  def incomplete(job, timeout=None):
    worker(job, timeout)
    path = job.output / "run_summary.json"
    row = json.loads(path.read_text())
    row["update_seconds"] = []
    cloud.write_json(path, row)

  monkeypatch.setattr(cloud, "train_arm", incomplete)
  with pytest.raises(ValueError, match="timing"):
    cloud.paired(args)
  assert saved(args)["arms"]["ppo"]["transitions"] == 0


def test_ppo_cpu_rng_restore_reproduces_next_sample():
  import torch

  original = torch.get_rng_state()
  try:
    torch.manual_seed(42005)
    state = torch.get_rng_state()
    expected = torch.randn(8)
    torch.randn(50)
    cloud.restore_ppo_rng({"kaggle_torch_rng": state}, "cpu")
    assert torch.equal(torch.randn(8), expected)
  finally:
    torch.set_rng_state(original)


def long_cell(cell_id):
  notebook = json.loads(
    (cloud.ROOT / "scripts/cloud/kaggle_airl_long.ipynb").read_text(encoding="utf8")
  )
  return "".join(next(c["source"] for c in notebook["cells"] if c["id"] == cell_id))


def test_long_notebook_cells_compile_and_launcher_uses_portable_state(tmp_path):
  notebook = json.loads(
    (cloud.ROOT / "scripts/cloud/kaggle_airl_long.ipynb").read_text(encoding="utf8")
  )
  for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
      compile("".join(cell["source"]), cell["id"], "exec")
      assert cell["outputs"] == [] and cell["execution_count"] is None
  calls = []
  pair = tmp_path / "pair"

  def run(*arguments):
    calls.append(arguments)
    cloud.write_json(
      pair / "paired_state.json",
      {
        "complete": False,
        "config": {"envs": 16384},
        "arms": {"ppo": {"transitions": ROLLOUT}, "airl": {"transitions": 0}},
      },
    )

  namespace: dict[str, Any] = {
    "cloud": run,
    "INPUT": tmp_path,
    "COMMON": tmp_path / "common.pt",
    "SELECTED": tmp_path / "selected.json",
    "PAIR": pair,
    "TRANSITIONS": 589824000,
    "CHUNK_TRANSITIONS": 23592960,
    "SESSION_SECONDS": 28800,
    "GPU": 0,
    "SEED": 42,
    "RESUME_STATE": tmp_path / "previous/paired_state.json",
    "Path": Path,
    "json": json,
  }
  exec(long_cell("long-10"), namespace)
  assert calls[0][0] == "paired" and "--state" in calls[0]
  exec(long_cell("long-10"), namespace)
  assert "--state" not in calls[1]
  assert calls[1][calls[1].index("--selected") + 1] == pair / "selected.json"
  namespace["OUTPUT"] = tmp_path
  namespace["SESSION"] = "test"
  namespace["evaluate"] = lambda *args: pytest.fail("Unequal endpoint budgets")
  exec(long_cell("long-12"), namespace)


def test_cloud_runner_saves_ppo_rng_and_advances_resume_index():
  import torch

  tree = ast.parse((cloud.ROOT / "scripts/cloud/kaggle_airl.py").read_text())
  runner = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.ClassDef) and node.name == "MeasuredRunner"
  )

  class Base:
    airl = None
    device = "cpu"
    current_learning_iteration = 999

    def save(self, path, infos):
      self.infos = infos

    def load(self):
      return "loaded"

  original = torch.get_rng_state()
  infos = {"kaggle_torch_rng": original, "kaggle_cuda_rng": original}
  namespace: dict[str, Any] = {
    "AirlVelocityOnPolicyRunner": Base,
    "args": SimpleNamespace(benchmark=False, resume=True),
    "torch": SimpleNamespace(
      get_rng_state=torch.get_rng_state,
      cuda=SimpleNamespace(get_rng_state=lambda device: original),
    ),
    "checkpoint_infos": infos,
    "PPO_RNG_KEYS": cloud.PPO_RNG_KEYS,
    "restore_ppo_rng": cloud.restore_ppo_rng,
  }
  exec(
    compile(ast.Module(body=[runner], type_ignores=[]), "MeasuredRunner", "exec"),
    namespace,
  )
  instance = namespace["MeasuredRunner"]()
  instance.save("checkpoint.pt", {"preserved": 123})
  assert instance.infos["preserved"] == 123
  assert all(key in instance.infos for key in cloud.PPO_RNG_KEYS)
  assert instance.load() == "loaded" and instance.current_learning_iteration == 1000
  assert torch.equal(torch.get_rng_state(), original)


def test_real_cli_can_pause_and_import_state_without_gpu(setup):
  args, calls, _ = setup
  script = cloud.ROOT / "scripts/cloud/kaggle_airl.py"
  subprocess.run(
    [
      sys.executable,
      str(script),
      "paired",
      "--input",
      str(args.input),
      "--selected",
      str(args.selected),
      "--checkpoint",
      str(args.checkpoint),
      "--output",
      str(args.output),
      "--transitions",
      str(args.transitions),
      "--chunk-transitions",
      str(args.chunk_transitions),
      "--session-seconds",
      "1",
    ],
    check=True,
  )
  assert saved(args)["complete"] is False and calls == []
  prior = args.output
  args.output = prior.parent / "cli_import"
  subprocess.run(
    [
      sys.executable,
      str(script),
      "paired",
      "--input",
      str(args.input),
      "--state",
      str(prior / "paired_state.json"),
      "--output",
      str(args.output),
      "--transitions",
      str(args.transitions),
      "--chunk-transitions",
      str(args.chunk_transitions),
      "--session-seconds",
      "1",
    ],
    check=True,
  )
  assert saved(args)["arms"]["ppo"]["transitions"] == 0
  assert (args.output / "initial.pt").read_bytes() == args.checkpoint.read_bytes()
