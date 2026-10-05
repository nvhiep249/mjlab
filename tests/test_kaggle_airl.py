"""Hardware-free checks for the cloud selection and matched sample budgets."""

import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
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
