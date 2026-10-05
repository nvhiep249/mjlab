from types import SimpleNamespace

import pytest
import torch

from mjlab.rl.config import GailCfg
from mjlab.rl.gail import GailBatch
from mjlab.tasks.velocity.rl.gail_runner import (
  GailVelocityOnPolicyRunner,
  guarded_frozen_reward,
)


def test_guard_caps_and_rejects_bad_pelvis_or_yaw():
  state = torch.zeros(4, 68)
  state[:, 3] = -1
  state[1, 1] = 0.35
  state[2, 9] = 0.4
  commands = torch.zeros(4, 3)
  commands[3, 2] = 0.4
  state[3, 9] = 0.4
  batch = GailBatch(state, torch.zeros(4, 29), commands)
  result = guarded_frozen_reward(torch.full((4,), 100.0), batch, 4.0)
  assert result.tolist() == pytest.approx(
    [4, 0, 4 * torch.exp(torch.tensor(-4)).item(), 4]
  )
  assert torch.equal(state[:, 3], torch.full((4,), -1.0))
  batch.commands = None
  with pytest.raises(ValueError, match="commands"):
    guarded_frozen_reward(torch.ones(4), batch, 4)


@pytest.mark.parametrize("cap", [-1, 0, float("nan"), float("inf")])
def test_invalid_guard_cap(cap):
  with pytest.raises(ValueError, match="cap"):
    GailCfg(
      enabled=True,
      dataset_path="x",
      frozen=True,
      discriminator_checkpoint="x",
      frozen_reward_cap=cap,
    ).validate()


def test_resume_rejects_changed_guard_before_ppo_load(tmp_path, monkeypatch):
  runner = object.__new__(GailVelocityOnPolicyRunner)
  monkeypatch.setattr(
    runner,
    "gail",
    SimpleNamespace(frozen=True, weight=0.0002, reward_cap=4.0),
    raising=False,
  )
  path = tmp_path / "resume.pt"
  torch.save({"infos": {"frozen_reward_settings": {"weight": 0.01, "cap": None}}}, path)
  with pytest.raises(ValueError, match="reward settings mismatch"):
    runner.load(str(path), allow_frozen_resume=True)


def test_paired_comparison_only_changes_imitation(monkeypatch):
  import importlib.util
  from dataclasses import asdict
  from pathlib import Path

  folder = Path(__file__).resolve().parents[1] / "scripts/training"
  monkeypatch.syspath_prepend(str(folder))
  spec = importlib.util.spec_from_file_location(
    "guarded_comparison", folder / "run_guarded_frozen_comparison.py"
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  for stage, budget in (("pilot", 200), ("phase2", 801), ("final", 500)):
    checkpoint = None if stage == "pilot" else Path("model.pt")
    ppo = module.configuration("ppo", stage, Path("logs"), checkpoint)
    guarded = module.configuration("guarded", stage, Path("logs"), checkpoint)
    assert ppo.agent.max_iterations == guarded.agent.max_iterations == budget
    assert guarded.agent.gail.weight == 0.0002
    assert guarded.agent.gail.frozen_reward_cap == module.CAP
    a, b = asdict(ppo), asdict(guarded)
    for data in (a, b):
      data["agent"].pop("gail")
      data["agent"].pop("experiment_name")
    assert a == b
