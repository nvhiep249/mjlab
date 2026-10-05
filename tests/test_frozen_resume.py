import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.rl.gail import GailDiscriminator
from mjlab.tasks.velocity.rl.gail_runner import GailVelocityOnPolicyRunner
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


def test_explicit_frozen_resume_checks_D_before_loading_PPO(tmp_path, monkeypatch):
  runner: Any = object.__new__(GailVelocityOnPolicyRunner)
  runner.gail = SimpleNamespace(discriminator=GailDiscriminator(4), frozen=True)
  state = runner.gail.discriminator.state_dict()
  path = tmp_path / "resume.pt"
  torch.save({"infos": {"gail_state_dict": state}}, path)
  calls = []
  monkeypatch.setattr(
    VelocityOnPolicyRunner, "load", lambda *a, **kw: calls.append(a) or {}
  )
  runner.load(str(path), allow_frozen_resume=True)
  assert len(calls) == 1
  assert runner.gail.frozen
  bad = {k: v.clone() for k, v in state.items()}
  bad["feature_mean"].add_(1)
  torch.save({"infos": {"gail_state_dict": bad}}, path)
  with pytest.raises(ValueError, match="discriminator mismatch"):
    runner.load(str(path), allow_frozen_resume=True)
  assert len(calls) == 1
  torch.save({"infos": {}}, path)
  with pytest.raises(ValueError, match="discriminator mismatch"):
    runner.load(str(path), allow_frozen_resume=True)
  assert len(calls) == 1


def test_frozen_extension_replays_recorded_budget_and_curriculum():
  path = Path(__file__).resolve().parents[1] / "scripts/training/extend_frozen_gail.py"
  spec = importlib.util.spec_from_file_location("extend_frozen_gail", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  cfg = module.build_frozen_config(
    "phase2", Path("model_199.pt"), Path("logs"), Path("D1200.pt")
  )
  final = module.build_frozen_config(
    "final", Path("model_999.pt"), Path("logs"), Path("D1200.pt")
  )
  assert cfg.agent.max_iterations == 801
  assert final.agent.max_iterations == 500
  assert 199 + cfg.agent.max_iterations - 1 == 999
  assert 999 + final.agent.max_iterations - 1 == 1498
  assert cfg.env.commands["twist"].target_speed_curriculum_stages == (
    (0, (0.3, 0.5, 0.8)),
    (12000, (0.6, 0.9, 1.2)),
    (24000, (1.0, 1.1, 1.2)),
  )
  assert final.env.commands["twist"].target_speed_curriculum_stages == (
    (0, (1.0, 1.1, 1.2)),
  )
  for c in (cfg, final):
    assert c.agent.gail.frozen and c.agent.gail.weight == 0.01
    assert c.agent.seed == c.env.seed == 42
    assert c.env.scene.num_envs == 16384
    assert c.agent.num_steps_per_env == 24
