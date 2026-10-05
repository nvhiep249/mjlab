from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.rl.gail import CommandMatchedSampler
from mjlab.scripts.gail_curriculum_comparison import (
  _phase2_iterations,
  build_config,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl.gail_runner import GailVelocityOnPolicyRunner


def test_matching_preserves_policy_command_frequencies_across_stage_boundary():
  expert = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [1.2, 0, 0]])
  policy = expert[torch.tensor([2, 0, 2, 2, 1])]
  sampler = CommandMatchedSampler(expert)
  indices = sampler.sample(policy)
  torch.testing.assert_close(expert[indices], policy)
  assert indices.dtype == torch.long


def test_matching_rejects_missing_expert_command():
  sampler = CommandMatchedSampler(torch.tensor([[1.2, 0, 0]]))
  with pytest.raises(ValueError, match="Missing expert command"):
    sampler.sample(torch.tensor([[0.3, 0, 0]]))


def test_matching_uses_all_rows_in_bin():
  sampler = CommandMatchedSampler(torch.tensor([[0.3, 0, 0]]).repeat(3, 1))
  torch.manual_seed(42)
  indices = sampler.sample(torch.tensor([[0.3, 0, 0]]).repeat(100, 1))
  assert set(indices.tolist()) == {0, 1, 2}


def test_comparison_replays_original_stop_and_resume_curriculum(tmp_path):
  first = build_config(1, tmp_path)
  second = build_config(2, tmp_path, Path("gail/model_199.pt"))
  assert first.checkpoint_file is None
  assert first.agent.max_iterations == 200
  assert second.agent.max_iterations == 801
  first_command = first.env.commands["twist"]
  second_command = second.env.commands["twist"]
  assert isinstance(first_command, UniformVelocityCommandCfg)
  assert isinstance(second_command, UniformVelocityCommandCfg)
  assert isinstance(first.agent, RslRlOnPolicyRunnerCfg)
  assert first_command.target_speed_curriculum_stages[1][0] == 4800
  assert second_command.target_speed_curriculum_stages[1][0] == 12000
  assert second_command.target_speed_curriculum_stages[2][0] == 24000
  assert first.agent.gail.match_expert_commands
  assert first.env.scene.num_envs == second.env.scene.num_envs == 16384


def test_comparison_rejects_warm_start_for_first_segment(tmp_path):
  with pytest.raises(ValueError, match="from scratch"):
    build_config(1, tmp_path, Path("ppo/model_999.pt"))


def test_comparison_resumes_interrupted_second_segment_to_label_999():
  assert _phase2_iterations({"iter": 199, "infos": {"gail_state_dict": {}}}) == 801
  assert _phase2_iterations({"iter": 500, "infos": {"gail_state_dict": {}}}) == 500


def test_comparison_rejects_non_gail_or_completed_second_segment():
  with pytest.raises(ValueError, match="GAIL checkpoint"):
    _phase2_iterations({"iter": 500, "infos": {}})
  with pytest.raises(ValueError, match="before model_999"):
    _phase2_iterations({"iter": 999, "infos": {"gail_state_dict": {}}})


def test_rollout_commands_are_snapshots_when_env_resamples(monkeypatch):
  command = torch.tensor([[0.3, 0.0, 0.0]])
  monkeypatch.setattr(
    "mjlab.tasks.velocity.rl.gail_runner.velocity_gail_state",
    lambda _: torch.zeros(1, 68),
  )
  runner = object.__new__(GailVelocityOnPolicyRunner)
  monkeypatch.setattr(
    runner,
    "env",
    SimpleNamespace(
      unwrapped=SimpleNamespace(
        scene={"robot": object()},
        command_manager=SimpleNamespace(get_command=lambda _: command),
      )
    ),
    raising=False,
  )
  monkeypatch.setattr(
    runner,
    "gail",
    SimpleNamespace(expert_commands=command, command_name="twist"),
    raising=False,
  )
  batch = runner._state_action(torch.zeros(1, 29))
  command.fill_(1.2)
  torch.testing.assert_close(batch.commands, torch.tensor([[0.3, 0.0, 0.0]]))
