from unittest.mock import Mock

import torch

from mjlab.scripts.gail_evaluate_expert import (
  ExpertGate,
  build_command_grid,
  classify_successful_episodes,
  evaluate_gate,
  failure_breakdown,
  refresh_fixed_command_observation,
  set_fixed_commands,
)


def test_command_grid_is_complete_and_deterministic():
  grid = build_command_grid(
    lin_vel_x=(-2.0, 0.0, 3.0),
    lin_vel_y=(-1.0, 0.0, 1.0),
    ang_vel_z=(-0.7, 0.0, 0.7),
  )

  assert len(grid) == 27
  assert grid[0] == (-2.0, -1.0, -0.7)
  assert grid[-1] == (3.0, 1.0, 0.7)


def test_fixed_commands_pin_at_native_command_boundary():
  term = Mock()
  commands = torch.tensor([[1.2, 0.0, 0.0]])
  set_fixed_commands(term, commands)
  term.pin_commands.assert_called_once_with(commands)


def test_gate_fails_when_any_command_bin_is_unsafe():
  bins = [
    {
      "success_rate": 0.99,
      "linear_rmse": 0.15,
      "yaw_rmse": 0.1,
      "upright_mean": 0.98,
    },
    {
      "success_rate": 0.90,
      "linear_rmse": 0.2,
      "yaw_rmse": 0.1,
      "upright_mean": 0.98,
    },
  ]

  result = evaluate_gate(bins, ExpertGate())

  assert result["passed"] is False
  assert result["failed_bins"] == [1]
  assert "success_rate" in result["failures"]["1"]


def test_gate_passes_only_when_every_bin_meets_thresholds():
  bins = [
    {
      "success_rate": 0.96,
      "linear_rmse": 0.24,
      "yaw_rmse": 0.19,
      "upright_mean": 0.971,
    }
  ]

  result = evaluate_gate(bins, ExpertGate())

  assert result == {"passed": True, "failed_bins": [], "failures": {}}


def test_timeout_is_not_success_when_policy_does_not_track_command():
  successful = classify_successful_episodes(
    timed_out=torch.tensor([True, True, False]),
    linear_rmse=torch.tensor([1.2, 0.20, 0.10]),
    yaw_rmse=torch.tensor([0.10, 0.10, 0.10]),
    upright_mean=torch.tensor([0.99, 0.99, 0.99]),
    gate=ExpertGate(),
  )

  torch.testing.assert_close(successful, torch.tensor([False, True, False]))


def test_failure_breakdown_keeps_overlapping_tracking_failures():
  breakdown = failure_breakdown(
    timed_out=torch.tensor([True, True, False]),
    linear_rmse=torch.tensor([0.30, 0.10, 0.10]),
    yaw_rmse=torch.tensor([0.10, 0.30, 0.10]),
    upright_mean=torch.tensor([0.99, 0.99, 0.90]),
    gate=ExpertGate(),
  )

  assert breakdown == {
    "success": 0,
    "fall": 1,
    "linear_rmse": 1,
    "yaw_rmse": 1,
    "upright": 1,
  }


def test_refresh_fixed_command_observation_recomputes_after_override():
  class FakeEnv:
    def __init__(self):
      self.calls = 0

    def get_observations(self):
      self.calls += 1
      return "fresh-observation"

  env = FakeEnv()
  result = refresh_fixed_command_observation(env)

  assert result == "fresh-observation"
  assert env.calls == 1
