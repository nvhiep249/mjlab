import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

import mjlab.tasks  # noqa: F401
from mjlab.scripts.gail_collect_expert import (
  CollectConfig,
  _activate_curriculum_step,
  _balanced_commands,
  _checkpoint_sha256,
  _collection_batch,
  _load_collection_env_cfg,
  _target_speed_commands,
  _trim_and_merge,
  collect,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel


@pytest.mark.parametrize("case", ["hash", "coverage", "legacy"])
def test_forward_collection_rejects_unqualified_provenance(tmp_path, case):
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.write_bytes(b"test-checkpoint")
  report = tmp_path / "gate.json"
  report.write_text(
    json.dumps(
      {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": "wrong"
        if case == "hash"
        else _checkpoint_sha256(checkpoint),
        "gate": {"passed": True},
        "bins": [
          {
            "command": [1.2 if case == "coverage" else 0.3, 0, 0],
            **({} if case == "legacy" else {"fall_rate": 0, "timeout_rate": 1}),
          }
        ],
      }
    )
  )
  with pytest.raises(ValueError, match="SHA-256|not covered"):
    collect(
      CollectConfig(
        checkpoint_file=checkpoint,
        output_path=tmp_path / "expert.pt",
        gate_report=report,
        forward_speeds=(0.3,),
      )
    )


def test_airl_collection_rejects_deterministic_gate_before_environment(tmp_path):
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.write_bytes(b"test-checkpoint")
  report = tmp_path / "gate.json"
  report.write_text(
    json.dumps(
      {
        "checkpoint": str(checkpoint.resolve()),
        "gate": {"passed": True},
        "expert_policy": "deterministic",
      }
    )
  )
  with pytest.raises(ValueError, match="expert policy mode"):
    collect(
      CollectConfig(
        checkpoint_file=checkpoint,
        output_path=tmp_path / "expert.pt",
        gate_report=report,
        airl_compatible=True,
      )
    )


@pytest.mark.parametrize("stochastic", [False, True])
def test_airl_collection_accepts_matching_policy_gate(
  tmp_path, monkeypatch, stochastic
):
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.write_bytes(b"test-checkpoint")
  report = tmp_path / "gate.json"
  report.write_text(
    json.dumps(
      {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "gate": {"passed": True},
        "expert_policy": "stochastic" if stochastic else "deterministic",
        "config": {"stochastic_policy": stochastic},
      }
    )
  )

  def stop_before_environment(*args):
    raise RuntimeError("matching gate accepted")

  monkeypatch.setattr(
    "mjlab.scripts.gail_collect_expert._load_collection_env_cfg",
    stop_before_environment,
  )
  with pytest.raises(RuntimeError, match="matching gate accepted"):
    collect(
      CollectConfig(
        checkpoint_file=checkpoint,
        output_path=tmp_path / "expert.pt",
        gate_report=report,
        airl_compatible=True,
        stochastic_expert=stochastic,
      )
    )


def test_trim_and_merge_keeps_exact_transition_count():
  batches = [
    {
      "observations": torch.full((3, 2), float(batch)),
      "actions": torch.full((3, 1), float(batch)),
      "commands": torch.full((3, 3), float(batch)),
    }
    for batch in range(2)
  ]

  result = _trim_and_merge(batches, count=5)

  assert {name: tuple(value.shape) for name, value in result.items()} == {
    "observations": (5, 2),
    "actions": (5, 1),
    "commands": (5, 3),
  }
  assert result["observations"][-1].tolist() == [1.0, 1.0]


def test_airl_collection_rejects_clipping_before_simulation(tmp_path, monkeypatch):
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.write_bytes(b"test-checkpoint")
  monkeypatch.setattr(
    "mjlab.scripts.gail_collect_expert.load_rl_cfg",
    lambda _: SimpleNamespace(clip_actions=1.0),
  )
  with pytest.raises(ValueError, match="unclipped raw policy actions"):
    collect(
      CollectConfig(
        checkpoint_file=checkpoint,
        output_path=tmp_path / "expert.pt",
        airl_compatible=True,
      )
    )


def test_airl_batch_preserves_timeout_and_termination_independently():
  result = _collection_batch(
    observations=torch.zeros(2, 68),
    next_observations=torch.ones(2, 68),
    actions=torch.zeros(2, 29),
    commands=torch.zeros(2, 3),
    next_commands=torch.ones(2, 3),
    actor_observations=torch.zeros(2, 96),
    next_actor_observations=torch.ones(2, 96),
    dones=torch.ones(2, dtype=torch.bool),
    time_outs=torch.tensor([True, False]),
    terminated=torch.tensor([True, True]),
    environment_ids=torch.arange(2),
    episode_ids=torch.arange(2),
  )
  assert result["terminated"].tolist() == [True, True]
  assert result["truncated"].tolist() == [True, False]
  assert result["next_actor_observations"].shape == (2, 96)
  assert result["next_commands"].tolist() == [[1, 1, 1], [1, 1, 1]]


def test_collection_batch_keeps_aligned_state_transitions():
  result = _collection_batch(
    observations=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    next_observations=torch.tensor([[1.5, 2.5], [30.0, 40.0]]),
    actions=torch.tensor([[0.1], [0.2]]),
    commands=torch.tensor([[1.2, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    dones=torch.tensor([False, True]),
    time_outs=torch.tensor([False, False]),
    environment_ids=torch.tensor([0, 1]),
    episode_ids=torch.tensor([5, 7]),
  )

  torch.testing.assert_close(result["observations"][0], torch.tensor([1.0, 2.0]))
  torch.testing.assert_close(result["next_observations"][0], torch.tensor([1.5, 2.5]))
  assert result["dones"].tolist() == [False, True]
  assert result["terminated"].tolist() == [False, True]


def test_collection_uses_training_command_ranges():
  cfg = _load_collection_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  ranges = cast(UniformVelocityCommandCfg, cfg.commands["twist"]).ranges

  assert ranges.lin_vel_x == (-1.0, 1.0)
  assert ranges.lin_vel_y == (-1.0, 1.0)
  assert ranges.ang_vel_z == (-0.5, 0.5)


def test_velocity_curriculum_maps_boundary_steps_to_expected_ranges():
  ranges = SimpleNamespace(
    lin_vel_x=(0.0, 0.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(0.0, 0.0)
  )
  env = SimpleNamespace(
    common_step_counter=0,
    command_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    ),
  )
  stages = [
    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {"step": 120_000, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    {"step": 240_000, "lin_vel_x": (-2.0, 3.0)},
  ]

  for step, expected_x, expected_yaw in [
    (0, (-1.0, 1.0), (-0.5, 0.5)),
    (120_000, (-1.5, 2.0), (-0.7, 0.7)),
    (240_000, (-2.0, 3.0), (-0.7, 0.7)),
  ]:
    env.common_step_counter = step
    metrics = commands_vel(
      cast(Any, env),
      torch.empty(0, dtype=torch.long),
      "twist",
      cast(Any, stages),
    )
    assert ranges.lin_vel_x == expected_x
    assert ranges.ang_vel_z == expected_yaw
    assert metrics["common_step_counter"].item() == step


def test_velocity_curriculum_reports_active_target_speed_stage():
  cfg = SimpleNamespace(
    ranges=SimpleNamespace(
      lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
    ),
    target_speed_command_grid=True,
    target_speed_curriculum=True,
    target_speed_curriculum_stages=(
      (0, (0.3, 0.5, 0.8)),
      (4_800, (0.6, 0.9, 1.2)),
    ),
  )
  env = SimpleNamespace(
    common_step_counter=4_800,
    command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(cfg=cfg)),
  )

  metrics = commands_vel(
    cast(Any, env),
    torch.empty(0, dtype=torch.long),
    "twist",
    cast(Any, [{"step": 0}]),
  )

  assert metrics["target_speed_min"].item() == pytest.approx(0.6)
  assert metrics["target_speed_max"].item() == pytest.approx(1.2)


def test_activate_curriculum_step_recomputes_before_reset():
  calls: list[str] = []
  inner = SimpleNamespace(
    common_step_counter=0,
    curriculum_manager=SimpleNamespace(compute=lambda: calls.append("compute")),
  )
  env = SimpleNamespace(
    unwrapped=inner,
    reset=Mock(side_effect=lambda: calls.append("reset")),
  )

  _activate_curriculum_step(cast(Any, env), 120_000)

  assert inner.common_step_counter == 120_000
  assert calls == ["compute", "reset"]


def test_balanced_commands_cover_range_boundaries():
  commands = _balanced_commands(
    {
      "lin_vel_x": (-2.0, 3.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-0.7, 0.7),
    },
    num_envs=10,
    bins=(2, 2, 2),
    device="cpu",
  )

  assert commands.shape == (10, 3)
  torch.testing.assert_close(
    commands[:8],
    torch.tensor(
      [
        [-2.0, -1.0, -0.7],
        [-2.0, -1.0, 0.7],
        [-2.0, 1.0, -0.7],
        [-2.0, 1.0, 0.7],
        [3.0, -1.0, -0.7],
        [3.0, -1.0, 0.7],
        [3.0, 1.0, -0.7],
        [3.0, 1.0, 0.7],
      ]
    ),
  )
  torch.testing.assert_close(commands[8:], commands[:2])


def test_target_speed_commands_repeat_the_qualified_gate_grid():
  commands = _target_speed_commands(num_envs=8, device="cpu")

  assert commands.shape == (8, 3)
  torch.testing.assert_close(
    commands,
    torch.tensor(
      [
        [1.2, 0.0, 0.0],
        [1.35, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.35, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.35, 0.0, 0.0],
      ]
    ),
  )
