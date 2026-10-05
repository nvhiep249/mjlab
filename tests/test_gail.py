from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.rl.config import GailCfg
from mjlab.rl.gail import (
  VELOCITY_GAIL_FEATURE_SCHEMA,
  GailBatch,
  GailDiscriminator,
  GailTransitionDataset,
  discriminator_loss,
  imitation_reward,
  velocity_gail_state,
)
from mjlab.tasks.velocity.rl.gail_runner import (
  GailVelocityOnPolicyRunner,
  _make_expert_batch,
)
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


def _batch(value: float, size: int = 8) -> GailBatch:
  return GailBatch(
    observations=torch.full((size, 2), value),
    actions=torch.full((size, 1), value),
    commands=torch.full((size, 1), value),
  )


def test_batch_features_concatenate_observation_action_and_command():
  batch = _batch(1.0, size=3)

  assert batch.features().shape == (3, 4)
  assert torch.all(batch.features() == 1.0)


def test_batch_features_support_state_transition_and_command():
  batch = GailBatch(
    observations=torch.tensor([[1.0, 2.0]]),
    actions=None,
    commands=torch.tensor([[1.2]]),
    next_observations=torch.tensor([[3.0, 4.0]]),
  )

  torch.testing.assert_close(
    batch.features(), torch.tensor([[1.0, 2.0, 3.0, 4.0, 1.2]])
  )


def test_batch_features_require_exactly_one_transition_target():
  with pytest.raises(ValueError, match="exactly one"):
    GailBatch(torch.zeros(1, 2), actions=None).features()
  with pytest.raises(ValueError, match="exactly one"):
    GailBatch(
      torch.zeros(1, 2),
      actions=torch.zeros(1, 1),
      next_observations=torch.zeros(1, 2),
    ).features()


def test_velocity_gail_state_is_body_local_and_translation_invariant():
  data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[10.0, -20.0, 0.75]]),
    projected_gravity_b=torch.tensor([[0.1, 0.2, -0.97]]),
    root_link_lin_vel_b=torch.tensor([[1.0, 2.0, 3.0]]),
    root_link_ang_vel_b=torch.tensor([[0.4, 0.5, 0.6]]),
    joint_pos=torch.arange(29, dtype=torch.float32).unsqueeze(0),
    default_joint_pos=torch.ones(1, 29),
    joint_vel=torch.arange(29, dtype=torch.float32).unsqueeze(0) * 0.1,
  )
  robot = SimpleNamespace(data=data)

  original = velocity_gail_state(robot)
  data.root_link_pos_w[:, :2] = torch.tensor([[500.0, -700.0]])
  translated = velocity_gail_state(robot)

  assert original.shape == (1, 68)
  torch.testing.assert_close(original, translated)
  torch.testing.assert_close(original[:, :1], torch.tensor([[0.75]]))
  torch.testing.assert_close(
    original[:, 10:39], data.joint_pos - data.default_joint_pos
  )


def test_discriminator_normalizes_features_with_expert_statistics():
  expert_features = torch.tensor(
    [[10.0, 1.0, -2.0], [14.0, 3.0, 2.0], [12.0, 2.0, 0.0]]
  )
  discriminator = GailDiscriminator.from_expert_features(
    expert_features, hidden_dims=(8,)
  )

  normalized = discriminator.normalize_features(expert_features)

  torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
  torch.testing.assert_close(
    normalized.std(dim=0, unbiased=False), torch.ones(3), atol=1e-6, rtol=0
  )


def test_discriminator_loss_learns_to_separate_easy_batches():
  torch.manual_seed(0)
  discriminator = GailDiscriminator(input_dim=4, hidden_dims=(16,))
  optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.05)
  expert = _batch(1.0)
  policy = _batch(-1.0)

  initial_loss, _ = discriminator_loss(discriminator, expert, policy)
  for _ in range(100):
    optimizer.zero_grad()
    loss, _ = discriminator_loss(discriminator, expert, policy)
    loss.backward()
    optimizer.step()
  final_loss, metrics = discriminator_loss(discriminator, expert, policy)

  assert final_loss < initial_loss
  assert metrics["accuracy"] > 0.95


def test_imitation_reward_is_finite_and_non_saturating():
  discriminator = GailDiscriminator(input_dim=4, hidden_dims=(8,))
  policy = _batch(0.0)

  reward = imitation_reward(discriminator, policy, normalize=True)

  assert reward.shape == (8,)
  assert torch.isfinite(reward).all()
  assert abs(float(reward.mean())) < 1e-5


def test_transition_dataset_validates_and_returns_float32_samples():
  dataset = GailTransitionDataset(
    {
      "observations": torch.ones(4, 2, dtype=torch.float64),
      "actions": torch.zeros(4, 1, dtype=torch.float64),
      "commands": torch.ones(4, 1, dtype=torch.float64),
    }
  )

  assert len(dataset) == 4
  assert dataset[0]["observations"].dtype == torch.float32
  assert set(dataset[0]) == {"observations", "actions", "commands"}


def test_transition_dataset_loads_state_only_next_observations(tmp_path):
  path = tmp_path / "transition_expert.pt"
  torch.save(
    {
      "observations": torch.zeros(4, 68),
      "next_observations": torch.ones(4, 68),
      "commands": torch.full((4, 3), 1.2),
      "metadata": {"feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA},
    },
    path,
  )

  dataset = GailTransitionDataset.load(path)

  assert set(dataset[0]) == {
    "observations",
    "next_observations",
    "commands",
  }
  assert dataset[0]["next_observations"].dtype == torch.float32


def test_transition_dataset_requires_action_or_next_observation():
  with pytest.raises(ValueError, match="actions.*next_observations"):
    GailTransitionDataset({"observations": torch.zeros(4, 68)})


def test_transition_expert_batch_excludes_episode_boundaries():
  dataset = GailTransitionDataset(
    {
      "observations": torch.tensor([[1.0], [2.0], [3.0]]),
      "next_observations": torch.tensor([[1.5], [20.0], [3.5]]),
      "commands": torch.tensor([[1.2], [1.2], [1.2]]),
      "dones": torch.tensor([False, True, False]),
    }
  )

  batch = _make_expert_batch(dataset, "state_transition")

  assert batch.actions is None
  assert batch.observations.flatten().tolist() == [1.0, 3.0]
  assert batch.next_observations is not None
  assert batch.next_observations.flatten().tolist() == [1.5, 3.5]


def test_transition_expert_batch_requires_next_observations():
  dataset = GailTransitionDataset(
    {
      "observations": torch.zeros(3, 2),
      "actions": torch.zeros(3, 1),
    }
  )

  with pytest.raises(ValueError, match="next_observations"):
    _make_expert_batch(dataset, "state_transition")


def test_transition_dataset_preserves_uint8_images():
  images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
  dataset = GailTransitionDataset(
    {
      "observations": torch.ones(4, 2),
      "actions": torch.zeros(4, 1),
      "images": images,
      "environment_ids": torch.tensor([0, 1, 0, 1]),
      "episode_ids": torch.tensor([0, 0, 0, 0]),
    },
    metadata={"image_layout": "NHWC"},
  )

  assert dataset[0]["images"].dtype == torch.uint8
  assert dataset._data["images"].data_ptr() == images.data_ptr()


def test_transition_dataset_builds_frame_stack_without_episode_leakage():
  images = torch.arange(6, dtype=torch.uint8).reshape(6, 1, 1, 1)
  dataset = GailTransitionDataset(
    {
      "observations": torch.ones(6, 2),
      "actions": torch.zeros(6, 1),
      "images": images,
      "environment_ids": torch.tensor([0, 1, 0, 1, 0, 1]),
      "episode_ids": torch.tensor([0, 0, 0, 0, 1, 0]),
    },
    metadata={"image_layout": "NHWC"},
  )

  env0_before_reset = dataset.image_stack(2, stack_size=3)
  env0_after_reset = dataset.image_stack(4, stack_size=3)
  env1 = dataset.image_stack(5, stack_size=3)

  assert env0_before_reset[:, 0, 0, 0].tolist() == [0, 0, 2]
  assert env0_after_reset[:, 0, 0, 0].tolist() == [4, 4, 4]
  assert env1[:, 0, 0, 0].tolist() == [1, 3, 5]
  assert env0_after_reset.dtype == torch.uint8


def test_transition_dataset_normalizes_only_sampled_image_batch():
  images = torch.tensor([[[[0, 127, 255]]]], dtype=torch.uint8)
  dataset = GailTransitionDataset(
    {
      "observations": torch.ones(1, 2),
      "actions": torch.zeros(1, 1),
      "images": images,
      "environment_ids": torch.tensor([0]),
      "episode_ids": torch.tensor([0]),
    },
    metadata={"image_layout": "NHWC"},
  )

  batch = dataset.normalized_image_batch(torch.tensor([0]), stack_size=1)

  assert batch.dtype == torch.float32
  torch.testing.assert_close(
    batch,
    torch.tensor([[[[[0.0]], [[127 / 255]], [[1.0]]]]]),
  )
  assert dataset._data["images"].dtype == torch.uint8


def test_transition_dataset_splits_complete_episodes_by_command_bin():
  dataset = GailTransitionDataset(
    {
      "observations": torch.arange(24, dtype=torch.float32).reshape(8, 3),
      "actions": torch.zeros(8, 1),
      "commands": torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]
      ),
      "environment_ids": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
      "episode_ids": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    }
  )

  train, validation = dataset.split_episode_stratified(0.5, seed=7)

  assert len(train) == 4
  assert len(validation) == 4
  train_groups = set(
    zip(
      train._data["environment_ids"].tolist(),
      train._data["episode_ids"].tolist(),
      strict=True,
    )
  )
  validation_groups = set(
    zip(
      validation._data["environment_ids"].tolist(),
      validation._data["episode_ids"].tolist(),
      strict=True,
    )
  )
  assert train_groups.isdisjoint(validation_groups)
  assert set(train._data["commands"].flatten().tolist()) == {0.0, 1.0}
  assert set(validation._data["commands"].flatten().tolist()) == {0.0, 1.0}


def test_transition_dataset_rejects_mismatched_lengths():
  try:
    GailTransitionDataset(
      {"observations": torch.zeros(2, 1), "actions": torch.zeros(3, 1)}
    )
  except ValueError as exc:
    assert "same number" in str(exc)
  else:
    raise AssertionError("Expected mismatched dataset lengths to fail")


def test_transition_dataset_preserves_and_validates_feature_schema(tmp_path):
  path = tmp_path / "expert.pt"
  torch.save(
    {
      "observations": torch.ones(4, 68),
      "actions": torch.zeros(4, 29),
      "commands": torch.ones(4, 3),
      "metadata": {"feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA},
    },
    path,
  )

  dataset = GailTransitionDataset.load(path)
  dataset.require_feature_schema(VELOCITY_GAIL_FEATURE_SCHEMA)

  assert dataset.metadata["feature_schema"] == VELOCITY_GAIL_FEATURE_SCHEMA


def test_transition_dataset_rejects_legacy_feature_schema(tmp_path):
  path = tmp_path / "legacy.pt"
  torch.save(
    {
      "observations": torch.ones(4, 71),
      "actions": torch.zeros(4, 29),
      "commands": torch.ones(4, 3),
      "metadata": {},
    },
    path,
  )

  dataset = GailTransitionDataset.load(path)

  try:
    dataset.require_feature_schema(VELOCITY_GAIL_FEATURE_SCHEMA)
  except ValueError as exc:
    assert "Recollect the expert dataset" in str(exc)
  else:
    raise AssertionError("Expected a legacy dataset to be rejected")


def test_enabled_gail_requires_a_dataset_path():
  cfg = GailCfg(enabled=True)

  try:
    cfg.validate()
  except ValueError as exc:
    assert "dataset_path" in str(exc)
  else:
    raise AssertionError("Expected enabled GAIL without a dataset to fail")


def test_gail_hyperparameters_must_be_positive():
  cfg = GailCfg(enabled=True, dataset_path="expert.pt", weight=0.0)

  try:
    cfg.validate()
  except ValueError as exc:
    assert "weight" in str(exc)
  else:
    raise AssertionError("Expected zero GAIL weight to fail")


def test_gail_input_mode_must_be_supported():
  cfg = GailCfg(
    enabled=True,
    dataset_path="expert.pt",
    input_mode=cast(Any, "unsupported"),
  )

  with pytest.raises(ValueError, match="input_mode"):
    cfg.validate()


def test_gail_checkpoint_state_is_passed_to_parent_save(monkeypatch):
  discriminator = GailDiscriminator(input_dim=4, hidden_dims=(8,))
  optimizer = torch.optim.Adam(discriminator.parameters())
  runner = object.__new__(GailVelocityOnPolicyRunner)
  cast(Any, runner).gail = SimpleNamespace(
    discriminator=discriminator, optimizer=optimizer, weight=0.01, reward_cap=None
  )
  captured = {}

  def fake_save(_runner, path, infos=None):
    captured["path"] = path
    captured["infos"] = infos

  monkeypatch.setattr(VelocityOnPolicyRunner, "save", fake_save)

  runner.save("model.pt", {"marker": 1})

  assert captured["path"] == "model.pt"
  assert captured["infos"]["marker"] == 1
  assert "gail_state_dict" in captured["infos"]
  assert "gail_optimizer_state_dict" in captured["infos"]


def test_gail_training_rejects_distributed_execution():
  runner = object.__new__(GailVelocityOnPolicyRunner)
  cast(Any, runner).gail = object()
  runner.is_distributed = True

  with pytest.raises(RuntimeError, match="discriminator gradients"):
    runner.learn(1)
