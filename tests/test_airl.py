from typing import cast

import pytest
import torch
from torch import nn

from mjlab.rl.airl import (
  AIRL_ACTION_CONTRACT,
  AIRL_ACTOR_OBSERVATION_CONTRACT,
  AIRL_TRANSITION_CONTRACT,
  AirlBatch,
  AirlDiscriminator,
  AirlTransitionDataset,
  airl_discriminator_loss,
  airl_reward,
)
from mjlab.rl.config import AirlCfg, RslRlOnPolicyRunnerCfg
from mjlab.rl.gail import VELOCITY_GAIL_FEATURE_SCHEMA


def _batch(log_prob=None):
  return AirlBatch(
    torch.tensor([[1.0], [1.0]]),
    torch.tensor([[2.0], [2.0]]),
    torch.tensor([[3.0], [3.0]]),
    torch.tensor([[4.0], [4.0]]),
    torch.tensor([True, False]),
    torch.zeros(2) if log_prob is None else log_prob,
  )


def test_airl_formula_terminal_bootstrap_and_next_command():
  model = AirlDiscriminator(1, 1, hidden_dims=(), gamma=0.5)
  reward = cast(nn.Linear, model.reward.network[0])
  potential = cast(nn.Linear, model.potential.network[0])
  with torch.no_grad():
    reward.weight.copy_(torch.tensor([[2.0, 0.0]]))
    assert reward.bias is not None and potential.bias is not None
    reward.bias.zero_()
    potential.weight.copy_(torch.tensor([[1.0, 1.0]]))
    potential.bias.zero_()
  batch = _batch(torch.tensor([-3.0, -3.0]))
  # Terminated drops h(s'); timeout/nonterminal still bootstraps.
  torch.testing.assert_close(model.shaped_reward(batch), torch.tensor([-2.0, 1.0]))
  torch.testing.assert_close(model(batch), torch.tensor([1.0, 4.0]))
  torch.testing.assert_close(airl_reward(model, batch), torch.tensor([-2.0, 1.0]))
  base, shaped = model.reward_components(batch)
  torch.testing.assert_close(base, torch.tensor([2.0, 2.0]))
  torch.testing.assert_close(shaped, model.shaped_reward(batch))


def test_airl_gradient_separation_and_finite_extreme_density():
  log_prob = torch.tensor([-10000.0, 10000.0], requires_grad=True)
  batch = _batch(log_prob)
  batch.observations.requires_grad_()
  model = AirlDiscriminator(1, 1, hidden_dims=(8,))
  loss, metrics = airl_discriminator_loss(model, batch, batch)
  loss.backward()
  assert torch.isfinite(loss)
  assert log_prob.grad is None
  assert batch.observations.grad is None
  assert any(parameter.grad is not None for parameter in model.parameters())
  assert 0 <= metrics["accuracy"] <= 1
  assert not airl_reward(model, batch).requires_grad


def test_airl_checkpoint_preserves_normalization_and_discount():
  model = AirlDiscriminator(
    1,
    1,
    hidden_dims=(8,),
    gamma=0.7,
    feature_mean=torch.tensor([3.0, 4.0]),
    feature_std=torch.tensor([2.0, 5.0]),
  )
  restored = AirlDiscriminator(1, 1, hidden_dims=(8,))
  restored.load_state_dict(model.state_dict())
  torch.testing.assert_close(restored(_batch()), model(_batch()))
  torch.testing.assert_close(restored.gamma, model.gamma)


def _data():
  return {
    "observations": torch.zeros(2, 68),
    "next_observations": torch.ones(2, 68),
    "actor_observations": torch.zeros(2, 96),
    "next_actor_observations": torch.ones(2, 96),
    "actions": torch.zeros(2, 29),
    "commands": torch.zeros(2, 3),
    "next_commands": torch.ones(2, 3),
    "terminated": torch.tensor([True, False]),
    "truncated": torch.tensor([False, True]),
  }


def _metadata() -> dict[str, object]:
  return {
    "feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA,
    "transition_contract": AIRL_TRANSITION_CONTRACT,
    "actor_observation_contract": AIRL_ACTOR_OBSERVATION_CONTRACT,
    "action_contract": AIRL_ACTION_CONTRACT,
    "expert_policy": "stochastic",
  }


def test_airl_dataset_roundtrip_preserves_density_inputs_and_timeout(tmp_path):
  path = tmp_path / "expert.pt"
  torch.save({**_data(), "metadata": _metadata()}, path)
  dataset = AirlTransitionDataset.load(path)
  assert set(dataset[0]) == set(_data())
  assert dataset[1]["truncated"].dtype == torch.bool
  assert dataset[1]["truncated"]
  assert not dataset[1]["terminated"]


def test_airl_dataset_accepts_deterministic_raw_mean_actions():
  metadata = _metadata()
  metadata.update(expert_policy="deterministic", action_contract="raw_policy_mean")
  dataset = AirlTransitionDataset(_data(), metadata)
  assert dataset.metadata["expert_policy"] == "deterministic"


@pytest.mark.parametrize(
  "policy,contract",
  [
    ("deterministic", "raw_policy_sample"),
    ("stochastic", "raw_policy_mean"),
  ],
)
def test_airl_dataset_rejects_mislabeled_action_mode(policy, contract):
  metadata = _metadata()
  metadata.update(expert_policy=policy, action_contract=contract)
  with pytest.raises(ValueError, match="action_contract"):
    AirlTransitionDataset(_data(), metadata)


@pytest.mark.parametrize("field", list(_data()))
def test_airl_dataset_rejects_missing_fields(field):
  data = _data()
  del data[field]
  with pytest.raises(ValueError, match="missing required"):
    AirlTransitionDataset(data, _metadata())


@pytest.mark.parametrize("field", list(_metadata()))
def test_airl_dataset_rejects_unsafe_metadata(field):
  metadata = _metadata()
  metadata[field] = "legacy"
  with pytest.raises(ValueError, match=field):
    AirlTransitionDataset(_data(), metadata)


def test_airl_dataset_rejects_invalid_terminal_mask():
  data = _data()
  data["terminated"] = torch.tensor([0.0, 0.5])
  with pytest.raises(ValueError, match="boolean mask"):
    AirlTransitionDataset(data, _metadata())


def test_airl_configuration_is_opt_in_and_validates_enabled_fields():
  assert not RslRlOnPolicyRunnerCfg().airl.enabled
  AirlCfg().validate()
  with pytest.raises(ValueError, match="dataset_path"):
    AirlCfg(enabled=True).validate()
  AirlCfg(enabled=True, dataset_path="expert.pt").validate()


def test_airl_potential_telescopes_on_complete_trajectory():
  model = AirlDiscriminator(1, 1, hidden_dims=(), gamma=0.5)
  reward = cast(nn.Linear, model.reward.network[0])
  potential = cast(nn.Linear, model.potential.network[0])
  with torch.no_grad():
    reward.weight.zero_()
    assert reward.bias is not None and potential.bias is not None
    reward.bias.zero_()
    potential.weight.copy_(torch.tensor([[1.0, 0.0]]))
    potential.bias.zero_()
  batch = AirlBatch(
    torch.tensor([[1.0], [2.0], [3.0]]),
    torch.tensor([[2.0], [3.0], [9.0]]),
    torch.zeros(3, 1),
    torch.zeros(3, 1),
    torch.tensor([False, False, True]),
    torch.zeros(3),
  )
  discounted_return = (airl_reward(model, batch) * torch.tensor([1.0, 0.5, 0.25])).sum()
  torch.testing.assert_close(discounted_return, torch.tensor(-1.0))


def test_airl_learned_reward_improves_two_state_toy_policy():
  """Fit AIRL on a toy task, then optimize a new policy against frozen f."""
  torch.manual_seed(17)
  model = AirlDiscriminator(1, 1, hidden_dims=(8,))
  states = torch.tensor([[-1.0], [1.0]])
  batch = AirlBatch(
    states,
    torch.zeros_like(states),
    torch.zeros_like(states),
    torch.zeros_like(states),
    torch.ones(2, dtype=torch.bool),
    torch.full((2,), -0.693147),
  )

  def rows(index: int) -> AirlBatch:
    return AirlBatch(
      *(
        value[index : index + 1]
        for value in (
          batch.observations,
          batch.next_observations,
          batch.commands,
          batch.next_commands,
          batch.terminated,
          batch.log_prob,
        )
      )
    )

  optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
  initial_loss, _ = airl_discriminator_loss(model, rows(1), rows(0))
  for _ in range(60):
    optimizer.zero_grad()
    loss, _ = airl_discriminator_loss(model, rows(1), rows(0))
    loss.backward()
    optimizer.step()
  final_loss, _ = airl_discriminator_loss(model, rows(1), rows(0))
  rewards = airl_reward(model, batch)
  assert final_loss < initial_loss * 0.1
  assert rewards[1] > rewards[0]
  logits = nn.Parameter(torch.zeros(2))
  policy_optimizer = torch.optim.Adam([logits], lr=0.1)
  for _ in range(40):
    policy_optimizer.zero_grad()
    (-torch.sum(logits.softmax(0) * rewards)).backward()
    policy_optimizer.step()
  assert float(logits.softmax(0)[1].detach()) > 0.95


@pytest.mark.parametrize("field", ["terminated", "truncated", "dones"])
def test_airl_dataset_rejects_nonfinite_flags(field):
  data = _data()
  data[field] = torch.tensor([False, float("inf")])
  with pytest.raises(ValueError, match="boolean mask"):
    AirlTransitionDataset(data, _metadata())


def test_airl_dataset_rejects_inconsistent_done_flags():
  data = _data()
  data["dones"] = torch.tensor([False, True])
  with pytest.raises(ValueError, match="dones must equal"):
    AirlTransitionDataset(data, _metadata())
