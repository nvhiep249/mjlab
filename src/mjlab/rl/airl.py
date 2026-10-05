"""State-only AIRL reward and strict pre-reset expert transition contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mjlab.rl.gail import (
  VELOCITY_GAIL_FEATURE_SCHEMA,
  GailDiscriminator,
  GailTransitionDataset,
)

AIRL_TRANSITION_CONTRACT = "pre_reset_v1"
AIRL_ACTOR_OBSERVATION_CONTRACT = "policy_input_v1"
AIRL_ACTION_CONTRACT = "raw_policy_sample"
AIRL_DETERMINISTIC_ACTION_CONTRACT = "raw_policy_mean"


@dataclass
class AirlBatch:
  observations: torch.Tensor
  next_observations: torch.Tensor
  commands: torch.Tensor
  next_commands: torch.Tensor
  terminated: torch.Tensor
  log_prob: torch.Tensor


class AirlTransitionDataset(GailTransitionDataset):
  """Reject legacy data lacking policy inputs or true terminal transitions."""

  def __init__(
    self,
    data: dict[str, torch.Tensor],
    metadata: dict[str, object] | None = None,
  ) -> None:
    required = {
      "observations",
      "next_observations",
      "commands",
      "next_commands",
      "actor_observations",
      "next_actor_observations",
      "actions",
      "terminated",
      "truncated",
    }
    missing = required - data.keys()
    if missing:
      raise ValueError(f"AIRL dataset missing required fields: {sorted(missing)}")
    metadata = metadata or {}
    contracts = {
      "feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA,
      "transition_contract": AIRL_TRANSITION_CONTRACT,
      "actor_observation_contract": AIRL_ACTOR_OBSERVATION_CONTRACT,
    }
    policy_mode = metadata.get("expert_policy")
    if policy_mode not in ("stochastic", "deterministic"):
      raise ValueError("AIRL requires explicit expert_policy mode")
    contracts["action_contract"] = (
      AIRL_ACTION_CONTRACT
      if policy_mode == "stochastic"
      else AIRL_DETERMINISTIC_ACTION_CONTRACT
    )
    for name, expected in contracts.items():
      if metadata.get(name) != expected:
        raise ValueError(f"AIRL requires metadata {name}={expected!r}")
    flag_names = (
      ("terminated", "truncated", "dones")
      if "dones" in data
      else ("terminated", "truncated")
    )
    for name in flag_names:
      flags = torch.as_tensor(data[name])
      if flags.ndim != 1 or not torch.all((flags == 0) | (flags == 1)):
        raise ValueError(f"AIRL {name} must be a rank-1 boolean mask")
    if "dones" in data and not torch.equal(
      data["dones"].bool(), data["terminated"].bool() | data["truncated"].bool()
    ):
      raise ValueError("AIRL dones must equal terminated | truncated")
    for name in required - {"terminated", "truncated"}:
      if torch.as_tensor(data[name]).ndim != 2:
        raise ValueError(f"AIRL {name} must be rank-2")
    for current, following in (
      ("observations", "next_observations"),
      ("commands", "next_commands"),
      ("actor_observations", "next_actor_observations"),
    ):
      if data[current].shape != data[following].shape:
        raise ValueError(f"AIRL {current} and {following} shapes must match")
    if data["observations"].shape[1] != 68:
      raise ValueError("AIRL velocity feature schema requires 68-D observations")
    if data["commands"].shape[1] != 3:
      raise ValueError("AIRL velocity commands require 3 dimensions")
    if data["actions"].shape[1] != 29:
      raise ValueError("AIRL G1 raw actions require 29 dimensions")
    if data["actor_observations"].shape[1] < 1:
      raise ValueError("AIRL actor observations must contain policy inputs")
    super().__init__(data, metadata)
    self._data["truncated"] = self._data["truncated"].bool()

  @classmethod
  def load(cls, path: str | Path) -> "AirlTransitionDataset":
    path = Path(path)
    if path.suffix != ".pt":
      raise ValueError("AIRL requires a .pt dataset with explicit metadata")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
      raise ValueError("AIRL dataset must contain a dictionary")
    metadata = loaded.get("metadata")
    data = {name: value for name, value in loaded.items() if name != "metadata"}
    return cls(data, metadata if isinstance(metadata, dict) else None)


class AirlDiscriminator(nn.Module):
  """AIRL logits f(s,s') - log pi(a|actor_obs), with state-only g."""

  gamma: torch.Tensor

  def __init__(
    self,
    state_dim: int,
    command_dim: int,
    hidden_dims: tuple[int, ...] = (256, 256),
    gamma: float = 0.99,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
  ) -> None:
    super().__init__()
    if not 0 <= gamma <= 1:
      raise ValueError("AIRL gamma must be between 0 and 1")
    self.register_buffer("gamma", torch.tensor(gamma))
    self.reward = GailDiscriminator(
      state_dim + command_dim, hidden_dims, feature_mean, feature_std
    )
    self.potential = GailDiscriminator(
      state_dim + command_dim, hidden_dims, feature_mean, feature_std
    )

  def shaped_reward(self, batch: AirlBatch) -> torch.Tensor:
    return self.reward_components(batch)[1]

  def reward_components(self, batch: AirlBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return g and f together so rollout diagnostics reuse the g forward."""
    state = torch.cat((batch.observations, batch.commands), dim=-1).detach()
    next_state = torch.cat(
      (batch.next_observations, batch.next_commands), dim=-1
    ).detach()
    terminal = batch.terminated.reshape(-1).bool()
    if terminal.shape != state.shape[:1] or next_state.shape != state.shape:
      raise ValueError("AIRL transition batch shapes must match")
    base = self.reward(state)
    return base, (
      base
      + self.gamma * (~terminal).float() * self.potential(next_state)
      - self.potential(state)
    )

  def forward(self, batch: AirlBatch) -> torch.Tensor:
    shaped = self.shaped_reward(batch)
    log_prob = batch.log_prob.detach().reshape(-1)
    if log_prob.shape != shaped.shape or not torch.isfinite(log_prob).all():
      raise ValueError("AIRL requires one finite current policy log_prob per row")
    return shaped - log_prob


def airl_discriminator_loss(
  discriminator: AirlDiscriminator,
  expert: AirlBatch,
  policy: AirlBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
  expert_logits, policy_logits = discriminator(expert), discriminator(policy)
  loss_fn = nn.BCEWithLogitsLoss()
  loss = loss_fn(expert_logits, torch.ones_like(expert_logits)) + loss_fn(
    policy_logits, torch.zeros_like(policy_logits)
  )
  with torch.no_grad():
    accuracy = 0.5 * (
      (expert_logits > 0).float().mean() + (policy_logits <= 0).float().mean()
    )
  return loss, {"loss": float(loss.detach()), "accuracy": float(accuracy)}


def airl_reward(discriminator: AirlDiscriminator, batch: AirlBatch) -> torch.Tensor:
  """Return detached f, excluding the policy-density term from PPO rewards."""
  with torch.no_grad():
    return discriminator.shaped_reward(batch)
