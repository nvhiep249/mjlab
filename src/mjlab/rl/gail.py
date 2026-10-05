"""Small PyTorch building blocks for GAIL experiments.

This module intentionally does not depend on a particular RL runner.  It provides
the data contract, discriminator, and numerically stable imitation reward used by
the toy experiments and by the future rsl_rl integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

VELOCITY_GAIL_FEATURE_SCHEMA = "g1_velocity_body_local_v1"


class CommandMatchedSampler:
  """Sample expert rows with the same discrete command as each policy row."""

  def __init__(self, commands: torch.Tensor) -> None:
    commands = commands.detach().cpu()
    if commands.ndim != 2 or not len(commands) or not torch.isfinite(commands).all():
      raise ValueError("Expected finite, nonempty expert commands")
    self.commands, inverse = torch.unique(commands, dim=0, return_inverse=True)
    self.rows = [torch.where(inverse == i)[0] for i in range(len(self.commands))]

  def sample(self, commands: torch.Tensor) -> torch.Tensor:
    commands = commands.detach().cpu()
    matches = torch.isclose(
      commands[:, None, :], self.commands[None, :, :], atol=1e-6, rtol=0
    ).all(dim=-1)
    if not torch.all(matches.sum(dim=1) == 1):
      raise ValueError("Missing expert command or ambiguous expert command bins")
    result = torch.empty(len(commands), dtype=torch.long)
    for i, rows in enumerate(self.rows):
      selected = torch.where(matches[:, i])[0]
      result[selected] = rows[torch.randint(len(rows), (len(selected),))]
    return result


@dataclass
class GailBatch:
  """A batch of state-action-command transitions."""

  observations: torch.Tensor
  actions: torch.Tensor | None = None
  commands: torch.Tensor | None = None
  next_observations: torch.Tensor | None = None

  def features(self) -> torch.Tensor:
    """Concatenate the discriminator inputs along the last dimension."""
    targets = [
      value for value in (self.actions, self.next_observations) if value is not None
    ]
    if len(targets) != 1:
      raise ValueError(
        "GAIL batch requires exactly one of actions or next_observations"
      )
    tensors = [self.observations, targets[0]]
    if self.commands is not None:
      tensors.append(self.commands)
    return torch.cat(tensors, dim=-1)


def velocity_gail_state(robot: Any) -> torch.Tensor:
  """Build translation- and heading-invariant velocity imitation features."""
  data = robot.data
  return torch.cat(
    (
      data.root_link_pos_w[:, 2:3],
      data.projected_gravity_b,
      data.root_link_lin_vel_b,
      data.root_link_ang_vel_b,
      data.joint_pos - data.default_joint_pos,
      data.joint_vel,
    ),
    dim=-1,
  )


class GailTransitionDataset(Dataset[dict[str, torch.Tensor]]):
  """Validated transition dataset for expert or policy demonstrations.

  Supported files contain ``observations`` and at least one of ``actions`` or
  ``next_observations``. ``commands`` are optional. ``torch.save`` files and
  NumPy ``.npz`` files are accepted. All floating-point arrays are converted to
  contiguous float32 tensors.
  """

  def __init__(
    self,
    data: dict[str, torch.Tensor],
    metadata: dict[str, object] | None = None,
  ) -> None:
    if "observations" not in data:
      raise ValueError("Missing required dataset field: observations")
    if not ({"actions", "next_observations"} & data.keys()):
      raise ValueError("Dataset requires actions or next_observations")
    dtype_by_field = {
      "images": torch.uint8,
      "environment_ids": torch.long,
      "episode_ids": torch.long,
      "dones": torch.bool,
      "terminated": torch.bool,
    }
    self._data = {
      name: torch.as_tensor(
        value, dtype=dtype_by_field.get(name, torch.float32)
      ).contiguous()
      for name, value in data.items()
    }
    lengths = {value.shape[0] for value in self._data.values()}
    if len(lengths) != 1:
      raise ValueError("All dataset fields must have the same number of transitions")
    if not lengths or next(iter(lengths)) == 0:
      raise ValueError("Dataset must contain at least one transition")
    if any(not torch.isfinite(value).all() for value in self._data.values()):
      raise ValueError("Dataset contains NaN or Inf values")
    self.metadata = metadata or {}

  @classmethod
  def load(cls, path: str | Path) -> "GailTransitionDataset":
    """Load a `.npz` or PyTorch file using the common field names."""
    path = Path(path)
    retired = path.with_suffix(".INVALID.md")
    if retired.exists():
      raise ValueError(f"Retired imitation dataset: {path}; see {retired}")
    if path.suffix == ".npz":
      import numpy as np

      with np.load(path) as loaded:
        data = {name: torch.from_numpy(loaded[name]) for name in loaded.files}
    else:
      loaded = torch.load(path, map_location="cpu", weights_only=True)
      if not isinstance(loaded, dict):
        raise ValueError("PyTorch dataset must contain a dictionary")
      data = {
        name: value
        for name, value in loaded.items()
        if name
        in {
          "observations",
          "next_observations",
          "actions",
          "commands",
          "images",
          "environment_ids",
          "episode_ids",
          "dones",
          "terminated",
        }
      }
    metadata = loaded.get("metadata") if isinstance(loaded, dict) else None
    return cls(data, metadata if isinstance(metadata, dict) else None)

  def require_feature_schema(self, expected: str) -> None:
    """Reject legacy datasets whose discriminator state contract is unsafe."""
    actual = self.metadata.get("feature_schema")
    if actual != expected:
      raise ValueError(
        f"Expected GAIL feature schema '{expected}', got {actual!r}. "
        "Recollect the expert dataset with the current gail-collect-expert."
      )

  def __len__(self) -> int:
    return self._data["observations"].shape[0]

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    return {name: value[index] for name, value in self._data.items()}

  def image_stack(self, index: int, stack_size: int = 4) -> torch.Tensor:
    """Return an episode-safe ``T,C,H,W`` uint8 history for one transition."""
    if stack_size < 1:
      raise ValueError("stack_size must be positive")
    required = {"images", "environment_ids", "episode_ids"}
    missing = required - self._data.keys()
    if missing:
      raise ValueError(f"Image stacking requires fields: {sorted(missing)}")
    if self.metadata.get("image_layout") != "NHWC":
      raise ValueError("Image stacking requires metadata image_layout='NHWC'")

    environment_ids = self._data["environment_ids"]
    episode_ids = self._data["episode_ids"]
    same_trajectory = (environment_ids[: index + 1] == environment_ids[index]) & (
      episode_ids[: index + 1] == episode_ids[index]
    )
    history_indices = torch.nonzero(same_trajectory, as_tuple=False).flatten()
    history_indices = history_indices[-stack_size:]
    if len(history_indices) < stack_size:
      padding = history_indices[0].repeat(stack_size - len(history_indices))
      history_indices = torch.cat((padding, history_indices))
    images = self._data["images"][history_indices]
    return images.permute(0, 3, 1, 2).contiguous()

  def normalized_image_batch(
    self, indices: torch.Tensor, stack_size: int = 4
  ) -> torch.Tensor:
    """Materialize and normalize only the requested image histories."""
    stacks = [self.image_stack(int(index), stack_size) for index in indices]
    return torch.stack(stacks).float().div_(255.0)

  def split_episode_stratified(
    self, validation_fraction: float = 0.2, seed: int = 42
  ) -> tuple["GailTransitionDataset", "GailTransitionDataset"]:
    """Split complete episodes, stratified by exact command-bin values.

    Every transition from an ``(environment_id, episode_id)`` pair stays in
    one split. Command values are used as stable bins, which works for both
    fixed-grid collection and sampled commands.
    """
    required = {"environment_ids", "episode_ids", "commands"}
    missing = required - self._data.keys()
    if missing:
      raise ValueError(f"Episode-stratified split requires fields: {sorted(missing)}")
    if not 0.0 < validation_fraction < 1.0:
      raise ValueError("validation_fraction must be between 0 and 1")

    groups: dict[tuple[int, int], list[int]] = {}
    env_ids = self._data["environment_ids"].tolist()
    episode_ids = self._data["episode_ids"].tolist()
    commands = self._data["commands"]
    for index, key in enumerate(zip(env_ids, episode_ids, strict=True)):
      groups.setdefault((int(key[0]), int(key[1])), []).append(index)

    bins: dict[tuple[float, ...], list[tuple[int, int]]] = {}
    for key, indices in groups.items():
      command = tuple(float(value) for value in commands[indices[0]].tolist())
      bins.setdefault(command, []).append(key)

    generator = torch.Generator().manual_seed(seed)
    validation_groups: set[tuple[int, int]] = set()
    for group_keys in bins.values():
      order = torch.randperm(len(group_keys), generator=generator).tolist()
      count = max(1, round(len(group_keys) * validation_fraction))
      if len(group_keys) > 1:
        count = min(count, len(group_keys) - 1)
      else:
        count = 0
      validation_groups.update(group_keys[index] for index in order[:count])

    validation_indices = [
      index
      for index, key in enumerate(zip(env_ids, episode_ids, strict=True))
      if key in validation_groups
    ]
    train_indices = [
      index
      for index, key in enumerate(zip(env_ids, episode_ids, strict=True))
      if key not in validation_groups
    ]
    if not train_indices or not validation_indices:
      raise ValueError("Split requires at least one train and validation episode")

    def make_subset(indices: list[int]) -> GailTransitionDataset:
      index_tensor = torch.tensor(indices, dtype=torch.long)
      subset = {name: value[index_tensor] for name, value in self._data.items()}
      subset_metadata = dict(self.metadata)
      subset_metadata.update(
        {"split": "validation" if indices is validation_indices else "train"}
      )
      return GailTransitionDataset(subset, subset_metadata)

    return make_subset(train_indices), make_subset(validation_indices)


class GailDiscriminator(nn.Module):
  """MLP discriminator returning logits for expert-vs-policy transitions."""

  feature_mean: torch.Tensor
  feature_std: torch.Tensor

  def __init__(
    self,
    input_dim: int,
    hidden_dims: tuple[int, ...] = (256, 256),
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
  ) -> None:
    super().__init__()
    if feature_mean is None:
      feature_mean = torch.zeros(input_dim)
    if feature_std is None:
      feature_std = torch.ones(input_dim)
    if feature_mean.shape != (input_dim,) or feature_std.shape != (input_dim,):
      raise ValueError("GAIL feature statistics must match input_dim")
    self.register_buffer("feature_mean", feature_mean.float())
    self.register_buffer("feature_std", feature_std.float().clamp_min(1e-6))
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(last_dim, hidden_dim), nn.Tanh()))
      last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, 1))
    self.network = nn.Sequential(*layers)

  @classmethod
  def from_expert_features(
    cls,
    expert_features: torch.Tensor,
    hidden_dims: tuple[int, ...] = (256, 256),
  ) -> "GailDiscriminator":
    """Create a discriminator normalized with per-feature expert statistics."""
    if expert_features.ndim != 2 or expert_features.shape[0] == 0:
      raise ValueError("Expert features must be a non-empty rank-2 tensor")
    return cls(
      expert_features.shape[1],
      hidden_dims,
      feature_mean=expert_features.mean(dim=0),
      feature_std=expert_features.std(dim=0, unbiased=False),
    )

  def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
    """Normalize discriminator inputs using the expert reference distribution."""
    return ((features - self.feature_mean) / self.feature_std).clamp(-10.0, 10.0)

  def forward(self, batch: GailBatch | torch.Tensor) -> torch.Tensor:
    """Return one discriminator logit per transition."""
    features = batch.features() if isinstance(batch, GailBatch) else batch
    return self.network(self.normalize_features(features)).squeeze(-1)


def discriminator_loss(
  discriminator: GailDiscriminator,
  expert: GailBatch,
  policy: GailBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Compute BCE-with-logits loss and detached monitoring metrics."""
  expert_logits = discriminator(expert)
  policy_logits = discriminator(policy)
  expert_targets = torch.ones_like(expert_logits)
  policy_targets = torch.zeros_like(policy_logits)
  loss_fn = nn.BCEWithLogitsLoss()
  loss = loss_fn(expert_logits, expert_targets) + loss_fn(policy_logits, policy_targets)
  with torch.no_grad():
    expert_correct = (expert_logits > 0).float().mean()
    policy_correct = (policy_logits <= 0).float().mean()
    accuracy = 0.5 * (expert_correct + policy_correct)
  return loss, {"loss": float(loss.detach()), "accuracy": float(accuracy)}


def imitation_reward(
  discriminator: GailDiscriminator,
  policy: GailBatch,
  *,
  normalize: bool = False,
  eps: float = 1e-6,
) -> torch.Tensor:
  """Return the non-saturating GAIL reward ``-log(1 - D)``.

  The discriminator is evaluated without gradients because this reward is consumed
  by the policy optimizer, while the discriminator is updated separately.
  """
  with torch.no_grad():
    logits = discriminator(policy)
    reward = torch.nn.functional.softplus(logits)
    if normalize:
      reward = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(eps)
    return reward
