"""Export and evaluate G1 AIRL shaped rewards without PPO or a D optimizer."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from mjlab.rl.airl import AIRL_TRANSITION_CONTRACT, AirlBatch, AirlDiscriminator
from mjlab.rl.gail import VELOCITY_GAIL_FEATURE_SCHEMA

FORMAT = "mjlab_frozen_airl_v1"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


class FrozenAirlReward:
  """Fixed f = g + gamma*(1-terminated)*h(next) - h(current), not logit_D."""

  def __init__(self, artifact: dict, device: str | torch.device = "cpu") -> None:
    if artifact.get("format") != FORMAT:
      raise ValueError("Not a standalone mjlab frozen AIRL artifact")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
      raise ValueError("Frozen AIRL metadata is required")
    expected = {
      "feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA,
      "transition_contract": AIRL_TRANSITION_CONTRACT,
      "state_dim": 68,
      "command_dim": 3,
      "reward_mode": "f",
    }
    for key, value in expected.items():
      if metadata.get(key) != value:
        raise ValueError(f"Frozen AIRL requires {key}={value!r}")
    hidden = metadata.get("hidden_dims")
    if not isinstance(hidden, (list, tuple)) or any(
      type(size) is not int or size < 1 for size in hidden
    ):
      raise ValueError("Frozen AIRL hidden_dims must contain positive integers")
    for key in ("source_checkpoint_sha256", "airl_dataset_sha256"):
      digest = metadata.get(key)
      if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
      ):
        raise ValueError(f"Frozen AIRL requires a valid {key}")
    weight = metadata.get("default_weight")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
      raise ValueError("Frozen AIRL default_weight must be finite and positive")
    state = artifact.get("state_dict")
    if not isinstance(state, dict) or not state:
      raise ValueError("Frozen AIRL state_dict is required")
    for value in state.values():
      if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
      ):
        raise ValueError("Frozen AIRL state must contain finite float32 tensors")
    gamma = state.get("gamma")
    if not isinstance(gamma, torch.Tensor) or gamma.shape != () or not 0 <= gamma <= 1:
      raise ValueError("Frozen AIRL requires scalar gamma in [0, 1]")
    for prefix in ("reward", "potential"):
      for name in ("feature_mean", "feature_std"):
        value = state.get(f"{prefix}.{name}")
        if not isinstance(value, torch.Tensor) or value.shape != (71,):
          raise ValueError("Frozen AIRL requires 71-D normalization buffers")
        if name == "feature_std" and not (value > 0).all():
          raise ValueError("Frozen AIRL feature_std must be positive")
    self.default_weight = float(weight)
    self.metadata = dict(metadata)
    # Loading a fixed reward must not perturb subsequent PPO initialization.
    with torch.random.fork_rng(devices=[]):
      self.discriminator = AirlDiscriminator(68, 3, tuple(hidden), float(gamma))
    self.discriminator.load_state_dict(state, strict=True)
    self.discriminator.to(device).eval().requires_grad_(False)
    self.device = self.discriminator.gamma.device

  @classmethod
  def load(
    cls, path: str | Path, device: str | torch.device = "cpu"
  ) -> FrozenAirlReward:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
      raise ValueError("Frozen AIRL artifact must be a dictionary")
    return cls(artifact, device)

  @torch.no_grad()
  def reward(
    self,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    commands: torch.Tensor,
    next_commands: torch.Tensor,
    terminated: torch.Tensor,
  ) -> torch.Tensor:
    """Inputs use body-local 68-D states and successors captured before reset."""
    count = observations.shape[0] if observations.ndim == 2 else -1
    for value, width in (
      (observations, 68),
      (next_observations, 68),
      (commands, 3),
      (next_commands, 3),
    ):
      if (
        value.shape != (count, width)
        or value.dtype != torch.float32
        or value.device != self.device
        or not torch.isfinite(value).all()
      ):
        raise ValueError("Frozen AIRL inputs must be finite matching float32 batches")
    if (
      terminated.shape != (count,)
      or terminated.dtype != torch.bool
      or terminated.device != self.device
    ):
      raise ValueError("Frozen AIRL terminated must be a matching boolean mask")
    batch = AirlBatch(
      observations,
      next_observations,
      commands,
      next_commands,
      terminated,
      torch.zeros(count, device=self.device),  # Unused by shaped_reward.
    )
    result = self.discriminator.shaped_reward(batch)
    if not torch.isfinite(result).all():
      raise ValueError("Frozen AIRL produced nonfinite rewards")
    return result


def export_frozen_airl(checkpoint_file: str | Path, output_file: str | Path) -> Path:
  """Extract full g/h/buffers from a trusted online checkpoint; never overwrite."""
  source, output = Path(checkpoint_file), Path(output_file)
  if output.exists():
    raise FileExistsError(output)
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  infos = checkpoint.get("infos") if isinstance(checkpoint, dict) else None
  if not isinstance(infos, dict) or "airl_state_dict" not in infos:
    raise ValueError("Export requires an online AIRL checkpoint, not PPO/GAIL")
  config, state = infos.get("airl_config"), infos["airl_state_dict"]
  if not isinstance(config, dict) or not isinstance(state, dict):
    raise ValueError("AIRL checkpoint config/state is missing")
  artifact = {
    "format": FORMAT,
    "metadata": {
      "feature_schema": VELOCITY_GAIL_FEATURE_SCHEMA,
      "transition_contract": AIRL_TRANSITION_CONTRACT,
      "state_dim": 68,
      "command_dim": 3,
      "hidden_dims": config.get("hidden_dims"),
      "reward_mode": "f",
      "default_weight": config.get("weight"),
      "source_checkpoint_sha256": _sha256(source),
      "airl_dataset_sha256": infos.get("airl_dataset_sha256"),
    },
    "state_dict": {
      key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
      for key, value in state.items()
    },
  }
  FrozenAirlReward(artifact)  # Validate before writing anything.
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("xb") as stream:
    torch.save(artifact, stream)
  return output
