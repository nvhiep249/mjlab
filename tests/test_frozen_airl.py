import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mjlab.rl.airl import AirlBatch, AirlDiscriminator
from mjlab.rl.frozen_airl import FrozenAirlReward, export_frozen_airl


@pytest.fixture
def online_checkpoint(tmp_path):
  model = AirlDiscriminator(68, 3, hidden_dims=(8,), gamma=0.95)
  # Keep distinct, nontrivial statistics in g and h.
  with torch.no_grad():
    model.reward.feature_mean.copy_(torch.linspace(-1, 1, 71))
    model.reward.feature_std.fill_(0.7)
    model.potential.feature_mean.fill_(0.2)
    model.potential.feature_std.fill_(1.3)
  path = tmp_path / "model_1058.pt"
  torch.save(
    {
      "actor_state_dict": {"sentinel": torch.ones(1)},
      "optimizer_state_dict": {"sentinel": 123},
      "infos": {
        "airl_state_dict": model.state_dict(),
        "airl_optimizer_state_dict": {"sentinel": 456},
        "airl_config": {"hidden_dims": (8,), "weight": 0.01},
        "airl_dataset_sha256": "a" * 64,
      },
    },
    path,
  )
  return model, path


def test_export_equal_reward_and_bitwise_frozen_state(online_checkpoint, tmp_path):
  online, source = online_checkpoint
  source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
  target = export_frozen_airl(source, tmp_path / "reward.pt")
  artifact = torch.load(target, weights_only=True)
  assert set(artifact) == {"format", "metadata", "state_dict"}
  assert artifact["metadata"]["source_checkpoint_sha256"] == source_hash
  assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
  rng = torch.get_rng_state().clone()
  frozen = FrozenAirlReward.load(target)
  assert torch.equal(torch.get_rng_state(), rng)
  assert frozen.default_weight == 0.01
  assert not frozen.discriminator.training
  assert all(not p.requires_grad for p in frozen.discriminator.parameters())
  before = copy.deepcopy(frozen.discriminator.state_dict())
  batch = AirlBatch(
    torch.randn(4, 68, requires_grad=True),
    torch.randn(4, 68),
    torch.randn(4, 3),
    torch.randn(4, 3),  # Commands can change at the successor.
    torch.tensor([True, False, False, True]),  # Timeout uses False.
    torch.full((4,), -100.0),  # Density must never enter exported f.
  )
  for _ in range(3):
    actual = frozen.reward(
      batch.observations,
      batch.next_observations,
      batch.commands,
      batch.next_commands,
      batch.terminated,
    )
    torch.testing.assert_close(actual, online.shaped_reward(batch), rtol=0, atol=0)
    assert not actual.requires_grad
  altered = copy.deepcopy(batch)
  altered.next_observations[0] += 5
  altered.next_commands[0] += 3
  torch.testing.assert_close(
    online.shaped_reward(batch)[0], online.shaped_reward(altered)[0], rtol=0, atol=0
  )  # True termination drops successor potential.
  for key, value in frozen.discriminator.state_dict().items():
    assert torch.equal(value, before[key])
    assert torch.equal(value, online.state_dict()[key])
  assert all(p.grad is None for p in frozen.discriminator.parameters())
  with pytest.raises(FileExistsError):
    export_frozen_airl(source, target)


@pytest.mark.parametrize("kind", ["ppo", "gail"])
def test_export_rejects_other_checkpoints(tmp_path, kind):
  source, target = tmp_path / "source.pt", tmp_path / "reward.pt"
  torch.save({"infos": {} if kind == "ppo" else {"gail_state_dict": {}}}, source)
  with pytest.raises(ValueError, match="online AIRL"):
    export_frozen_airl(source, target)
  assert not target.exists()


@pytest.mark.parametrize(
  "fault",
  ["schema", "hidden", "hash", "weight", "gamma", "std", "nan", "shape", "missing"],
)
def test_load_rejects_incompatible_artifact(online_checkpoint, tmp_path, fault):
  _, source = online_checkpoint
  target = export_frozen_airl(source, tmp_path / "reward.pt")
  artifact = torch.load(target, weights_only=True)
  if fault == "schema":
    artifact["metadata"]["feature_schema"] = "legacy"
  elif fault == "hidden":
    artifact["metadata"]["hidden_dims"] = [False]
  elif fault == "hash":
    artifact["metadata"]["airl_dataset_sha256"] = "bad"
  elif fault == "weight":
    artifact["metadata"]["default_weight"] = float("inf")
  elif fault == "gamma":
    artifact["state_dict"]["gamma"] = torch.tensor(1.1)
  elif fault == "std":
    artifact["state_dict"]["potential.feature_std"][0] = 0
  elif fault == "nan":
    artifact["state_dict"]["reward.feature_mean"][0] = float("nan")
  elif fault == "shape":
    artifact["state_dict"]["reward.feature_mean"] = torch.zeros(100)
  else:
    del artifact["state_dict"]["reward.network.0.weight"]
  with pytest.raises((ValueError, RuntimeError)):
    FrozenAirlReward(artifact)


def test_reward_input_validation(online_checkpoint, tmp_path):
  _, source = online_checkpoint
  frozen = FrozenAirlReward.load(export_frozen_airl(source, tmp_path / "reward.pt"))
  args = [torch.zeros(2, 68), torch.zeros(2, 68), torch.zeros(2, 3), torch.zeros(2, 3)]
  with pytest.raises(ValueError, match="boolean"):
    frozen.reward(args[0], args[1], args[2], args[3], torch.zeros(2))
  args[0][0, 0] = float("nan")
  with pytest.raises(ValueError, match="finite"):
    frozen.reward(args[0], args[1], args[2], args[3], torch.zeros(2, dtype=torch.bool))
  args[0] = torch.zeros(2, 99)
  with pytest.raises(ValueError, match="matching"):
    frozen.reward(args[0], args[1], args[2], args[3], torch.zeros(2, dtype=torch.bool))


def test_export_cli(online_checkpoint, tmp_path):
  _, source = online_checkpoint
  target = tmp_path / "cli_reward.pt"
  result = subprocess.run(
    [
      sys.executable,
      "-m",
      "mjlab.scripts.airl_export_reward",
      "--checkpoint-file",
      str(source),
      "--output-file",
      str(target),
    ],
    cwd=Path(__file__).resolve().parents[1],
    capture_output=True,
    text=True,
    check=True,
  )
  assert "Frozen AIRL reward:" in result.stdout
  FrozenAirlReward.load(target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_reward_accepts_cuda_device_alias(online_checkpoint, tmp_path):
  _, source = online_checkpoint
  target = export_frozen_airl(source, tmp_path / "reward.pt")
  frozen = FrozenAirlReward.load(target, device="cuda")
  assert frozen.device == torch.device("cuda", torch.cuda.current_device())
  reward = frozen.reward(
    torch.zeros(2, 68, device="cuda"),
    torch.zeros(2, 68, device="cuda"),
    torch.zeros(2, 3, device="cuda"),
    torch.ones(2, 3, device="cuda"),
    torch.tensor([True, False], device="cuda"),
  )
  assert reward.shape == (2,)
  assert not reward.requires_grad
