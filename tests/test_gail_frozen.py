from types import SimpleNamespace

import pytest
import torch

from mjlab.rl.config import GailCfg
from mjlab.rl.gail import GailBatch, GailDiscriminator, imitation_reward
from mjlab.tasks.velocity.rl.gail_runner import GailVelocityOnPolicyRunner
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


@pytest.mark.parametrize("nested", [False, True])
def test_frozen_runner_keeps_discriminator_fixed_during_learning(
  tmp_path, monkeypatch, nested
):
  source = GailDiscriminator(4)
  source.feature_mean.fill_(0.7)
  source.feature_std.fill_(2.0)
  payload = {"gail_state_dict": source.state_dict()}
  path = tmp_path / "model.pt"
  torch.save({"infos": payload} if nested else payload, path)
  dataset = SimpleNamespace(
    _data={"observations": torch.zeros(8, 2), "actions": torch.zeros(8, 2)},
    require_feature_schema=lambda _: None,
  )

  # Supply the dataset protocol without creating a simulator.
  class Dataset:
    _data = dataset._data

    def require_feature_schema(self, schema):
      dataset.require_feature_schema(schema)

    def __getitem__(self, index):
      return {key: value[index] for key, value in self._data.items()}

    def __len__(self):
      return 8

  monkeypatch.setattr(
    "mjlab.tasks.velocity.rl.gail_runner.GailTransitionDataset.load",
    lambda _: Dataset(),
  )
  monkeypatch.setattr(VelocityOnPolicyRunner, "__init__", lambda self, *a, **kw: None)
  runner = object.__new__(GailVelocityOnPolicyRunner)
  runner.device = "cpu"
  runner.__init__(
    None,
    {
      "gail": {
        "enabled": True,
        "dataset_path": "expert.pt",
        "frozen": True,
        "discriminator_checkpoint": str(path),
      }
    },
  )
  assert runner.gail is not None
  assert runner.gail.updates == 0
  assert not runner.gail.discriminator.training
  assert all(not p.requires_grad for p in runner.gail.discriminator.parameters())
  before = {k: v.clone() for k, v in runner.gail.discriminator.state_dict().items()}
  actor = torch.nn.Linear(2, 2)
  optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
  actor_before = actor.weight.detach().clone()
  rewards = []

  def update():
    optimizer.zero_grad()
    actor(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    return {}

  monkeypatch.setattr(
    runner,
    "alg",
    SimpleNamespace(
      train_mode=lambda: None,
      act=lambda obs: actor(obs),
      process_env_step=lambda obs, reward, *args: rewards.append(reward.clone()),
      compute_returns=lambda _: None,
      update=update,
      learning_rate=0.01,
      get_policy=lambda: SimpleNamespace(output_std=1.0),
    ),
    raising=False,
  )
  monkeypatch.setattr(
    runner,
    "env",
    SimpleNamespace(
      device="cpu",
      get_observations=lambda: torch.ones(2, 2),
      step=lambda actions: (torch.ones(2, 2), torch.ones(2), torch.zeros(2), {}),
    ),
    raising=False,
  )
  runner._state_action = lambda actions: GailBatch(torch.ones(2, 2), actions)
  monkeypatch.setattr(
    runner,
    "logger",
    SimpleNamespace(
      writer=None,
      init_logging_writer=lambda: None,
      process_env_step=lambda *args: None,
      log=lambda **kwargs: None,
    ),
    raising=False,
  )
  runner.cfg = {"num_steps_per_env": 2, "check_for_nan": False}
  runner.is_distributed = False
  runner.current_learning_iteration = 0
  monkeypatch.setattr(
    runner, "_update_discriminator", lambda _: pytest.fail("Frozen D was updated")
  )
  runner.learn(3)
  assert not torch.equal(actor_before, actor.weight)
  assert all(torch.isfinite(reward).all() and (reward > 1).all() for reward in rewards)
  for key, value in runner.gail.discriminator.state_dict().items():
    assert torch.equal(value, before[key])
    assert torch.equal(value, source.state_dict()[key])
  batch = GailBatch(torch.ones(2, 2), torch.ones(2, 2))
  assert torch.isfinite(imitation_reward(source, batch)).all()
  with pytest.raises(ValueError, match="fresh PPO"):
    runner.load(str(path))


def test_frozen_config_requires_checkpoint():
  with pytest.raises(ValueError, match="discriminator_checkpoint"):
    GailCfg(enabled=True, dataset_path="expert.pt", frozen=True).validate()


def test_frozen_loader_rejects_missing_and_mismatched_state(tmp_path, monkeypatch):
  runner = object.__new__(GailVelocityOnPolicyRunner)
  runner.device = "cpu"
  monkeypatch.setattr(
    runner,
    "gail",
    SimpleNamespace(discriminator=GailDiscriminator(4), frozen=True),
    raising=False,
  )
  path = tmp_path / "bad.pt"
  torch.save({"model_state_dict": {}}, path)
  with pytest.raises(ValueError, match="gail_state_dict"):
    runner._load_frozen_discriminator(str(path))
  torch.save({"gail_state_dict": GailDiscriminator(5).state_dict()}, path)
  with pytest.raises(RuntimeError, match="size mismatch"):
    runner._load_frozen_discriminator(str(path))
  with pytest.raises(RuntimeError, match="frozen"):
    runner._update_discriminator([])


def test_frozen_runner_rejects_resume_before_parent_init(monkeypatch):
  monkeypatch.setattr(
    VelocityOnPolicyRunner,
    "__init__",
    lambda *args, **kwargs: pytest.fail("PPO was initialized before rejecting resume"),
  )
  with pytest.raises(ValueError, match="fresh PPO"):
    GailVelocityOnPolicyRunner(
      None,
      {
        "resume": True,
        "gail": {
          "enabled": True,
          "dataset_path": "expert.pt",
          "frozen": True,
          "discriminator_checkpoint": "model.pt",
        },
      },
    )
