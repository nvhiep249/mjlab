"""CPU integration checks with the installed RSL-RL PPO implementation."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.rl.config import AirlCfg, RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.rl.airl_runner import (
  AirlVelocityOnPolicyRunner,
  current_policy_log_prob,
)


def test_qualified_deterministic_expert_uses_current_learner_density(tmp_path):
  path = tmp_path / "deterministic.pt"
  _dataset(path)
  data = torch.load(path, weights_only=True)
  data["metadata"].update(
    expert_policy="deterministic", action_contract="raw_policy_mean", gate_passed=True
  )
  torch.save(data, path)
  cfg = _config(path)
  cfg["airl"]["allow_unqualified_expert"] = False
  runner = AirlVelocityOnPolicyRunner(_CpuEnv(), cfg)
  assert runner.airl is not None
  batch = runner._batch(runner.airl.expert._data)
  assert torch.isfinite(batch.log_prob).all()


def test_current_density_uses_given_actions_without_rng_or_normalizer_updates():
  obs = TensorDict({"actor": torch.randn(5, 4)}, batch_size=[5])
  actor = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    29,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  actions = torch.randn(5, 29)
  before_rng = torch.get_rng_state().clone()
  before_stats = {
    key: value.clone() for key, value in actor.obs_normalizer.state_dict().items()
  }
  actor_obs = obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  result = current_policy_log_prob(actor, actor_obs, actions)
  expected = (
    torch.distributions.Normal(actor(obs), actor.output_std).log_prob(actions).sum(-1)
  )
  torch.testing.assert_close(result, expected)
  assert not result.requires_grad
  assert torch.equal(before_rng, torch.get_rng_state())
  for key, value in actor.obs_normalizer.state_dict().items():
    torch.testing.assert_close(value, before_stats[key])
  with torch.no_grad():
    cast(torch.nn.Linear, actor.mlp[-1]).bias.add_(0.5)
  assert not torch.equal(result, current_policy_log_prob(actor, actor_obs, actions))


class _CpuEnv:
  """Synthetic transitions only; no claim of G1 locomotion qualification."""

  device = torch.device("cpu")
  num_envs = 4
  num_actions = 29
  max_episode_length = 5
  clip_actions = None

  def __init__(self):
    self.cfg = SimpleNamespace(is_finite_horizon=False)
    self.unwrapped = self
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(4, dtype=torch.long)
    self.states = torch.zeros(4, 68)
    self.commands = torch.zeros(4, 3)
    self.scene = {"robot": self}
    self.command_manager = SimpleNamespace(get_command=lambda _: self.commands)
    self.capture = None
    self.obs_buf = {"actor": torch.zeros(4, 4), "critic": torch.zeros(4, 4)}

  def set_transition_capture(self, callback):
    self.capture = callback

  def get_observations(self):
    return TensorDict(self.obs_buf, batch_size=[4]).clone()

  def step(self, actions):
    self.common_step_counter += 1
    self.episode_length_buf += 1
    self.states[:, :29] += actions * 0.01
    self.obs_buf = {
      "actor": self.states[:, :4].clone(),
      "critic": self.states[:, :4].clone(),
    }
    terminated = torch.zeros(4, dtype=torch.bool)
    truncated = self.episode_length_buf >= self.max_episode_length
    assert self.capture is not None
    transition = {key: value.clone() for key, value in self.capture(self).items()}
    transition.update(terminated=terminated, truncated=truncated)
    self.states[truncated] = 0
    self.episode_length_buf[truncated] = 0
    for value in self.obs_buf.values():
      value[truncated] = 0
    return (
      self.get_observations(),
      -actions.square().mean(-1),
      truncated.long(),
      {
        "transition": transition,
        "time_outs": truncated,
      },
    )


def _dataset(path):
  generator = torch.Generator().manual_seed(19)
  n = 32
  data = {
    "observations": torch.randn(n, 68, generator=generator),
    "next_observations": torch.randn(n, 68, generator=generator),
    "actor_observations": torch.randn(n, 4, generator=generator),
    "next_actor_observations": torch.randn(n, 4, generator=generator),
    "actions": torch.randn(n, 29, generator=generator),
    "commands": torch.zeros(n, 3),
    "next_commands": torch.zeros(n, 3),
    "terminated": torch.zeros(n, dtype=torch.bool),
    "truncated": torch.zeros(n, dtype=torch.bool),
    "dones": torch.zeros(n, dtype=torch.bool),
    "episode_ids": torch.arange(n) // 4,
    "environment_ids": torch.zeros(n, dtype=torch.long),
    "metadata": {
      "feature_schema": "g1_velocity_body_local_v1",
      "transition_contract": "pre_reset_v1",
      "expert_policy": "stochastic",
      "action_contract": "raw_policy_sample",
      "actor_observation_contract": "policy_input_v1",
    },
  }
  torch.save(data, path)


def _config(path):
  cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4,
    logger="tensorboard",
    save_interval=2,
    upload_model=False,
    airl=AirlCfg(
      enabled=True,
      dataset_path=str(path),
      hidden_dims=(8,),
      allow_unqualified_expert=True,
    ),
  )
  cfg.actor.hidden_dims = (8,)
  cfg.critic = RslRlModelCfg(hidden_dims=(8,))
  cfg.algorithm.num_learning_epochs = 1
  cfg.algorithm.num_mini_batches = 1
  return asdict(cfg)


def test_cpu_rollout_updates_reward_and_policy_and_resumes(tmp_path, monkeypatch):
  from mjlab.tasks.velocity.rl import airl_runner

  monkeypatch.setattr(
    airl_runner, "velocity_gail_state", lambda robot: robot.states.clone()
  )
  monkeypatch.setattr(
    AirlVelocityOnPolicyRunner, "export_policy_to_onnx", lambda *a, **k: None
  )
  dataset = tmp_path / "expert.pt"
  _dataset(dataset)
  runner = AirlVelocityOnPolicyRunner(
    _CpuEnv(), _config(dataset), str(tmp_path / "run")
  )
  before_actor = [param.detach().clone() for param in runner.alg.actor.parameters()]
  assert runner.airl is not None
  before_reward = [
    param.detach().clone() for param in runner.airl.discriminator.parameters()
  ]
  runner.learn(5)
  assert any(
    not torch.equal(old, new)
    for old, new in zip(before_actor, runner.alg.actor.parameters(), strict=True)
  )
  assert any(
    not torch.equal(old, new)
    for old, new in zip(
      before_reward, runner.airl.discriminator.parameters(), strict=True
    )
  )
  for param in runner.airl.discriminator.parameters():
    assert torch.isfinite(param).all()
  checkpoint = tmp_path / "run" / "model_4.pt"
  cfg = _config(dataset)
  cfg["resume"] = True
  resumed = AirlVelocityOnPolicyRunner(_CpuEnv(), cfg, str(tmp_path / "resume"))
  resumed.load(str(checkpoint), map_location="cpu")
  assert resumed.current_learning_iteration == 5
  assert resumed.airl is not None
  for key, value in runner.airl.discriminator.state_dict().items():
    torch.testing.assert_close(resumed.airl.discriminator.state_dict()[key], value)
  resumed.learn(1)
  assert (tmp_path / "resume" / "model_5.pt").is_file()
  # Full AIRL loading always checks compatibility, even without resume=True.
  incompatible = _config(dataset)
  incompatible["seed"] = 43
  other = AirlVelocityOnPolicyRunner(_CpuEnv(), incompatible)
  with pytest.raises(ValueError, match="split configuration"):
    other.load(str(checkpoint), map_location="cpu")
  # Actor-only warm start must leave the fresh reward/optimizer/RNG alone.
  assert other.airl is not None
  fresh_reward = {
    key: value.clone() for key, value in other.airl.discriminator.state_dict().items()
  }
  other.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
  assert other.current_learning_iteration == 0
  for key, value in fresh_reward.items():
    torch.testing.assert_close(other.airl.discriminator.state_dict()[key], value)


def test_airl_and_gail_cannot_be_enabled_together(tmp_path):
  cfg = _config(tmp_path / "unused.pt")
  cfg["gail"]["enabled"] = True
  with pytest.raises(ValueError, match="GAIL"):
    AirlVelocityOnPolicyRunner(_CpuEnv(), cfg)


def test_rollout_reuses_density_and_regular_storage_without_losing_pre_reset(
  tmp_path, monkeypatch
):
  from mjlab.tasks.velocity.rl import airl_runner

  monkeypatch.setattr(
    airl_runner, "velocity_gail_state", lambda robot: robot.states.clone()
  )
  path = tmp_path / "expert.pt"
  _dataset(path)
  runner = AirlVelocityOnPolicyRunner(_CpuEnv(), _config(path))
  original_density = airl_runner.current_policy_log_prob
  density_sizes = []

  def density(actor, observations, actions):
    density_sizes.append(len(actions))
    return original_density(actor, observations, actions)

  monkeypatch.setattr(airl_runner, "current_policy_log_prob", density)
  original_act = runner.alg.act

  def act(obs):
    actions = original_act(obs)
    torch.testing.assert_close(
      runner.alg.transition.actions_log_prob,
      original_density(runner.alg.actor, obs["actor"], actions),
    )
    return actions

  monkeypatch.setattr(runner.alg, "act", act)
  original_update = runner._update_discriminator
  pointers = []

  def update(policy):
    assert not torch.is_inference(policy["observations"])
    pointers.append(policy["observations"].data_ptr())
    torch.testing.assert_close(
      policy["next_observations"][:, :29] - policy["observations"][:, :29],
      policy["actions"] * 0.01,
      atol=1e-7,
      rtol=1e-5,
    )
    # Real D backward must work on the reused normal tensors.
    return original_update(policy)

  monkeypatch.setattr(runner, "_update_discriminator", update)
  runner.learn(2)
  assert pointers[0] == pointers[1]
  # Only expert + policy D minibatches and held-out diagnostics evaluate density.
  assert density_sizes == [16, 16, 8] * 2


def test_disabled_airl_delegates_baseline(tmp_path, monkeypatch):
  from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

  cfg = _config(tmp_path / "unused.pt")
  cfg["airl"]["enabled"] = False
  runner = AirlVelocityOnPolicyRunner(_CpuEnv(), cfg)
  called = []
  monkeypatch.setattr(
    VelocityOnPolicyRunner, "learn", lambda *args: called.append(args)
  )
  runner.learn(1)
  assert len(called) == 1
  assert isinstance(runner.env, _CpuEnv)
  assert runner.env.capture is None


def test_unqualified_expert_rejected_outside_smoke(tmp_path):
  path = tmp_path / "unqualified.pt"
  _dataset(path)
  cfg = _config(path)
  cfg["airl"]["allow_unqualified_expert"] = False
  with pytest.raises(ValueError, match="qualified expert"):
    AirlVelocityOnPolicyRunner(_CpuEnv(), cfg)
