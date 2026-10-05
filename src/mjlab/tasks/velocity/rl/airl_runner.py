"""Opt-in state-only AIRL shaping with the existing RSL-RL PPO optimizer."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from mjlab.rl.airl import (
  AirlBatch,
  AirlDiscriminator,
  AirlTransitionDataset,
  airl_discriminator_loss,
)
from mjlab.rl.config import AirlCfg
from mjlab.rl.gail import CommandMatchedSampler, velocity_gail_state
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


@torch.no_grad()
def current_policy_log_prob(
  actor: MLPModel, actor_observations: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
  """Evaluate supplied raw actions without sampling or updating normalization."""
  if type(actor) is not MLPModel or tuple(actor.obs_groups) != ("actor",):
    raise ValueError("AIRL currently requires an MLP actor with only the actor group")
  if not isinstance(actor.distribution, GaussianDistribution):
    raise ValueError("AIRL currently requires GaussianDistribution")
  obs = TensorDict({"actor": actor_observations}, batch_size=[len(actions)])
  latent = actor.get_latent(obs)
  actor.distribution.update(actor.mlp(latent))
  log_prob = actor.get_output_log_prob(actions)
  if log_prob.shape != (len(actions),) or not torch.isfinite(log_prob).all():
    raise ValueError("AIRL policy density must be one finite log_prob per action")
  return log_prob.detach()


@dataclass
class _AirlRuntime:
  config: AirlCfg
  expert: AirlTransitionDataset
  validation: AirlTransitionDataset
  discriminator: AirlDiscriminator
  optimizer: torch.optim.Optimizer
  dataset_sha256: str
  command_sampler: CommandMatchedSampler | None


class AirlVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO + lambda*f; exporting g for transfer is a separate experiment."""

  def __init__(self, *args, **kwargs):
    train_cfg = args[1] if len(args) > 1 else kwargs["train_cfg"]
    airl_cfg = AirlCfg(**train_cfg.pop("airl", {}))
    airl_cfg.validate()
    if train_cfg.pop("gail", {}).get("enabled", False):
      raise ValueError("The AIRL runner cannot enable GAIL simultaneously")
    self._airl_training_contract = copy.deepcopy(
      {
        key: train_cfg[key]
        for key in (
          "seed",
          "num_steps_per_env",
          "actor",
          "critic",
          "algorithm",
          "obs_groups",
        )
      }
    )
    super().__init__(*args, **kwargs)
    self.airl: _AirlRuntime | None = None
    if not airl_cfg.enabled:
      return
    if self.is_distributed:
      raise ValueError("AIRL discriminator synchronization is not implemented")
    actor = self.alg.actor
    if type(actor) is not MLPModel or tuple(actor.obs_groups) != ("actor",):
      raise ValueError(
        "AIRL requires a non-recurrent MLP actor with only actor observations"
      )
    if not isinstance(actor.distribution, GaussianDistribution):
      raise ValueError("AIRL requires a stochastic Gaussian actor")
    if self.alg.critic.is_recurrent or self.env.clip_actions is not None:
      raise ValueError("AIRL requires a non-recurrent critic and unclipped raw actions")
    if self.env.cfg.is_finite_horizon:
      raise ValueError(
        "AIRL v1 requires time limits to be truncations, not MDP terminals"
      )
    dataset = AirlTransitionDataset.load(airl_cfg.dataset_path)
    if (
      not airl_cfg.allow_unqualified_expert
      and dataset.metadata.get("gate_passed") is not True
    ):
      raise ValueError(
        "AIRL requires a qualified expert; use allow_unqualified_expert only for smoke tests"
      )
    terms = dataset.metadata.get("actor_observation_terms")
    if (
      terms is not None
      and list(self.env.unwrapped.observation_manager.active_terms["actor"]) != terms
    ):
      raise ValueError("AIRL expert actor observation terms do not match the learner")
    if dataset._data["actor_observations"].shape[1] != actor.obs_dim:
      raise ValueError("AIRL expert actor observations do not match the learner input")
    if dataset._data["actions"].shape[1] != self.env.num_actions:
      raise ValueError("AIRL expert action dimension does not match the environment")
    train, validation = dataset.split_episode_stratified(seed=self.cfg["seed"])
    expert = AirlTransitionDataset(train._data, train.metadata)
    held_out = AirlTransitionDataset(validation._data, validation.metadata)
    features = torch.cat((expert._data["observations"], expert._data["commands"]), -1)
    discriminator = AirlDiscriminator(
      68,
      3,
      airl_cfg.hidden_dims,
      self.alg.gamma,
      features.mean(0),
      features.std(0, unbiased=False),
    ).to(self.device)
    digest = hashlib.sha256()
    with Path(airl_cfg.dataset_path).open("rb") as stream:
      for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    self.airl = _AirlRuntime(
      airl_cfg,
      expert,
      held_out,
      discriminator,
      torch.optim.Adam(discriminator.parameters(), lr=airl_cfg.learning_rate),
      digest.hexdigest(),
      CommandMatchedSampler(expert._data["commands"])
      if airl_cfg.match_expert_commands
      else None,
    )
    self.env.unwrapped.set_transition_capture(self._capture_transition)
    print(
      f"[AIRL] train={len(expert)} holdout={len(held_out)} weight={airl_cfg.weight}"
    )

  def _capture_transition(self, env) -> dict[str, torch.Tensor]:
    assert self.airl is not None
    commands = env.command_manager.get_command(self.airl.config.command_name)
    if not isinstance(commands, torch.Tensor):
      raise ValueError("AIRL command is unavailable")
    return {
      "observations": velocity_gail_state(env.scene["robot"]),
      "commands": commands,
    }

  def _batch(self, data: dict[str, torch.Tensor]) -> AirlBatch:
    return AirlBatch(
      data["observations"],
      data["next_observations"],
      data["commands"],
      data["next_commands"],
      data["terminated"],
      current_policy_log_prob(
        self.alg.actor, data["actor_observations"], data["actions"]
      ),
    )

  def _update_discriminator(self, policy: dict[str, torch.Tensor]) -> dict[str, float]:
    assert self.airl is not None
    runtime = self.airl
    count = min(runtime.config.batch_size, len(policy["actions"]), len(runtime.expert))
    policy_indices = torch.randperm(len(policy["actions"]), device=self.device)[:count]
    policy_data = {key: value[policy_indices] for key, value in policy.items()}
    expert_indices = (
      runtime.command_sampler.sample(policy_data["commands"])
      if runtime.command_sampler is not None
      else torch.randint(len(runtime.expert), (count,))
    )
    expert_data = {
      key: value[expert_indices].to(self.device)
      for key, value in runtime.expert._data.items()
    }
    # Both densities use the same current policy, before PPO updates parameters.
    loss, metrics = airl_discriminator_loss(
      runtime.discriminator, self._batch(expert_data), self._batch(policy_data)
    )
    if not torch.isfinite(loss):
      raise RuntimeError("AIRL discriminator loss is not finite")
    runtime.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
      runtime.discriminator.parameters(), self.alg.max_grad_norm
    )
    runtime.optimizer.step()
    with torch.no_grad():
      indices = torch.arange(min(count, len(runtime.validation)))
      heldout = {
        key: value[indices].to(self.device)
        for key, value in runtime.validation._data.items()
      }
      metrics["heldout_expert_accuracy"] = float(
        (runtime.discriminator(self._batch(heldout)) > 0).float().mean()
      )
    return metrics

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if self.airl is None:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    runtime = self.airl
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    steps = self.cfg["num_steps_per_env"]
    policy: dict[str, torch.Tensor] | None = None
    for it in range(start_it, total_it):
      start = time.perf_counter()
      sums = torch.zeros(4, device=self.device)
      with torch.inference_mode():
        for step in range(steps):
          actions = self.alg.act(obs)
          # PPO already evaluated these raw samples with the current policy.
          log_prob = self.alg.transition.actions_log_prob
          assert log_prob is not None
          if (
            log_prob.shape != (self.env.num_envs,) or not torch.isfinite(log_prob).all()
          ):
            raise ValueError(
              "AIRL policy density must be one finite log_prob per action"
            )
          current = self._capture_transition(self.env.unwrapped)
          # Clone mutable command/state buffers before env.step resamples/resets.
          current = {
            key: value.detach().clone().to(self.device)
            for key, value in current.items()
          }
          current["actor_observations"] = obs["actor"].detach().clone()
          next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          captured = extras.get("transition")
          if captured is None:
            raise RuntimeError("AIRL requires pre-reset transition capture")
          data = {
            **current,
            "actions": actions.detach().clone(),
            "next_observations": captured["observations"].to(self.device),
            "next_commands": captured["commands"].to(self.device),
            "terminated": captured["terminated"].to(self.device),
          }
          batch = AirlBatch(
            data["observations"],
            data["next_observations"],
            data["commands"],
            data["next_commands"],
            data["terminated"],
            log_prob,
          )
          base, reward = runtime.discriminator.reward_components(batch)
          sums += torch.stack(
            (rewards.mean(), reward.mean(), base.mean(), log_prob.mean())
          )
          if policy is None:
            # D autograd needs regular tensors. Allocate once, then overwrite
            # each slot after capturing its true pre-reset successor.
            with torch.inference_mode(False):
              policy = {
                key: torch.empty(
                  (steps * self.env.num_envs, *value.shape[1:]),
                  dtype=value.dtype,
                  device=self.device,
                )
                for key, value in data.items()
              }
          slot = slice(step * self.env.num_envs, (step + 1) * self.env.num_envs)
          for key, value in data.items():
            policy[key][slot].copy_(value)
          rewards = rewards.to(self.device) + runtime.config.weight * reward
          obs, dones = next_obs.to(self.device), dones.to(self.device)
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
        self.alg.compute_returns(obs)
      collect_time = time.perf_counter() - start
      start = time.perf_counter()
      assert policy is not None
      metrics = {}
      for _ in range(runtime.config.updates):
        metrics = self._update_discriminator(policy)
      loss_dict = self.alg.update()
      learn_time = time.perf_counter() - start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )
      if self.logger.writer is not None:
        means = dict(
          zip(
            ("env_reward", "reward", "base_reward", "log_prob"),
            sums.tolist(),
            strict=True,
          )
        )
        for key, value in means.items():
          self.logger.writer.add_scalar(f"AIRL/{key}_mean", value / steps, it)
        self.logger.writer.add_scalar(
          "AIRL/weighted_reward_mean",
          runtime.config.weight * means["reward"] / steps,
          it,
        )
        for key, value in metrics.items():
          self.logger.writer.add_scalar(f"AIRL/discriminator_{key}", value, it)
        if torch.device(self.device).type == "cuda":
          self.logger.writer.add_scalar(
            "AIRL/peak_vram_bytes", torch.cuda.max_memory_allocated(self.device), it
          )
        if it % self.cfg["save_interval"] == 0:
          self.save(f"{self.logger.log_dir}/model_{it}.pt")
    if self.logger.writer is not None:
      self.save(f"{self.logger.log_dir}/model_{self.current_learning_iteration}.pt")
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    if self.airl is not None:
      infos = {
        **(infos or {}),
        "airl_state_dict": self.airl.discriminator.state_dict(),
        "airl_optimizer_state_dict": self.airl.optimizer.state_dict(),
        "airl_config": asdict(self.airl.config),
        "airl_dataset_sha256": self.airl.dataset_sha256,
        "airl_training_contract": self._airl_training_contract,
        "airl_next_iteration": self.current_learning_iteration + 1,
        "airl_torch_rng": torch.get_rng_state(),
        "airl_cuda_rng": torch.cuda.get_rng_state(self.device)
        if torch.device(self.device).type == "cuda"
        else None,
      }
    super().save(path, infos)

  def load(
    self, path: str, load_cfg=None, strict: bool = True, map_location=None
  ) -> dict:
    # Inspect AIRL state before mutating PPO when this is an explicit resume.
    saved = torch.load(path, map_location=map_location, weights_only=False)
    airl_infos = saved.get("infos") or {}
    resume = self.cfg.get("resume", False)
    restore_airl = (
      self.airl is not None and load_cfg is None and "airl_state_dict" in airl_infos
    )
    if self.airl is not None and resume and (not restore_airl):
      raise ValueError("AIRL resume requires a complete AIRL checkpoint and full load")
    if restore_airl:
      assert self.airl is not None
      required = {
        "airl_state_dict",
        "airl_optimizer_state_dict",
        "airl_config",
        "airl_dataset_sha256",
        "airl_next_iteration",
        "airl_torch_rng",
        "airl_training_contract",
      }
      if not required.issubset(airl_infos):
        raise ValueError("AIRL resume requires a complete AIRL checkpoint")
      if airl_infos["airl_training_contract"] != self._airl_training_contract:
        raise ValueError("AIRL resume PPO/episode split configuration mismatch")
      if airl_infos["airl_dataset_sha256"] != self.airl.dataset_sha256:
        raise ValueError("AIRL resume dataset SHA-256 mismatch")
      saved_cfg = dict(airl_infos["airl_config"])
      active_cfg = asdict(self.airl.config)
      saved_cfg.pop("dataset_path", None)
      active_cfg.pop("dataset_path", None)
      if saved_cfg != active_cfg or float(
        airl_infos["airl_state_dict"]["gamma"]
      ) != float(self.airl.discriminator.gamma):
        raise ValueError("AIRL resume reward configuration mismatch")
    infos = super().load(path, load_cfg, strict, map_location)
    if restore_airl:
      assert self.airl is not None
      self.airl.discriminator.load_state_dict(airl_infos["airl_state_dict"])
      self.airl.optimizer.load_state_dict(airl_infos["airl_optimizer_state_dict"])
      self.current_learning_iteration = airl_infos["airl_next_iteration"]
      torch.set_rng_state(airl_infos["airl_torch_rng"].cpu())
      if (
        torch.device(self.device).type == "cuda"
        and airl_infos.get("airl_cuda_rng") is not None
      ):
        torch.cuda.set_rng_state(airl_infos["airl_cuda_rng"].cpu(), self.device)
    return infos
