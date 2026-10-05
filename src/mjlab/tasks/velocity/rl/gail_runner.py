"""Optional GAIL runner for the G1 velocity task."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, cast

import torch
from rsl_rl.utils import check_nan

from mjlab.rl import (
  VELOCITY_GAIL_FEATURE_SCHEMA,
  GailBatch,
  GailDiscriminator,
  discriminator_loss,
  imitation_reward,
  velocity_gail_state,
)
from mjlab.rl.config import GailCfg
from mjlab.rl.gail import CommandMatchedSampler, GailTransitionDataset
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


def guarded_frozen_reward(
  reward: torch.Tensor, batch: GailBatch, cap: float
) -> torch.Tensor:
  """Bound imitation bonus using root gravity and yaw from the same pre-step state.

  State layout is g1_velocity_body_local_v1: height, gravity, linear/angular
  velocity, joints. Upright uses the evaluator's exp(-gravity_xy² / 0.2).
  Yaw tolerance 0.2 rad/s is provisional; this gate is reward shaping, not success.
  """
  if batch.commands is None or batch.observations.shape[1] != 68:
    raise ValueError("Guarded frozen reward requires 68-D state and commands")
  state = batch.observations.to(reward.device)
  commands = batch.commands.to(reward.device)
  upright = torch.exp(-state[:, 1:3].square().sum(dim=1) / 0.2)
  # Zero bonus below upright 0.90; full bonus at the evaluator gate 0.97.
  posture = ((upright - 0.90) / 0.07).clamp(0, 1)
  yaw = torch.exp(-((state[:, 9] - commands[:, 2]) / 0.2).square())
  return reward.clamp(0, cap) * posture * yaw


def _make_expert_batch(
  dataset: GailTransitionDataset,
  input_mode: Literal["state_action", "state_transition"],
) -> GailBatch:
  """Build expert discriminator inputs and reject reset-boundary transitions."""
  data = dataset._data
  commands = data.get("commands")
  if input_mode == "state_action":
    actions = data.get("actions")
    if actions is None:
      raise ValueError("state_action GAIL requires expert actions")
    return GailBatch(data["observations"], actions, commands)
  next_observations = data.get("next_observations")
  if next_observations is None:
    raise ValueError("state_transition GAIL requires expert next_observations")
  valid = ~data.get("dones", torch.zeros(len(dataset), dtype=torch.bool))
  if not valid.any():
    raise ValueError("state_transition GAIL requires a non-terminal transition")
  return GailBatch(
    observations=data["observations"][valid],
    actions=None,
    commands=commands[valid] if commands is not None else None,
    next_observations=next_observations[valid],
  )


@dataclass
class _GailRuntime:
  expert_obs: torch.Tensor
  expert_actions: torch.Tensor
  expert_commands: torch.Tensor | None
  command_name: str
  discriminator: GailDiscriminator
  optimizer: torch.optim.Optimizer
  weight: float
  updates: int
  frozen: bool = False
  reward_cap: float | None = None
  command_sampler: CommandMatchedSampler | None = None


class GailVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner that adds an opt-in discriminator reward to velocity rollouts.

  The discriminator sees the body-local velocity state, action, and commands when
  the expert dataset contains commands, so it is independent of the actor
  observation layout.
  """

  def __init__(self, *args, **kwargs):
    train_cfg = args[1] if len(args) > 1 else kwargs["train_cfg"]
    gail_cfg = GailCfg(**train_cfg.pop("gail", {}))
    gail_cfg.validate()
    if gail_cfg.enabled and gail_cfg.frozen and train_cfg.get("resume", False):
      raise ValueError("Frozen GAIL requires fresh PPO; disable resume")
    super().__init__(*args, **kwargs)
    if not gail_cfg.enabled:
      self.gail = None
      return
    dataset = GailTransitionDataset.load(gail_cfg.dataset_path)
    dataset.require_feature_schema(VELOCITY_GAIL_FEATURE_SCHEMA)
    state_dim = int(dataset[0]["observations"].numel())
    action_dim = int(dataset[0]["actions"].numel())
    command_dim = int(dataset[0].get("commands", torch.empty(0)).numel())
    if gail_cfg.frozen_reward_cap is not None and (state_dim != 68 or command_dim != 3):
      raise ValueError("Guarded frozen reward requires 68-D state and 3-D commands")
    expert_features = GailBatch(
      dataset._data["observations"],
      dataset._data["actions"],
      dataset._data.get("commands"),
    ).features()
    discriminator = GailDiscriminator.from_expert_features(expert_features).to(
      self.device
    )
    self.gail = _GailRuntime(
      expert_obs=dataset._data["observations"],
      expert_actions=dataset._data["actions"],
      expert_commands=dataset._data.get("commands"),
      command_name=gail_cfg.command_name,
      discriminator=discriminator,
      optimizer=torch.optim.Adam(discriminator.parameters(), lr=gail_cfg.learning_rate),
      weight=gail_cfg.weight,
      updates=0 if gail_cfg.frozen else gail_cfg.updates,
      frozen=gail_cfg.frozen,
      reward_cap=gail_cfg.frozen_reward_cap,
      command_sampler=(
        CommandMatchedSampler(dataset._data["commands"])
        if gail_cfg.match_expert_commands
        else None
      ),
    )
    if gail_cfg.frozen:
      self._load_frozen_discriminator(gail_cfg.discriminator_checkpoint)
    print(
      f"[GAIL] dataset={gail_cfg.dataset_path} transitions={len(dataset)} "
      f"state_dim={state_dim} action_dim={action_dim} command_dim={command_dim} "
      f"weight={self.gail.weight} frozen={self.gail.frozen}"
    )

  def _load_frozen_discriminator(self, path: str) -> None:
    assert self.gail is not None
    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    state = checkpoint.get("gail_state_dict")
    if state is None:
      state = (checkpoint.get("infos") or {}).get("gail_state_dict")
    if state is None:
      raise ValueError("Checkpoint does not contain gail_state_dict")
    self.gail.discriminator.load_state_dict(state, strict=True)
    self.gail.discriminator.eval()
    self.gail.discriminator.requires_grad_(False)

  def _state_action(self, actions: torch.Tensor) -> GailBatch:
    robot = self.env.unwrapped.scene["robot"]
    state = velocity_gail_state(robot).detach()
    commands = None
    if self.gail is not None and self.gail.expert_commands is not None:
      commands = self.env.unwrapped.command_manager.get_command(self.gail.command_name)
      if not isinstance(commands, torch.Tensor):
        raise RuntimeError(f"Command '{self.gail.command_name}' is unavailable")
      # The environment resamples this buffer in-place during step/reset.
      commands = commands.detach().clone()
    return GailBatch(state, actions.detach(), commands)

  def _update_discriminator(
    self, policy_batches: list[GailBatch]
  ) -> tuple[float, float]:
    assert self.gail is not None
    if self.gail.frozen:
      raise RuntimeError("Cannot update a frozen GAIL discriminator")
    discriminator = self.gail.discriminator
    optimizer = self.gail.optimizer
    commands = [batch.commands for batch in policy_batches]
    actions = [batch.actions for batch in policy_batches]
    if any(action is None for action in actions):
      raise ValueError("state_action GAIL requires policy actions")
    policy_commands = None
    if commands[0] is not None:
      policy_commands = torch.cat(cast(list[torch.Tensor], commands))
    policy = GailBatch(
      torch.cat([batch.observations for batch in policy_batches]),
      torch.cat(cast(list[torch.Tensor], actions)),
      policy_commands,
    )
    batch_size = min(1024, len(policy.observations), len(self.gail.expert_obs))
    expert_idx = torch.randint(len(self.gail.expert_obs), (batch_size,))
    policy_idx = torch.randperm(len(policy.observations))[:batch_size]
    if self.gail.command_sampler is not None:
      if policy.commands is None:
        raise ValueError("Command matching requires policy commands")
      expert_idx = self.gail.command_sampler.sample(policy.commands[policy_idx])
    expert_commands = self.gail.expert_commands
    expert = GailBatch(
      self.gail.expert_obs[expert_idx].to(self.device),
      self.gail.expert_actions[expert_idx].to(self.device),
      expert_commands[expert_idx].to(self.device)
      if expert_commands is not None
      else None,
    )
    if policy.actions is None:
      raise ValueError("state_action GAIL requires policy actions")
    policy = GailBatch(
      policy.observations[policy_idx],
      policy.actions[policy_idx],
      policy.commands[policy_idx] if policy.commands is not None else None,
    )
    loss, metrics = discriminator_loss(discriminator, expert, policy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return metrics["loss"], metrics["accuracy"]

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if self.gail is None:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    if self.is_distributed:
      raise RuntimeError(
        "GAIL training does not support distributed execution because the "
        "discriminator gradients are not synchronized"
      )
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      rollout_start = time.perf_counter()
      policy_batches: list[GailBatch] = []
      env_reward_sum = 0.0
      gail_reward_sum = 0.0
      raw_gail_reward_sum = 0.0
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          policy_batch = self._state_action(actions.to(self.env.device))
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          gail_reward = imitation_reward(self.gail.discriminator, policy_batch).to(
            rewards.device
          )
          raw_gail_reward_sum += float(gail_reward.mean())
          if self.gail.reward_cap is not None:
            gail_reward = guarded_frozen_reward(
              gail_reward, policy_batch, self.gail.reward_cap
            )
          env_reward_sum += float(rewards.mean())
          gail_reward_sum += float(gail_reward.mean())
          rewards = rewards + self.gail.weight * gail_reward
          if not self.gail.frozen:
            policy_batches.append(policy_batch)
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
        self.alg.compute_returns(obs)
      collect_time = time.perf_counter() - rollout_start
      update_start = time.perf_counter()
      disc_loss, disc_accuracy = 0.0, 0.0
      for _ in range(self.gail.updates):
        disc_loss, disc_accuracy = self._update_discriminator(policy_batches)
      loss_dict = self.alg.update()
      learn_time = time.perf_counter() - update_start
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
        steps = self.cfg["num_steps_per_env"]
        mean_env_reward = env_reward_sum / steps
        mean_gail_reward = gail_reward_sum / steps
        self.logger.writer.add_scalar("GAIL/env_reward_mean", mean_env_reward, it)
        self.logger.writer.add_scalar("GAIL/reward_mean", mean_gail_reward, it)
        self.logger.writer.add_scalar(
          "GAIL/raw_reward_mean", raw_gail_reward_sum / steps, it
        )
        self.logger.writer.add_scalar(
          "GAIL/weighted_reward_mean", self.gail.weight * mean_gail_reward, it
        )
        self.logger.writer.add_scalar("GAIL/frozen", int(self.gail.frozen), it)
        if not self.gail.frozen:
          self.logger.writer.add_scalar("GAIL/discriminator_loss", disc_loss, it)
          self.logger.writer.add_scalar(
            "GAIL/discriminator_accuracy", disc_accuracy, it
          )
        if it % self.cfg["save_interval"] == 0:
          self.save(f"{self.logger.log_dir}/model_{it}.pt")
      if self.gail.frozen:
        print(f"[GAIL] iteration={it} frozen=True")
      else:
        print(
          f"[GAIL] iteration={it} discriminator_loss={disc_loss:.4f} discriminator_accuracy={disc_accuracy:.3f}"
        )
    if self.logger.writer is not None:
      self.save(f"{self.logger.log_dir}/model_{self.current_learning_iteration}.pt")
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    if self.gail is not None:
      infos = {
        **(infos or {}),
        "gail_state_dict": self.gail.discriminator.state_dict(),
        "gail_optimizer_state_dict": self.gail.optimizer.state_dict(),
        "frozen_reward_settings": {
          "weight": self.gail.weight,
          "cap": self.gail.reward_cap,
        },
      }
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg=None,
    strict: bool = True,
    map_location=None,
    *,
    allow_frozen_resume: bool = False,
  ) -> dict:
    if self.gail is not None and self.gail.frozen:
      if not allow_frozen_resume:
        raise ValueError(
          "Frozen GAIL requires fresh PPO; use discriminator_checkpoint, "
          "not a PPO checkpoint or resume"
        )
      if load_cfg is not None or not strict:
        raise ValueError("Frozen resume requires full strict PPO restore")
      checkpoint = torch.load(path, map_location="cpu", weights_only=False)
      settings = checkpoint.get("infos", {}).get("frozen_reward_settings")
      cap = getattr(self.gail, "reward_cap", None)
      if cap is not None or (settings is not None and settings.get("cap") is not None):
        if settings != {"weight": self.gail.weight, "cap": cap}:
          raise ValueError("Frozen resume reward settings mismatch")
      saved = checkpoint.get("infos", {}).get("gail_state_dict", {})
      expected = self.gail.discriminator.state_dict()
      if saved.keys() != expected.keys() or any(
        not torch.equal(saved[key].cpu(), value.cpu())
        for key, value in expected.items()
      ):
        raise ValueError("Frozen resume discriminator mismatch")
      return super().load(path, strict=True, map_location=map_location)
    infos = super().load(path, load_cfg, strict, map_location)
    if self.gail is not None:
      gail_infos = infos or {}
      if "gail_state_dict" not in gail_infos:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        gail_infos = checkpoint
      if "gail_state_dict" in gail_infos:
        self.gail.discriminator.load_state_dict(gail_infos["gail_state_dict"])
        if "gail_optimizer_state_dict" in gail_infos:
          self.gail.optimizer.load_state_dict(gail_infos["gail_optimizer_state_dict"])
    return infos
