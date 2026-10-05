from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def sample_frontier_velocity_x(
  random_values: torch.Tensor,
  continuous_range: tuple[float, float],
  frontier_probability: float,
  frontier_values: tuple[float, ...],
) -> torch.Tensor:
  """Sample continuous velocity with optional explicit frontier mass."""
  if not 0.0 <= frontier_probability <= 1.0:
    raise ValueError("frontier_velocity_prob must be between 0 and 1")
  if frontier_probability > 0.0 and not frontier_values:
    raise ValueError("frontier_velocity_values must not be empty")
  values = random_values.uniform_(*continuous_range)
  frontier_mask = torch.rand_like(random_values) < frontier_probability
  if frontier_mask.any():
    choices = torch.tensor(frontier_values, device=random_values.device)
    indices = torch.randint(
      len(choices), (int(frontier_mask.sum()),), device=random_values.device
    )
    values[frontier_mask] = choices[indices]
  return values


def select_target_speed_values(
  common_step_counter: int,
  stages: tuple[tuple[int, tuple[float, ...]], ...],
) -> tuple[float, ...]:
  """Select the latest target-speed curriculum stage reached by the runner."""
  if not stages or stages[0][0] != 0:
    raise ValueError("target_speed_curriculum_stages must start at step 0")
  selected = stages[0][1]
  previous_step = -1
  for step, values in stages:
    if step <= previous_step:
      raise ValueError("target_speed_curriculum_stages must be strictly ordered")
    if not values:
      raise ValueError("target-speed curriculum stages must not be empty")
    previous_step = step
    if common_step_counter >= step:
      selected = values
  return selected


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._pinned_commands: torch.Tensor | None = None
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def pin_commands(self, commands: torch.Tensor | None) -> None:
    """Keep body-local commands fixed across timer resampling and resets.

    Passing None restores the configured sampling distribution.
    """
    if commands is None:
      self._pinned_commands = None
      return
    if commands.shape != self.vel_command_b.shape:
      raise ValueError(f"Pinned commands must have shape {self.vel_command_b.shape}")
    if not torch.isfinite(commands).all():
      raise ValueError("Pinned commands must be finite")
    self._pinned_commands = commands.to(self.vel_command_b).clone()
    self._apply_pinned_commands()

  def _apply_pinned_commands(self, env_ids: torch.Tensor | None = None) -> None:
    commands = self._pinned_commands
    assert commands is not None
    ids = slice(None) if env_ids is None else env_ids
    self.vel_command_b[ids] = commands[ids]
    self.vel_command_w[ids] = commands[ids]
    self.is_heading_env[ids] = False
    self.is_standing_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = False

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if getattr(self, "_pinned_commands", None) is not None:
      self._apply_pinned_commands(env_ids)
      return
    if self.cfg.target_speed_command_grid:
      target_speeds = (1.2, 1.35, 1.5)
      if getattr(self.cfg, "target_speed_curriculum", False):
        target_speeds = select_target_speed_values(
          self._env.common_step_counter,
          self.cfg.target_speed_curriculum_stages,
        )
      targets = torch.tensor(
        tuple((speed, 0.0, 0.0) for speed in target_speeds),
        dtype=self.vel_command_b.dtype,
        device=self.device,
      )
      commands = targets[env_ids.remainder(len(targets))]
      self.vel_command_b[env_ids] = commands
      self.vel_command_w[env_ids] = commands
      self.is_heading_env[env_ids] = False
      self.is_standing_env[env_ids] = False
      self.is_world_env[env_ids] = False
      self.is_forward_env[env_ids] = False
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = sample_frontier_velocity_x(
      r,
      self.cfg.ranges.lin_vel_x,
      getattr(self.cfg, "frontier_velocity_prob", 0.0),
      getattr(self.cfg, "frontier_velocity_values", (-2.0, 2.0, 3.0)),
    )
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    if self.cfg.init_velocity_prob > 0.0:
      assert isinstance(env_ids, torch.Tensor)
      r = torch.empty(len(env_ids), device=self.device)
      init_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob]
      if len(init_ids) > 0:
        # Start these envs already moving at the commanded planar velocity.
        # Safe pre-forward: the body-frame write reads orientation from qpos.
        vel_b = torch.zeros(len(init_ids), 6, device=self.device)
        vel_b[:, :2] = self.vel_command_b[init_ids, :2]
        vel_b[:, 5] = self.vel_command_b[init_ids, 2]
        self.robot.write_root_link_velocity_b_to_sim(vel_b, env_ids=init_ids)
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if getattr(self, "_pinned_commands", None) is not None:
      self._apply_pinned_commands(env_ids)
      return
    # Pure function of the current state; refreshing all envs is safe.
    del env_ids
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[heading_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[heading_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super().compute(dt, env_ids)
    if self._pinned_commands is not None:
      return
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0
  """Probability that an env starts its episode already moving at its sampled
  planar command velocity. Applied on reset only."""
  frontier_velocity_prob: float = 0.0
  """Probability of sampling a configured frontier ``lin_vel_x`` value."""
  frontier_velocity_values: tuple[float, ...] = (-2.0, 2.0, 3.0)
  """Discrete forward-velocity values used by the coverage experiment."""
  target_speed_command_grid: bool = False
  """Cycle the fixed 1.2, 1.35, and 1.5 m/s forward target grid."""
  target_speed_curriculum: bool = False
  """Progress the target grid through easier forward-speed stages."""
  target_speed_curriculum_stages: tuple[tuple[int, tuple[float, ...]], ...] = (
    (0, (0.3, 0.5, 0.8)),
    (4_800, (0.6, 0.9, 1.2)),
    (12_000, (1.0, 1.1, 1.2)),
  )
  """Pairs of common-step threshold and forward speeds used by the curriculum."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
