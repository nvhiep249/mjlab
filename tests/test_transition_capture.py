"""True successor capture must survive auto-reset without tensor aliasing."""

from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.utils.noise import UniformNoiseCfg


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_transition_capture_precedes_reset_and_clones(device):
  if device.startswith("cuda") and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 2
  cfg.episode_length_s = 0.05
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    env.set_transition_capture(
      lambda inner: {
        "episode_length": inner.episode_length_buf,
        "next_actor_observations": cast(torch.Tensor, inner.obs_buf["actor"]),
      }
    )
    _, _, terminated, truncated, extras = env.step(torch.zeros(2, 1, device=device))
    snapshot = extras["transition"]
    assert (terminated | truncated).all()
    assert snapshot["episode_length"].tolist() == [1, 1]
    assert env.episode_length_buf.tolist() == [0, 0]
    torch.testing.assert_close(snapshot["truncated"], truncated)
    env.set_transition_capture(None)
    assert "transition" not in env.step(torch.zeros(2, 1, device=device))[-1]
    assert snapshot["episode_length"].tolist() == [1, 1]
  finally:
    env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_transition_capture_ordinary_observation_is_returned_frame(device):
  if device.startswith("cuda") and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 2
  cfg.episode_length_s = 1.0
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    env.set_transition_capture(
      lambda inner: {"actor_observations": cast(torch.Tensor, inner.obs_buf["actor"])}
    )
    obs, _, terminated, truncated, extras = env.step(torch.zeros(2, 1, device=device))
    assert not (terminated | truncated).any()
    torch.testing.assert_close(extras["transition"]["actor_observations"], obs["actor"])
  finally:
    env.close()


def test_transition_capture_mixed_reset_preserves_noisy_ordinary_frame_cpu():
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 2
  cfg.episode_length_s = 0.1
  cart_pos = cfg.observations["actor"].terms["cart_pos"]
  assert cart_pos is not None
  cart_pos.noise = UniformNoiseCfg(n_min=-0.5, n_max=0.5)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    env.episode_length_buf[0] = env.max_episode_length - 1
    env.set_transition_capture(
      lambda inner: {"actor_observations": cast(torch.Tensor, inner.obs_buf["actor"])}
    )
    obs, _, terminated, truncated, extras = env.step(torch.zeros(2, 1))
    assert (terminated | truncated).tolist() == [True, False]
    captured = extras["transition"]["actor_observations"]
    actor_obs = cast(torch.Tensor, obs["actor"])
    torch.testing.assert_close(captured[1], actor_obs[1], rtol=0, atol=0)
    torch.testing.assert_close(env.get_observations()["actor"], obs["actor"])
    assert not torch.equal(captured[0], actor_obs[0])
  finally:
    env.close()


def test_merge_reset_observations_preserves_nested_ordinary_rows():
  previous = {"nested": {"term": torch.tensor([[1.0], [2.0]])}}
  refreshed = {"nested": {"term": torch.tensor([[3.0], [4.0]])}}
  merged = ManagerBasedRlEnv._merge_reset_observations(
    previous, refreshed, torch.tensor([0])
  )
  assert merged["nested"]["term"].tolist() == [[3.0], [2.0]]
