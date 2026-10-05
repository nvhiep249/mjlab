from pathlib import Path

import pytest

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.scripts.gail_curriculum_comparison import _normalize_legacy_agent_config
from mjlab.scripts.gail_curriculum_extension import (
  FINAL_COMMANDS,
  ITERATIONS,
  build_extension_config,
  expected_final_iteration,
  validate_extension_checkpoint,
)
from mjlab.scripts.gail_resume_ablation import build_ablation_config
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_extension_uses_explicit_corrected_dataset_and_online_discriminator():
  dataset = Path("expert_v3.pt")
  cfg = build_extension_config(
    "gail", Path("v3/model_999.pt"), Path("logs"), dataset=dataset
  )
  assert isinstance(cfg.agent, RslRlOnPolicyRunnerCfg)
  assert cfg.agent.gail.dataset_path == str(dataset)
  assert cfg.agent.gail.frozen is False
  assert cfg.agent.gail.updates == 1
  assert cfg.agent.gail.weight == 0.01
  assert cfg.checkpoint_file == Path("v3/model_999.pt")


def test_legacy_config_accepts_only_disabled_new_airl_block():
  expected = {}
  disabled = {"airl": {"enabled": "false"}}
  _normalize_legacy_agent_config(expected, disabled)
  assert expected == disabled
  enabled = {"airl": {"enabled": "true"}}
  expected = {}
  _normalize_legacy_agent_config(expected, enabled)
  assert expected != enabled
  expected = {"airl": {"enabled": "true"}}
  _normalize_legacy_agent_config(expected, disabled)
  assert expected != disabled


def test_extension_configs_hold_the_final_curriculum_from_step_zero():
  ppo = build_extension_config("ppo", Path("ppo.pt"), Path("logs"))
  gail = build_extension_config("gail", Path("gail.pt"), Path("logs"))
  assert isinstance(ppo.agent, RslRlOnPolicyRunnerCfg)
  assert isinstance(gail.agent, RslRlOnPolicyRunnerCfg)
  assert isinstance(ppo.env.commands["twist"], UniformVelocityCommandCfg)
  assert isinstance(gail.env.commands["twist"], UniformVelocityCommandCfg)

  assert ppo.env.commands["twist"].target_speed_curriculum_stages == (
    (0, FINAL_COMMANDS),
  )
  assert gail.env.commands["twist"].target_speed_curriculum_stages == (
    (0, FINAL_COMMANDS),
  )
  assert ppo.agent.max_iterations == gail.agent.max_iterations == ITERATIONS
  assert ppo.agent.gail.enabled is False
  assert gail.agent.gail.enabled is True


def test_extension_iteration_labels_continue_from_model_999():
  assert expected_final_iteration(999, ITERATIONS) == 1498


def test_extension_checkpoint_must_match_its_arm_and_iteration():
  with pytest.raises(ValueError, match="model_999"):
    validate_extension_checkpoint("ppo", {"iter": 998})
  with pytest.raises(ValueError, match="PPO"):
    validate_extension_checkpoint(
      "ppo", {"iter": 999, "infos": {"gail_state_dict": {}}}
    )
  with pytest.raises(ValueError, match="GAIL"):
    validate_extension_checkpoint("gail", {"iter": 999, "infos": {}})


def test_resume_ablation_changes_only_gail_and_budget():
  from mjlab.rl.config import RslRlOnPolicyRunnerCfg
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  on = build_ablation_config("on", Path("gail.pt"), Path("logs"))
  off = build_ablation_config("off", Path("gail.pt"), Path("logs"))
  assert isinstance(on.agent, RslRlOnPolicyRunnerCfg)
  assert isinstance(off.agent, RslRlOnPolicyRunnerCfg)
  command = on.env.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)

  assert on.agent.max_iterations == off.agent.max_iterations == 200
  assert on.agent.gail.enabled is True
  assert off.agent.gail.enabled is False
  assert command.target_speed_curriculum_stages == ((0, FINAL_COMMANDS),)
