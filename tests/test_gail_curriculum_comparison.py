from pathlib import Path

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.scripts.gail_curriculum_comparison import build_config


def test_build_config_accepts_verified_dataset_path():
  cfg = build_config(1, Path("logs"), dataset=Path("expert_v3.pt"))
  assert isinstance(cfg.agent, RslRlOnPolicyRunnerCfg)
  assert cfg.agent.gail.dataset_path == "expert_v3.pt"
