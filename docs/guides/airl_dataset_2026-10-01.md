# Deterministic expert dataset and AIRL MVP handoff

This supersedes the earlier claim that AIRL demonstrations require a stochastic
expert. AIRL evaluates both expert and learner actions under the current
**learner** density, `log pi_learner(a | actor_observation)`. Expert deterministic
raw means are supported; the learner must still have the supported Gaussian
distribution. See the [official AIRL API](https://imitation.readthedocs.io/en/latest/algorithms/airl.html).
Deterministic demonstrations are practical imitation data; they do not establish
the paper's ideal maximum-entropy expert assumptions or guarantee recovery of
a transferable ground-truth reward.

## Qualification

Current GPU evaluations use the unchanged gate: success >=95%, linear RMSE
<=0.25, yaw RMSE <=0.20, upright >=0.97, at `(1.2, 0, 0)`.

| Source | Seed | Episodes | Success | Linear RMSE | Yaw RMSE | Falls | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| teacher35550 | 42002 | 100 | 99% | 0.212479 | 0.172291 | 0 | PASS |
| teacher35550 | 42003 | 100 | 100% | 0.211104 | 0.171173 | 0 | PASS |
| teacher35550 | 42004 | 100 | 97% | 0.214578 | 0.174795 | 0 | PASS |
| fine-tuned35749 | 42002 | 100 | 99% | 0.212021 | 0.161166 | 0 | PASS |

Selected teacher35550 because it passed all three independent seed gates.
The newer teacher is slightly better on yaw in one seed, which does not yet
establish superiority. These are finite-sample simulation results, not a
statistical guarantee of >=95% success for every initial-state distribution.

Gate JSONs live in `report/airl_smoke_20261001/teacher*_deterministic_gate_*.json`.
`teacher35550_deterministic_qualified.json` preserves the first gate's stats
and references all three validation reports; its first bin is not an aggregate
of 300 episodes. Teacher SHA256:
`6fc1028d35ad7b56bb3389ed8a6b014fe7083e72d22f1db8ca4d73af37bd94c0`.

## Dataset

File: `experiments/g1_velocity/airl_expert_1p2_deterministic_v1.pt`.
Size: 382,726,465 bytes; SHA256:
`3cf8b910c4957fc80d257762d55945e5777411ba2785659dfb29dd08d69e72e8`.
Collection seed 43001, 256 envs, 1000 steps/env (20 simulated seconds),
256,000 transitions. Rollout collector reported 35 seconds; this excludes
gate evaluation, simulator initialization, file serialization, and audit.

Strict metadata: `g1_velocity_body_local_v1`, `pre_reset_v1`, `policy_input_v1`,
`expert_policy=deterministic`, `action_contract=raw_policy_mean`. Dimensions:
state 68, raw action 29, actor input 99, command 3. Current and successor actor
inputs are stored for evaluating actions under the learner. No action clipping
or artificial Gaussian noise. Provenance includes the exact checkpoint hash,
gate report, actor terms, seed, environment/episode IDs, and termination flags.

Audit: 256 complete episodes, all timeout, zero true terminations; success
256/256, mean episode linear RMSE 0.212917, yaw RMSE 0.172796, upright 0.996235.
No episode filtering. Verified true successor continuity for all nonterminal
steps and fixed commands. This collection seed is separate from qualification
seeds. The dataset's own success is a data audit, not another held-out gate.
Seed-42 split: 205 episodes / 205,000 train transitions; 51 episodes / 51,000
holdout transitions. No shared episodes. The dataset covers only the 1.2 m/s
MVP command, not 1.35/1.5 m/s, lateral commands, or turns.

## User-run training command

No AIRL training has been started. Run in PowerShell:

```powershell
& D:\Robotics\mjlab\scripts\training\train_airl_1p2.ps1
```

Optional parser-only check: append `-CheckOnly`. Launcher fixes the main
workspace interpreter and source path, including when a worktree virtualenv
is activated. The command chooses the opt-in AIRL-MVP task, which reuses the
fixed `(1.2, 0, 0)` teacher environment and current flat-task rewards/randomization.
Gaussian learner remains stochastic. GAIL is disabled; AIRL qualified-data
guard stays enabled, command matching enabled, provisional weight 0.01.

Learner source is the established common `model_999.pt` checkpoint, **not the
expert**. Standard checkpoint loading restores PPO actor/critic/optimizer and
iteration 999; the newly created AIRL reward/optimizer is fresh. Loaded PPO LR
is 0.0000759375, adaptive. Budget is 200 additional updates, 1024 envs,
24 steps/env, 4,915,200 transitions; expected final label is model_1198.pt,
AIRL next-iteration 1199. This is a pilot, not a matched multi-seed benchmark.
Pilot log root: `logs/rsl_rl/g1_velocity_airl_1p2`.

Preflight parsed the exact training arguments, constructed 1024 real G1 envs
on CUDA, loaded the common base, evaluated finite expert-action densities and
AIRL discriminator values, and stepped three stochastic learner action batches.
Zero optimizer steps. Expert-action log-probabilities range from -605.66 to
9.12 on the sampled batch: finiteness is checked, but this broad range may
make discriminator training difficult and requires monitoring during the pilot.
No convergence or AIRL task-performance improvement is claimed.

Verification: focused suite 66 passed, targeted Ruff clean, Pyright zero
errors/warnings, PowerShell launcher CheckOnly passed under injected old
worktree environment. Runnable audit:
`uv run --no-sync python .tmp/verify_deterministic_airl.py`.
Evidence: `report/airl_smoke_20261001/deterministic_dataset_audit.json`,
`deterministic_dataset_audit_console.log`, `deterministic_collection_console.log`,
and `airl_launcher_check.log`.

After training, evaluate saved checkpoints for actual velocity/yaw RMSE,
success and falls on independent seeds. Compare PPO/GAIL/AIRL only under the
same command support, base checkpoint, PPO settings, seeds and transition
budget. Task reward and discriminator accuracy alone do not establish improvement.
