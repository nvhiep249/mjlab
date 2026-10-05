# mjlab — PPO + AIRL trên Kaggle và frozen reward

Quy trình cho Unitree G1: **clone GitHub → kiểm tra dữ liệu → smoke → benchmark → train PPO/AIRL cùng budget → evaluate → export discriminator → tích hợp frozen reward vào PPO**.

Notebook publish hiện là **continuation từ PPO999**, mặc định **60 updates bổ sung/arm tại 16.384 env**. Đây là pilot, chưa phải train mới từ đầu hoặc kết quả hội tụ. Export/load frozen AIRL đã có code độc lập; runner AIRL hiện vẫn cập nhật D online, chưa có launcher PPO frozen AIRL hoàn chỉnh.

## 1. File và setup Kaggle

| File | Mục đích |
|---|---|
| [kaggle_airl.ipynb](scripts/cloud/kaggle_airl.ipynb) | Import vào Kaggle; clone source từ GitHub |
| [kaggle_airl.py](scripts/cloud/kaggle_airl.py) | CLI benchmark, chọn env/minibatches, train từng arm |
| [kaggle_source_manifest.json](scripts/cloud/kaggle_source_manifest.json) | SHA256 source được kiểm tra sau clone |
| [frozen_airl.py](src/mjlab/rl/frozen_airl.py) | Export/load và tính reward AIRL cố định |
| [Hướng dẫn benchmark](docs/guides/kaggle_airl_2026-10-05.md) | VRAM math, phép đo và quy tắc chọn cấu hình |
| [Hợp đồng dataset](docs/guides/airl_dataset_2026-10-01.md) | Expert, schema, qualification |

1. Lấy notebook từ [branch `feature/kaggle-airl-16k`](https://github.com/nvhiep249/mjlab/tree/feature/kaggle-airl-16k), import vào Kaggle Notebook.
2. Bật **Internet**, chọn **Tesla T4**, dùng một GPU (`GPU = 0`). Stack hiện tại không hỗ trợ P100; AIRL chưa đồng bộ D qua DDP nên không dùng hai GPU.
3. Attach Dataset [nvhiep2409/airl-ppo](https://www.kaggle.com/datasets/nvhiep2409/airl-ppo), có đường dẫn `/kaggle/input/datasets/nvhiep2409/airl-ppo`.

Notebook tìm đệ quy hai file dưới đường dẫn trên:

| Artifact | Tên chấp nhận | Bytes |
|---|---|---:|
| Expert đã qualify tại `(1.2, 0, 0)` | `airl_expert_1p2_deterministic_v1.pt` | 382726465 |
| PPO chung khởi tạo hai arm | `ppo_common_999.pt` hoặc `model_999.pt` | 5316259 |

Hash expert: `3cf8b910c4957fc80d257762d55945e5777411ba2785659dfb29dd08d69e72e8`.
Hash PPO chung: `b318a6da5db37a6795e4a943d0d52eaa75743bd403341ecf47b10b3f20972dc8`.
Tên trùng hoặc hash sai sẽ dừng trước train. Teacher tạo expert và PPO999 khởi tạo learner là hai checkpoint khác nhau. Không cần upload source ZIP; ZIP cũ trong Dataset không được dùng.

Cell cấu hình:

```python
REPO_URL = "https://github.com/nvhiep249/mjlab.git"
REPO_BRANCH = "feature/kaggle-airl-16k"
REPO_COMMIT = None  # Hoặc full SHA để chạy lại đúng source.
INPUT_ROOT = Path("/kaggle/input/datasets/nvhiep2409/airl-ppo")
OUTPUT = Path("/kaggle/working/outputs")
GPU = 0
SESSION = "pilot01"  # Đổi tên khi chạy lượt mới, giữ output cũ.
TRANSITIONS = 23592960  # Budget mỗi arm.
```

Source/venv ở `/tmp/mjlab-github-<SESSION>`, cache ở `/tmp/uv-cache`. Notebook clone branch, ghi commit thực tế vào `<SESSION>_github_source.json`, kiểm tra manifest, rồi bootstrap **Python 3.11**:

```bash
uv sync --locked --no-dev --extra cu128 --python 3.11
```

Runtime T4 bạn đã gửi: Torch `2.9.0+cu128`, CUDA `12.8`, Warp `1.14.0`, MuJoCo/MuJoCo-Warp `3.11.0`, RSL-RL `5.5.0`. Preflight kiểm tra phép tính CUDA thật và Warp, không chỉ đọc `nvidia-smi`.

## 2. Thứ tự chạy và cách đọc kết quả

| Bước | Việc làm | Output trong `/kaggle/working/outputs` |
|---|---|---|
| Input / clone / bootstrap | Xác minh data/source, tạo môi trường | `<SESSION>_github_source.json` |
| Smoke | PPO/AIRL: 1024 env, 5 updates; AIRL resume thêm 1 | `<SESSION>_smoke_*` |
| Baseline | PPO999 tại 1.2 m/s, 100 episodes | `<SESSION>_common_base_seed42005.json` |
| Sweep | 4 mức env × 4/8 minibatches, 3 warmup + 8 measured updates/candidate | `<SESSION>_sweep/selected.json` và logs |
| Paired pilot | Hai arm tuần tự, cùng COMMON, seed/config/budget | `<SESSION>_pilot_ppo`, `<SESSION>_pilot_airl` |
| Endpoint | 100 episodes/seed/arm, seeds 42005 và 42006 | `<SESSION>_<arm>_eval_seed*.json` |
| Export | Thêm cell ở mục 4, chỉ lấy checkpoint AIRL | `*_frozen_airl_reward.pt` |

**Log `999–1005`** thuộc smoke vài updates, kế thừa nhãn iteration PPO999. Không có nghĩa D đã train 999 updates. Pilot chính ở cell sau sweep; đọc `run_summary.json`: `additional_updates` và `additional_transitions`.

**`Expert gate: FAIL` ở baseline:** notebook cũ dùng chung nhãn evaluator. Code mới ghi `Baseline gate` và metric/failure predicates. FAIL của PPO999 không phải FAIL của teacher/expert dataset. Giữ JSON, đọc `gate.failures`, `bins`, `failure_breakdown` và checkpoint SHA. Nếu evaluation hoàn tất với metrics hợp lệ, tiếp tục hai arm từ cùng COMMON; không hạ gate hoặc thay teacher vào riêng một arm.

Gate giữ success ≥95%, linear RMSE ≤0.25, yaw RMSE ≤0.20, upright ≥0.97. Dùng actual velocity, success, RMSE, upright và falls để kết luận chất lượng policy; reward hoặc D accuracy cao chưa chứng minh locomotion tốt.

### Chạy lại cell pilot khi output đã tồn tại

Cell pilot mới dùng `--reuse-completed --restart-incomplete`:

- Có `run_summary.json` hoàn tất và checkpoint: kiểm tra arm, env/minibatches, seed, COMMON hash, budget, resume và runtime; khớp thì bỏ qua training arm đó.
- Chưa có summary, output cũ có file: chạy lại đủ budget từ checkpoint yêu cầu trong `attempt_001`, `attempt_002`, …; giữ checkpoint/log cũ. Đây là restart, không tự resume checkpoint chạy dở.
- Summary sai config hoặc thiếu checkpoint: dừng và yêu cầu output mới. Summary mới ghi thêm source/dataset/final-checkpoint hashes; summary cũ được báo thiếu các provenance hashes này, không sửa nội dung cũ.

Endpoint và export vẫn đọc `run_summary.json` tại thư mục pilot, dùng `run_dir` trỏ tới attempt hoàn tất. Chạy các arm tuần tự, chờ subprocess hiện tại kết thúc trước khi chạy lại cell.

Trong runtime đang có `SOURCE`, `COMMON`, `SELECTED` và kết quả sweep, cập nhật source rồi chạy lại **cell pilot đã sửa**; giữ SESSION và budget:

```python
subprocess.run(
  ["git", "pull", "--ff-only", "origin", "feature/kaggle-airl-16k"],
  cwd=SOURCE,
  check=True,
)
updated_manifest = json.loads(
  (SOURCE / "scripts/cloud/kaggle_source_manifest.json").read_text()
)
for relative, expected in updated_manifest["source_files"].items():
  assert digest(SOURCE / relative) == expected, f"Source hash mismatch: {relative}"
recovery_commit = subprocess.check_output(
  ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
).strip()
(OUTPUT / f"{SESSION}_recovery_source.json").write_text(
  json.dumps({"commit": recovery_commit, "purpose": "pilot recovery"}, indent=2)
)
print("Updated source:", recovery_commit)
```

Không cần chạy lại smoke hoặc sweep cho lỗi output trùng này. Notebook cũ phải thêm hai flags trên vào lệnh `cloud("train", ...)`; `git pull` chỉ cập nhật tool trên đĩa, không sửa cell đã import trong Kaggle.

## 3. Env, tăng tốc và budget

Sweep: `16384 / 20480 / 24576 / 32768` env, `4 / 8` minibatches; giữ **24 steps/env**, **5 PPO epochs**, **1 D update với batch tối đa 1024 learner + 1024 expert/update**. Chọn candidate >16000 env, ≥15% VRAM trống; trong 3% throughput nhanh nhất ưu tiên peak VRAM thấp hơn. Đây là tối ưu trong các cấu hình đã đo.

Tăng tốc đã triển khai: reuse log-prob PPO khi rollout, tính g một lần cho reward/diagnostic, gom scalar logs trên GPU và tái sử dụng transition buffer. D minibatches vẫn tính current learner density; không đổi solver, validation hoặc tự scale LR. Đổi 4→8 minibatches tăng PPO optimizer steps 20→40/update, nên hai arm phải giữ cùng selection.

Kết quả T4 bạn gửi chọn **16384 env, 8 minibatches**:

```text
rollout               = 16384 × 24 = 393216 transitions/update
PPO minibatch         = 393216 / 8 = 49152 transitions
median full update    = 11.933695996 s
throughput            = 32950.060 transitions/s
peak device memory    = 5065 / 14911.6875 MiB (~34%)
pilot updates/arm     = 23592960 / 393216 = 60
AIRL ước tính         = 60 × 11.933695996 ≈ 716 s (~12 phút)
```

Ước tính chưa gồm startup, checkpoint và evaluation; PPO cần đo riêng. Hai JSON giống nhau bạn gửi là cùng một kết quả in lặp. Chưa có endpoint paired pilot để kết luận AIRL học tốt hơn.

Ở selection này, **1500 updates bổ sung/arm = 589824000 transitions/arm**; selection khác phải tính lại. Notebook hiện chặn chunk dự kiến >2 giờ: budget dài cần chia chunk cùng lịch cho hai arm, chưa có launcher tự động train dài chỉ bằng đổi một biến.

Lưu checkpoint mỗi 50 updates và final. Chunk sau dùng checkpoint **riêng từng arm**, `--resume`, giữ selection/seed/config. AIRL resume khôi phục g/h, D optimizer, PPO và RNG; simulator reset, không khôi phục chính xác simulator state. AIRL checkpoint không dùng cho PPO control.

## 4. Sau train: export discriminator AIRL

Checkpoint chứa `checkpoint["infos"]["airl_state_dict"]`. Công thức online của repo:

```text
f = g(s,c) + gamma × (1 - terminated) × h(s_next,c_next) - h(s,c)
logit_D = f - log_pi(a | actor_obs)
r_total = r_env + lambda_D × f
```

Để giữ reward này khi frozen, export **cả g, h, gamma và normalization buffers**. Artifact không chứa actor/critic hoặc optimizer. Load gọi `eval()`, `requires_grad_(False)`, tính reward no-grad; không tính lại normalization. Giai đoạn frozen chỉ cập nhật PPO actor/critic, **D updates = 0**.

Thêm cell sau train/evaluate trong notebook:

```python
summary = json.loads((OUTPUT / f"{SESSION}_pilot_airl/run_summary.json").read_text())
checkpoint = max(
  Path(summary["run_dir"]).glob("model_*.pt"),
  key=lambda p: int(p.stem.split("_")[-1]),
)
frozen_file = OUTPUT / f"{SESSION}_frozen_airl_reward.pt"
subprocess.run(
  [
    "uv",
    "run",
    "--no-sync",
    "python",
    "-m",
    "mjlab.scripts.airl_export_reward",
    "--checkpoint-file",
    str(checkpoint),
    "--output-file",
    str(frozen_file),
  ],
  cwd=SOURCE,
  check=True,
)
print("Source checkpoint:", checkpoint)
print("Frozen reward:", frozen_file, "SHA256:", digest(frozen_file))
```

Cell chọn final checkpoint theo số iteration của run AIRL; không lấy smoke hoặc PPO control. Final không mặc định là D tốt nhất: giữ endpoint report và đánh giá reward alignment trước khi chọn donor để chạy frozen dài.

CLI từ thư mục source đã clone:

```bash
uv run --no-sync python -m mjlab.scripts.airl_export_reward \
  --checkpoint-file /duong/dan/airl/model_1058.pt \
  --output-file /kaggle/working/outputs/pilot01_frozen_airl_reward.pt
```

`model_1058.pt` chỉ là ví dụ; dùng path thực tế từ summary. Export không ghi đè, không sửa checkpoint nguồn và từ chối PPO/GAIL. Chỉ export checkpoint nguồn tin cậy do bạn tạo. Metadata ghi schema, contract, architecture, weight, SHA checkpoint/dataset. Artifact standalone load bằng `weights_only=True`; không cần expert dataset hoặc teacher để tính reward.

## 5. Nạp frozen reward và cộng vào reward có sẵn

Mỗi g/h nhận **68-D body-local state + 3-D command**; không phải actor observation 99-D hoặc GAIL state-action 100-D. `f` có thể âm. Không thêm sigmoid/softplus hoặc dùng `logit_D` làm frozen reward.

Hook mẫu cho PPO rollout, với `env` là wrapper của runner:

```python
import torch
from mjlab.rl.frozen_airl import FrozenAirlReward
from mjlab.rl.gail import velocity_gail_state

frozen = FrozenAirlReward.load(
  "/kaggle/working/outputs/pilot01_frozen_airl_reward.pt", device="cuda:0"
)
weight = frozen.default_weight  # 0.01 hiện tại.
base_env = env.unwrapped


def capture(base_env):
  return {
    "observations": velocity_gail_state(base_env.scene["robot"]),
    "commands": base_env.command_manager.get_command("twist"),
  }


base_env.set_transition_capture(capture)


@torch.no_grad()
def step_with_frozen_reward(actions):
  current = {k: v.detach().clone() for k, v in capture(base_env).items()}
  next_obs, r_env, dones, extras = env.step(actions)
  successor = extras["transition"]  # Trước auto-reset.
  r_frozen = frozen.reward(
    current["observations"].to(frozen.device),
    successor["observations"].to(frozen.device),
    current["commands"].to(frozen.device),
    successor["commands"].to(frozen.device),
    successor["terminated"].to(frozen.device),
  )
  r_total = r_env.to(frozen.device) + weight * r_frozen
  return next_obs, r_total, dones, extras
```

Trong PPO rollout, thay `env.step(actions)` bằng `step_with_frozen_reward(actions)` ở ngoài helper, đưa `r_total` vào `alg.process_env_step`. Log riêng `r_env`, `r_frozen`, `weight*r_frozen`. Tắt đường GAIL/AIRL online để không cộng hai lần hoặc cập nhật D. Frozen f không cần actions/log-prob; actions ở helper chỉ dùng để step simulator.

**Đây là hook tích hợp, chưa phải lệnh train frozen PPO hoàn chỉnh.** Không dùng `airl.enabled=True` hoặc resume của runner online để gọi frozen: runner đó vẫn cập nhật D. So sánh PPO control với PPO + frozen reward cùng initialization, seed/config/budget; kiểm tra mọi parameter/buffer D không đổi.

`terminated=True` tắt successor potential; timeout/truncated vẫn bootstrap. Dùng successor/next command trước reset, không dùng actor observation sau reset hoặc `dones` thay `terminated`. Giữ gamma đã export khớp gamma shaping dự định dùng. Task reward đã có dt scaling; không nhân f thêm dt để giả định tương đương online. Weight `0.01` không có nghĩa 1% reward; kiểm tra phân phối reward và task metrics trước khi chọn weight.

Export dùng **f** để giữ công thức online của repo. Dùng riêng **g** cho transfer là phương án khác, cần triển khai/ghi mode và đánh giá riêng. Tham khảo [AIRL API](https://imitation.readthedocs.io/en/stable/_api/imitation.algorithms.adversarial.airl.html) (reward_train/reward_test) và [bài báo AIRL](https://arxiv.org/abs/1710.11248).

## 6. Lưu output và kiểm tra

Lưu `/kaggle/working/outputs` qua output notebook hoặc tải về/tạo Dataset để attach session sau. Giữ checkpoint **đầy đủ** cho online resume; frozen artifact không thay thế checkpoint resume. Nếu export sau cell output hashes, chạy lại cell đó để cập nhật `output_hashes.json`.

Giữ source commit, hashes, `selected.json`, `run_summary.json`, YAML config, TensorBoard, baseline/endpoint reports và frozen artifact. Runtime/GPU/packages thay đổi thì sweep lại. Đổi SESSION/output cho chunk mới; `/tmp` không được coi là artifact đã lưu.

Kiểm tra code từ checkout đúng source:

```bash
uv run --no-sync pytest tests/test_frozen_airl.py tests/test_airl.py tests/test_airl_runner.py tests/test_kaggle_airl.py -q
uv run --no-sync ruff check
uv run --no-sync ty check
uv run --no-sync pyright
```

Tests xác minh export/load giữ đúng f, normalization bitwise, terminal/changing command, không gradients/thay đổi D, checkpoint nguồn không đổi, CLI và rejection checks. Chúng chưa chứng minh D frozen giúp PPO đi tốt hơn; cần chạy frozen PPO và task evaluation sau tích hợp hook.

Tài liệu framework gốc: [MuJoCo Lab](https://github.com/mujocolab/mjlab).
