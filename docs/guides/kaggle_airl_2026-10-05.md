# Kaggle: chọn số env và tăng throughput PPO + AIRL

## Cấu hình và phép tính

Mức lịch sử là **16.384 env × 24 steps = 393.216 transitions/update**.
Không suy ra số env tối ưu chỉ từ VRAM: simulator, PPO activations và GPU model
quyết định peak/throughput. Quét cùng task `Mjlab-Velocity-Flat-Unitree-G1-AIRL-MVP`
với checkpoint PPO999 và dataset deterministic đã qualify cho `(1.2,0,0)`.

| Env | Transitions/update | PPO batch với4 minibatches | Với8 | PPO + AIRL persistent storage tối thiểu |
|---:|---:|---:|---:|---:|
| 16.384 | 393.216 | 98.304 | 49.152 | 858,75 MiB |
| 20.480 | 491.520 | 122.880 | 61.440 | 1.073,44 MiB |
| 24.576 | 589.824 | 147.456 | 73.728 | 1.288,13 MiB |
| 32.768 | 786.432 | 196.608 | 98.304 | 1.717,50 MiB |

Storage math: actor99 + critic111, action29, mean/std58,5 PPO scalars,
doneuint8 =1209 bytes/transition. AIRL state68/current actor99/cmd3/action29/
nextstate68/nextcmd3 float32 +terminatedbool =1081 bytes/transition.
Bảng chỉ là storage, **không gồm** simulator, autograd/activations, indexing,
temporary transition tensors, CUDA graphs, allocator reserve hoặc expert CPU data.

Quét 4 và8 PPO minibatches, giữ5 epochs. Dùng8 giảm activation peak nhưng tăng
optimizer steps từ20 lên40/update; đây là thay đổi cấu hình học được ghi rõ,
không coi là tối ưu tính toán giữ hoàn toàn learning dynamics. Hai arms cùng
env/minibatches và ngân sách. Không tự scale LR, thay std/entropy, giảm solver,
contact buffers, validation hoặc bật AMP/compile.

## Những thay đổi đã triển khai

- Rollout AIRL dùng log-prob PPO vừa tính khi lấy raw Gaussian actions.
  D minibatches vẫn tính lại **current learner density** cho expert và policy.
- Tính g(s) một lần để dùng chung cho f và diagnostic.
- Gom4 scalar diagnostics trên GPU, chuyển sang CPU một lần/update.
- AIRL giữ một normal preallocated transition buffer, ghi từng slot bằng `copy_`;
  không giữ list + concatenate + clone qua các iterations. Vẫn bảo vệ mutable
  pre-step state/command/input và dùng successor trước reset.
- Giữ Warp CUDA graphs/TF32 mặc định khi backend hỗ trợ; không tuyên bố bật mới.
- Single GPU; không dùng AIRL DDP vì chưa có discriminator synchronization.

Công thức không đổi:
`f=g(s,c)+gamma*(1-terminated)*h(s_next,c_next)-h(s,c)`;
`logit_D=f-log_pi_current`; `r_total=r_env+0.01*f.detach()`.
Task reward có dt scaling; weight0.01 không có nghĩa1% reward.

## Artifact và cách chạy

Tool: [kaggle_airl.py](../../scripts/cloud/kaggle_airl.py).
Notebook: [kaggle_airl.ipynb](../../scripts/cloud/kaggle_airl.ipynb).

Source được clone trực tiếp từ [nhánh feature/kaggle-airl-16k](https://github.com/nvhiep249/mjlab/tree/feature/kaggle-airl-16k).
Không cần upload source ZIP. Kaggle Dataset đã có: `nvhiep2409/airl-ppo`.

1. Import `scripts/cloud/kaggle_airl.ipynb` từ nhánh này vào Kaggle.
2. Attach Dataset `nvhiep2409/airl-ppo`, bật Internet và chọn GPU tương thích cu128.
3. Cell cấu hình đã đặt `INPUT_ROOT=/kaggle/input/datasets/nvhiep2409/airl-ppo`.
   Nó tìm đệ quy `airl_expert_1p2_deterministic_v1.pt` và `ppo_common_999.pt`
   (hoặc tên gốc `model_999.pt`), từ chối nếu tên bị trùng hoặc hash không đúng.
4. Chạy các cell theo thứ tự. `REPO_BRANCH=feature/kaggle-airl-16k` lấy code ở đầu nhánh.
   Để replay chính xác, đặt `REPO_COMMIT` bằng full SHA đã ghi trong output.
   Đổi `SESSION` khi chạy lượt mới để giữ artifact cũ.

Expert 382.726.465 bytes, SHA256
`3cf8b910c4957fc80d257762d55945e5777411ba2785659dfb29dd08d69e72e8`.
Common PPO 5.316.259 bytes, SHA256
`b318a6da5db37a6795e4a943d0d52eaa75743bd403341ecf47b10b3f20972dc8`.
Expert đã qualification; common PPO được xác minh hash/provenance, không mặc định pass task gate.
Dataset khác phải được audit trước khi đổi hashes.
Expert metadata giữ teacher provenance; teacher không phải learner initializer.

Baseline FAIL là kết quả đánh giá common PPO999, không phải đánh giá teacher/expert dataset.
Evaluator hiện ghi rõ vai trò `baseline`/`endpoint`/`expert`, các metric và predicates fail.
Giữ report baseline; nếu metrics hợp lệ, tiếp tục sweep và paired pilot từ cùng COMMON.
Không hạ gate hoặc thay teacher vào một arm để làm baseline pass. Success95% nghĩa là
ít nhất95/100 episodes cùng đạt tracking, upright và survival; mean RMSE tốt chưa đủ.
Muốn chẩn đoán một FAIL cụ thể, đọc `gate.failures`, `bins`, `failure_breakdown`,
checkpoint SHA và config của report đó; không suy từ seed/GPU/protocol khác.

Cell order: input/hash → clone đúng branch/commit + source SHA256 → locked Python3.11/cu128
bootstrap → PPO/AIRL1024env smoke5updates + AIRL resume1update → evaluate common base
→ sweep → paired pilot → endpoint evaluation → output hashes.
Source/venv ở `/tmp/mjlab-github-<session>`, cache ở `/tmp/uv-cache`, outputs ở
`/kaggle/working/outputs`. `<session>_github_source.json` ghi repo, branch, commit,
source-file hashes và data hashes; notebook kiểm tra `kaggle_source_manifest.json`.
Clone tắt `core.autocrlf` để bytes khớp manifest trên Linux và Windows.
Linux dependencies và GPU driver phải qua preflight thật.
Không dùng P100 với lock hiện tại; không tự fallback CPU/downgrade.

Lệnh `pack` trong tool vẫn hỗ trợ snapshot ZIP cho chạy offline trước đây;
notebook hiện tại dùng GitHub. Dataset có source ZIP cũ sẽ được bỏ qua.

Benchmark mỗi candidate trong process riêng:3 warmup +8 timed updates,
không checkpoint/ONNX trong sweep. Median wall time bao gồm rollout + D + PPO,
CUDA synchronize tại update boundary; logger console chính không nằm trong timer.
GPU memory lấy mẫu `nvidia-smi` mỗi0,25s và bổ sung Torch peak reserved.
Peak lấy mẫu có thể bỏ lỡ spike; yêu cầu tối thiểu15% headroom và tiếp tục theo dõi
run dài. OOM/error/timeout có log riêng; không hạ validation để candidate pass.

`selected.json` chọn các cấu hình >16.000 env pass và có headroom. Trong3%
throughput tốt nhất, ưu tiên ít peak VRAM hơn. Đây là tối ưu **trong các candidates
đã đo**, chưa là global optimum hoặc chứng minh đạt task gate nhanh hơn.
Nếu không candidate đạt thì dừng, không tạo kết quả lựa chọn giả.

```bash
uv run --no-sync python scripts/cloud/kaggle_airl.py sweep \
  --input <thu-muc-chua-expert> --checkpoint <checkpoint-da-kiem-tra> --output /kaggle/working/outputs/pilot01_sweep
uv run --no-sync python scripts/cloud/kaggle_airl.py train \
  --input <thu-muc-chua-expert> --checkpoint <checkpoint-da-kiem-tra> \
  --selected /kaggle/working/outputs/pilot01_sweep/selected.json \
  --output /kaggle/working/outputs/pilot01_airl --arm airl --transitions 23592960
```

Control đổi `--arm ppo` và output; bắt đầu cùng `ppo_common_999.pt`.
23.592.960 transitions/arm =60/48/40/30 additional updates tương ứng4 mức env.
Không gọi là 200-update pilot1024 cũ; đây là pilot mới theo yêu cầu scale.
Giữ1 D update/batch1024 để cô lập scaling; theo dõi D/density/weighted reward.
Không tăng D updates tùy tiện theo env count. Tăng env khiến policy đổi sau nhiều
samples hơn; cần task evaluation để kiểm tra chất lượng học.

Notebook chặn chunk dự kiến >2h. Lưu mỗi50 updates và final; nếu checkpoint gap
dài hơn15phút thì cần rút save interval giống nhau trước khi chạy cả hai arms.
Chunk sau dùng `--checkpoint <own-arm-checkpoint> --resume` và giữ selection/config.
AIRL restore đầy đủ g/h, D optimizer, PPO/normalizers và next iteration. Launcher
sửa control next iteration từ saved last index, chặn checkpoint AIRL cho control.
Train kiểm tra GPU/totalVRAM/CUDA/packages khớp selection trước khi tạo env;
runtime mới phải đo lại, không dùng selection từ GPU khác.
Simulator reset khi resume; không có exact simulator-state restore. Không restart
fresh khi missing checkpoint. Đổi SESSION/output để không ghi đè.

## Kiểm chứng và giới hạn

Nhánh GitHub đã qua 178 focused tests, Ruff format/check, Ty, Pyright và
`uv lock --check`. Windows không có make; đã chạy đúng các lệnh tương đương
`make check`. Test thực thi clone branch/commit, từ chối source/data bị sửa và
checkout tồn tại. Manifest 272 files khớp canonical Git index; absolute imports
đã được rà soát đầy đủ. CUDA RTX3050 smoke từ chính nhánh: PPO và AIRL mỗi arm
64env ×24steps ×1update pass; đây chỉ là kiểm tra tích hợp, không so throughput.

Focused CPU tests:42 passed gồm formula, density, terminal/pre-reset, regular
storage reuse + real D backward, PPO update/resume và selection/budget math.
CUDA local RTX3050:256env,2updates,12.288 transitions; actor/critic và D cập nhật
được. Đây là integration check, không là benchmark Kaggle hoặc locomotion verdict.
Runtime evidence: `.tmp/kaggle_large_env_cuda_smoke/result.json` và console log.
So sánh runner trước/sau trên RTX3050,1024env/24steps/4minibatches, cùng base/seed,
5updates trong process riêng, bỏ2warmup và đo3updates: median4.6343→4.3585s/update,
5.303→5.639 transitions/s (+6.33%); Torch peak reserved278→254MiB.
Evidence `.tmp/kaggle_airl_perf_before/result.json` và `kaggle_airl_perf_after/result.json`.
Đây là phép đo ngắn, chạy tuần tự, có clock/noise; chưa chứng minh lợi ích trên
T4 hoặc16k+env, chưa đo time-to-task-gate hoặc learning-quality equivalence.
Nhánh GitHub chứa source AIRL và dependency GAIL cần thiết; không mang theo
nhánh thử nghiệm vision. Kiểm tra clone thực tế, format/lint và hai type checkers
được chạy trên checkout riêng trước khi push.

Chưa bootstrap Linux, sweep16k+ hoặc train/evaluate trên Kaggle.
Chỉ kết luận tốc độ học bằng actual velocity, success, linear/yaw RMSE, upright,
falls và per-seed evaluations. Gate giữ95%/0.25/0.20/0.97; evaluate base và endpoint
100episodes trên42005/42006. Nếu base đã pass, time-to-gate=0; báo delta/stability.
Continuation không chứng minh fresh training nhanh hơn.

Nguồn backend: [MuJoCo Warp performance](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html),
[AIRL density API](https://imitation.readthedocs.io/en/latest/algorithms/airl.html),
[PyTorch cu128 architecture support](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).
