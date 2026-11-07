# thor-int8-gemm

**[English](#english) | [中文](#中文)**

---

<a name="english"></a>
## English

Performance tuning & forensics for comfy-kitchen INT8 GEMM on Jetson Thor (sm_110, unified-memory Blackwell).
Background: comfy-kitchen's int8 kernel sustains only ~50 TOPS on Thor — 2.4× *slower* than plain bf16 cuBLAS (120 TFLOPS). This repo carries the full evidence chain, a Triton fused-dequant port (from the GB10 solution), tile sweeps, and the cuBLASLt route that finally shipped to production. Upstream: [comfy-kitchen#312](https://github.com/Comfy-Org/comfy-kitchen/issues/312), prior art [#280](https://github.com/Comfy-Org/comfy-kitchen/issues/280) (GB10/sm_121, same disease).

### Approaches (weights untouched in all three)
1. **Triton fused-dequant port** (`src/ck_int8_patch_thor.py`) — ports the GB10-validated Triton kernel (int32 accumulate + fused dequant epilogue, `.pth` import-proxy swapping `_C.cutlass_int8_dequant`) to sm_110; 168-config tile sweep.
2. **Direct cuBLASLt** (`src/ck_int8_patch_intmm.py`) — raw `torch._int_mm`.
3. **kitchen's built-in cublas fallback** — container env `COMFY_KITCHEN_DISABLE_CUTLASS=1`, zero code (shipped to production).

### Results (real MiniMax H3 DiT layer shapes, M=8k–112k)

| Path | TOPS | Notes |
|---|---|---|
| kitchen stock (cutlass / PTX-JIT, no sm_110 cubin) | 50–55 | baseline |
| Triton port (GB10 config verbatim) | 73.3 | ×1.32 |
| **Triton local sweep best `128×128×128 w4s3g8`** | **98.4** | **×1.78**; BK=128 + 4 warps is the key |
| cuBLASLt sustained / isolated replay | **172 / 310** | `cutlass3x_sm100 i256x256x32` (wide tcgen05 path) |

Triton ceiling: SASS contains only `IMMA.16832.S8` (narrow legacy MMA) and tensor-pipe activity is hard-capped at 72.5% (ncu, tile-independent) ⇒ theoretical max ~120 TOPS — cannot reach cuBLASLt.

### Engine-level impact (MiniMax H3 ref2va, 720p/5s, 6 steps)

| | stock | Triton patch | **cublas route (production)** |
|---|---|---|---|
| DiT step | 41.0s | ~34s (est.) | **22.9s (−44%)** |
| End-to-end | 326s | ~245s (est.) | **173s (−47%)** |

Correctness: Triton and `torch._int_mm` are both `torch.equal` to stock (kernel level); the cublas route differs 10–13% at video level (epilogue float ordering → chaotic divergence; VLM judged quality-equivalent).

### Layout
`src/` both patches + upstream original · `bench/` sweep & bench scripts, `sweep_results.jsonl` (all 168 configs) · `tests/` bit-exactness · `docs/` write-ups.
Environment: Thor devkit (96 SM @1386MHz, 120W mode), torch 2.15+cu132, comfy-kitchen 0.2.40.

## Background / Why it matters / Use cases / Roadmap

### Background
Jetson Thor (sm_110) and DGX Spark (GB10, sm_121) are unified-memory Blackwell parts. comfy-kitchen — the kernel library behind ComfyUI's quantized checkpoints (int8_convrot etc. for MiniMax H3, Qwen-Image …) — ships **no sm_110/sm_110a cubins**, so on these devices its INT8 GEMM runs via PTX JIT on a generic path and sustains only ~50 TOPS. Absurdly, that is **2.4× slower than plain bf16 cuBLAS**, i.e. the entire benefit of INT8 quantization is inverted on this hardware class. The same disease on GB10 was documented in comfy-kitchen#280.

### Why this project matters
- **End-user impact**: MiniMax H3 (and other int8_convrot models) render ~47% faster once the right GEMM path is chosen — measured end-to-end on a production pipeline (326s → 173s for 720p/5s).
- **Evidence chain**: V0 (missing cubins, cuobjdump) → V1 (flat 50 TOPS ceiling, 43W "full" load) → ncu (54%/72.5% tensor-pipe caps) → SASS (only `IMMA.16832` emitted) — a reusable playbook for diagnosing "quantized slower than fp16" on any Blackwell unified-memory device.
- **Upstream leverage**: everything here (bench harness, tile table, ncu data) is directly usable by comfy-kitchen maintainers; issue #312 offers a PR.

### Potential applications
- Any ComfyUI user on Jetson Thor / AGX Thor devkits running int8/w4a8 checkpoints (video DiT, image DiT).
- A reference for porting GB10-style Triton fused-dequant kernels to other under-served archs (sm_110 siblings).
- The `.pth` proxy pattern (version-guarded, auto-fallback to stock) is a safe A/B mechanism for any kernel experiment in production.
- cuBLASLt fallback via `COMFY_KITCHEN_DISABLE_CUTLASS=1` is a **zero-code workaround** usable today.

### Roadmap / open directions
- [x] Round-2 sweep: no gain (98.6 confirmed ceiling for the plain-pointer kernel)
- [x] tcgen05 ON: stable Triton 3.9.0 already emits tcgen05 on sm_110 — **trigger = TMA tensor descriptors + `warp_specialize=True`** (TTGIR `tc_gen5_mma`, SASS `UTCHMMA`); plain-pointer kernels fall back to `IMMA.16832`
- [ ] Rewrite the int8 fused-dequant kernel in persistent TMA+WS form (tutorial-09 pattern, TMEM multi-buffer + autotune): naive TMA-WS prototype only reaches 31 TFLOPS (bf16) vs cuBLAS 120 — the gate is open but needs proper kernel structure; upside ~cuBLASLt-class 172+ for the upstream-fixable Triton path
- [ ] TMA (`tl.make_tensor_descriptor`) arm — async copies free warps for MMA
- [ ] Warp specialization on Blackwell Triton
- [ ] Upstream: PR tile table `128×128×128 w4s3g8` for sm_110 (pending maintainer reply on #312)
- [ ] cuBLASLt sustained-vs-isolated gap (172 vs 310 TOPS): clock-governor investigation; note tcgen05 async cores may not be counted by `sm__pipe_tensor`
- [ ] fp8_scaled weights arm (tcgen05-wide path, theoretical ~2× over int8)


---

<a name="中文"></a>
## 中文

Jetson Thor（sm_110，统一内存 Blackwell）上 comfy-kitchen INT8 GEMM 的性能调优与取证。
背景：comfy-kitchen 的 int8 kernel 在 Thor 上只有 ~50 TOPS，比朴素 bf16 cuBLAS（120 TFLOPS）还慢 2.4×——本仓给出完整证据链、Triton fused-dequant 移植（GB10 方案）、tile 扫描，以及最终生产采用的 cuBLASLt 路线对照。上游 issue：[comfy-kitchen#312](https://github.com/Comfy-Org/comfy-kitchen/issues/312)、参照 [#280](https://github.com/Comfy-Org/comfy-kitchen/issues/280)（GB10/sm_121 同病）。

## 方案（三条，权重均不动）

1. **Triton fused-dequant 移植**（`src/ck_int8_patch_thor.py`）——把 GB10 验证过的 Triton kernel（int32 累加＋反量化融合 epilogue，`.pth` 代理替换 `_C.cutlass_int8_dequant`）移植到 sm_110，本机扫 168 组 tile。
2. **cuBLASLt 直调**（`src/ck_int8_patch_intmm.py`）——`torch._int_mm` 裸 GEMM。
3. **kitchen 自带 cublas 后备路线**——容器 env `COMFY_KITCHEN_DISABLE_CUTLASS=1`，零代码（最终生产采用）。

## 结果（真实 MiniMax H3 DiT 层形状，M=8k–112k）

| 路径 | TOPS | 说明 |
|---|---|---|
| kitchen stock（cutlass/PTX JIT，无 sm_110 cubin） | 50–55 | 现状基线 |
| Triton 移植（GB10 配置照搬） | 73.3 | ×1.32 |
| **Triton 本机扫描最优 `128×128×128 w4s3g8`** | **98.4** | **×1.78**；BK=128＋w4 是关键 |
| cuBLASLt 持续 / 隔离重放 | **172 / 310** | `cutlass3x_sm100 i256×256×32`（tcgen05 宽指令） |

Triton 天花板判定：SASS 只有 `IMMA.16832.S8`（窄指令），张量核活跃度 72.5% 硬顶（ncu，tile 无关）⇒ 理论极限 ~120 TOPS，追不上 cuBLASLt。

## 引擎级效果（MiniMax H3 ref2va，720p/5s，6 步）

| | stock | Triton patch | **cublas 路线（生产）** |
|---|---|---|---|
| DiT 步时 | 41.0s | ~34s（推算） | **22.9s（−44%）** |
| 整笔 | 326s | ~245s（推算） | **173s（−47%）** |

正确性：Triton 与 `torch._int_mm` 均与 stock `torch.equal` 逐位一致（kernel 级）；cublas 路线整笔视频级差异 10–13%（epilogue 浮点序→混沌分歧，VLM 判定质量等价）。

## 目录

- `src/` patch 两版（Thor 适配＋intmm）＋上游原版对照
- `bench/` 扫描与 bench 脚本、`sweep_results.jsonl`（168 配置全量）
- `tests/` 逐位校验
- `docs/` 方案、扫描记录（中文）、上游 writeup

环境：Thor devkit（96SM@1386MHz、120W 档）、torch 2.15+cu132、comfy-kitchen 0.2.40、triton 随 torch。

### 背景 / 意义 / 应用 / 后续（中文摘要）
- **背景**：Thor（sm_110）与 DGX Spark（GB10/sm_121）同属统一内存 Blackwell；comfy-kitchen 不带 sm_110 原生 cubin，INT8 GEMM 走 PTX JIT 只有 ~50 TOPS——**比朴素 bf16 还慢 2.4×**，量化收益整体倒挂（GB10 同病见 #280）。
- **意义**：选对 GEMM 路径后 MiniMax H3 生产渲染快 47%（326→173s/720p·5s）；V0→V1→ncu→SASS 的证据链是排查"量化反而更慢"的通用方法论；全套数据可直接支撑上游 PR（#312）。
- **应用场景**：Thor/AGX Thor 上跑 int8/w4a8 检查点的 ComfyUI 用户；向其他冷门架构移植 GB10 式 Triton kernel 的参照；`.pth` 代理（版本守卫＋自动回落）是生产环境 kernel 实验的安全范式；`COMFY_KITCHEN_DISABLE_CUTLASS=1` 是今天就能用的零代码规避。
- **后续方向**：第二轮扫描（BK256/num_ctas/stages，跑中）；TMA 与 warp-specialization 臂；上游 tile 表 PR；cuBLASLt 持续 172 vs 隔离 310 的频率调查；fp8 权重臂（理论再 ×2）。

