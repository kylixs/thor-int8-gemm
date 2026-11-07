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
