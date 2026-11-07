# Triton tcgen05（sm_110）INT8 GEMM 方案与执行计划（2026-11-07 立项）

> **EN summary**: Plan to rebuild the int8 fused-dequant kernel on the tcgen05 path
> (TMA descriptors + warp specialization, persistent grid). Feasibility proven:
> stable Triton 3.9.0 emits `UTCHMMA` on sm_110 with this pattern; bf16 persistent
> prototype already reaches 108 TFLOPS (vs cuBLAS 120, naive form 31). Target:
> int8 ≥ cuBLASLt's 172 TOPS → upstreamable fix for comfy-kitchen#312.

## 1. 已验证事实（方案依据）

| 事实 | 数据 |
|---|---|
| tcgen05 触发条件 | TMA tensor descriptor ＋ `warp_specialize=True`（TTGIR `tc_gen5_mma`、SASS `UTCHMMA`）；朴素指针→IMMA.16832 |
| naive TMA-WS（非 persistent） | 31 TFLOPS（bf16）——结构不行 |
| persistent（tutorial-09 模式） | bf16 最优 **108 TFLOPS**（`128×256×64, s2, w8`）；cuBLAS bf16＝120 |
| 关键规律 | stages 越少越快（s2>s3>s4）＝**SMEM 227KB 是约束**；BK=128（bf16）直接爆 SMEM |
| 参照系 | cuBLASLt int8＝172 TOPS 持续（`i256x256x32` kernel） |

## 2. 核心假设（本方案要验证的）

**int8 数据宽度是 bf16 的一半 ⇒ 同 SMEM 预算下 tile 可以加倍**
（如 `128×256×128` 或 `256×256×64`），配合 tcgen05 宽指令，
int8 吞吐应显著超过 bf16 的 108（理论 2×→~200，目标 ≥172）。

## 3. 阶段与判定门

| 阶段 | 内容 | 判定门 | 预计 |
|---|---|---|---|
| T0 基线锁定 | bf16 persistent 108 配置复测＋TTGIR 确认 tc_gen5_mma（防回归） | 复现 ±3% | 0.5h |
| T1 int8 骨架 | persistent kernel 换 int8 输入＋int32 acc＋反量化 epilogue（xs·ws＋bias，fp32 序与 stock 一致）；SMEM 允许的 tile 网格扫（128×256×{64,128}、256×256×{32,64}、s∈{2,3}、w∈{4,8}） | **≥120 TOPS** 才继续 | 2h |
| T2 逐位校验 | 对 stock `cutlass_int8_dequant`：`torch.equal`（随机＋真实形状，含 per-row xs/per-col ws/bias） | 全等（GB10 已证此 epilogue 可逐位一致） | 1h |
| T3 精调 | 胜出配置邻域细扫＋GROUP/epilogue 子块（EPILOGUE_SUBTILE）；ncu 复核张量核活跃度 | 收敛即停 | 1h |
| T4 引擎臂 | `.pth` 代理挂主机1 comfyui-h3（CK_INT8_MODE=tcgen05），单笔 720p/5s dit_step | vs cuBLASLt 22.9s（持平或更好） | 1h（需空窗） |
| T5 上游回馈 | 结果＋kernel 提到 #312（"正统修法"：sm_110 tcgen05 Triton 回退） | — | 0.5h |

## 4. 风险与止损

- **风险 A**：tcgen05 的 int8 变体（`UTCIMMA`）在 Triton 3.9.0 未实现/未启用
  → T1 第一个 tile 就会看到（SASS 仍 IMMA）⇒ **止损：记结论、关项**，
  等 Triton 后续版本，不在 kernel 上硬调；
- 风险 B：epilogue 逐位不一致（浮点序差异）⇒ 降级为"数值等价"（相对误差 <1e-3）
  ＋整笔 VLM 质检，按 w4a8 先例处理；
- 风险 C：T1 过但只有 130–160（不到 172）⇒ 仍值得：比朴素 Triton ×1.3–1.6，
  作为上游修复的可用实现；生产继续用 cuBLASLt；
- 主机1 生产不动（全部实验在主机2；T4 才碰主机1 且 opt-in env）。

## 5. 产出物

- `src/ck_int8_patch_tcgen05.py`（persistent TMA-WS int8 fused-dequant）
- `bench/tcgen05_sweep.jsonl`＋扫描记录新轮次
- GitHub README 更新；#312 追评（T5）
