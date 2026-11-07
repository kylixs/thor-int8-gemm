"""A2 融合 kernel bench（主机2）——簇1 rms+mod+res+quant、簇2 SwiGLU+quant。
基线＝生产同款 eager 链（comfy_kitchen CUDA fp8 算子：stochastic_round(生产)
与 quantize_per_tensor_fp8(确定性参照)）；融合＝ck_fused_mod_norm_quant。
输出 /tmp/a2_bench_result.json
"""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nvidia/a2-fused")
from ck_fused_mod_norm_quant import (
    F8_E4M3_MAX,
    fused_rms_mod_quant,
    fused_swiglu_quant,
    _quant_fp8_kernel,
    _rms_mod_res_amax_kernel,
    _swiglu_amax_kernel,
)
import triton

from comfy_kitchen.backends.triton import _register
_register()
import comfy_kitchen as ck

S, K, FFN = 40960, 5376, 14336
dev = "cuda"
torch.manual_seed(0)

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return {"med_ms": round(ts[len(ts)//2], 3), "min_ms": round(ts[0], 3)}

def build():
    d = {}
    d["x"] = torch.randn(S, K, device=dev, dtype=torch.bfloat16) * 0.5
    d["w"] = torch.randn(K, device=dev, dtype=torch.bfloat16).abs() + 0.5
    d["s"] = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1
    d["b"] = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1
    d["gate"] = torch.randn(S, FFN, device=dev, dtype=torch.bfloat16)
    d["up"] = torch.randn(S, FFN, device=dev, dtype=torch.bfloat16)
    return d

# ------------------------------------------------------------------ 簇 1
def c1_baseline_stoch(d):
    """生产同款：rms_norm -> mod -> abs/amax -> GPU RNG -> stochastic_round + 残差。"""
    h = F.rms_norm(d["x"], (K,), d["w"], 1e-5)
    h = h * d["s"] + d["b"]
    amax = h.abs().amax()
    scale = (amax.float() / F8_E4M3_MAX).clamp_min(1e-12)
    rng = torch.randint(0, 256, (h.numel(),), device=dev, dtype=torch.uint8)
    q = ck.stochastic_rounding_fp8(h, rng)
    res = h + d["x"]
    return q, scale, res

def c1_baseline_det(d):
    h = F.rms_norm(d["x"], (K,), d["w"], 1e-5)
    h = h * d["s"] + d["b"]
    amax = h.abs().amax()
    scale = (amax.float() / F8_E4M3_MAX).clamp_min(1e-12)
    q = ck.quantize_per_tensor_fp8(h, scale)
    res = h + d["x"]
    return q, scale, res

def c1_fused(d, **kw):
    return fused_rms_mod_quant(d["x"], d["w"], d["s"], d["b"], 1e-5, **kw)

# ------------------------------------------------------------------ 簇 2
def c2_baseline_stoch(d):
    act = F.silu(d["gate"]) * d["up"]
    amax = act.abs().amax()
    scale = (amax.float() / F8_E4M3_MAX).clamp_min(1e-12)
    rng = torch.randint(0, 256, (act.numel(),), device=dev, dtype=torch.uint8)
    q = ck.stochastic_rounding_fp8(act, rng)
    return q, scale, act

def c2_baseline_det(d):
    act = F.silu(d["gate"]) * d["up"]
    amax = act.abs().amax()
    scale = (amax.float() / F8_E4M3_MAX).clamp_min(1e-12)
    q = ck.quantize_per_tensor_fp8(act, scale)
    return q, scale, act

def c2_fused(d, **kw):
    return fused_swiglu_quant(d["gate"], d["up"], **kw)

# ------------------------------------------------------------------ 数值校验
def numerics(d):
    out = {}
    # 簇 1：fp32 精确参考 + 确定性 eager 基线
    xf = d["x"].float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    h_ref32 = (xf * inv * d["w"].float() * d["s"].float() + d["b"].float())
    res_ref32 = h_ref32 + xf
    q_ref = (h_ref32 / (h_ref32.abs().amax() / F8_E4M3_MAX)).clamp(-F8_E4M3_MAX, F8_E4M3_MAX).to(torch.float8_e4m3fn)

    q, scale, h, res = c1_fused(d)
    out["c1_h_maxabs_vs_fp32ref"] = (h.float() - h_ref32).abs().max().item()
    out["c1_h_maxrel"] = ((h.float() - h_ref32).abs() / h_ref32.abs().clamp_min(1e-6)).max().item()
    out["c1_res_maxabs_vs_fp32ref"] = (res.float() - res_ref32).abs().max().item()
    # 逐位：给定融合 kernel 自己的 h，eager 确定性量化是否逐位一致
    q_det = ck.quantize_per_tensor_fp8(h, scale)
    out["c1_q_bitexact_given_h"] = bool((q.view(torch.uint8) == q_det.view(torch.uint8)).all().item())
    # 端到端 mismatch 率（h 的 1-ulp bf16 差是否翻转 fp8 码字）
    out["c1_q_mismatch_frac_vs_fp32ref"] = (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean().item()
    out["c1_scale_relerr"] = abs(scale.item() - (h_ref32.abs().amax() / F8_E4M3_MAX).item()) / (h_ref32.abs().amax() / F8_E4M3_MAX).item()

    # 簇 2
    a_ref32 = F.silu(d["gate"].float()) * d["up"].float()
    q2, scale2, act = c2_fused(d)
    out["c2_act_maxabs_vs_fp32ref"] = (act.float() - a_ref32).abs().max().item()
    q2_det = ck.quantize_per_tensor_fp8(act, scale2)
    out["c2_q_bitexact_given_act"] = bool((q2.view(torch.uint8) == q2_det.view(torch.uint8)).all().item())
    q2_ref = (a_ref32 / (a_ref32.abs().amax() / F8_E4M3_MAX)).clamp(-F8_E4M3_MAX, F8_E4M3_MAX).to(torch.float8_e4m3fn)
    out["c2_q_mismatch_frac_vs_fp32ref"] = (q2.view(torch.uint8) != q2_ref.view(torch.uint8)).float().mean().item()
    # 确定性可复现（同输入两次调用逐位一致）
    q1b, *_ = c1_fused(d)
    out["c1_deterministic_repeat"] = bool((q.view(torch.uint8) == q1b.view(torch.uint8)).all().item())
    return out

def main():
    d = build()
    res = {"S": S, "K": K, "FFN": FFN, "gpu": torch.cuda.get_device_name()}

    # 配置扫描
    c1_grid = []
    for BK in (512, 1024, 2048):
        for w in (4, 8):
            if BK == 2048 and w == 4:
                continue
            t = bench(lambda: c1_fused(d, BLOCK_K=BK, num_warps=w))
            c1_grid.append({"BLOCK_K": BK, "warps": w, **t})
            print("c1", c1_grid[-1])
    c2_grid = []
    for B in (4096, 8192, 16384):
        for w in (4, 8):
            t = bench(lambda: c2_fused(d, BLOCK=B, num_warps=w))
            c2_grid.append({"BLOCK": B, "warps": w, **t})
            print("c2", c2_grid[-1])

    res["c1_sweep"] = c1_grid
    res["c2_sweep"] = c2_grid
    best1 = min(c1_grid, key=lambda r: r["med_ms"])
    best2 = min(c2_grid, key=lambda r: r["med_ms"])
    res["c1_best"] = best1
    res["c2_best"] = best2

    # QBLOCK 扫描（共用 quant kernel，用簇 2 的大张量）
    q_grid = []
    h2 = F.silu(d["gate"]) * d["up"]
    qout = torch.empty(S * FFN, device=dev, dtype=torch.float8_e4m3fn)
    scale2 = (h2.abs().amax() / F8_E4M3_MAX).clamp_min(1e-12)
    n = h2.numel()
    for QB in (4096, 8192, 16384):
        for w in (4, 8):
            t = bench(lambda: _quant_fp8_kernel[(triton.cdiv(n, QB),)](
                h2, qout, scale2, F8_E4M3_MAX, n, BLOCK=QB, num_warps=w))
            q_grid.append({"QBLOCK": QB, "warps": w, **t})
            print("q", q_grid[-1])
    res["q_sweep"] = q_grid

    # 主对比（最优配置）
    res["c1_baseline_stoch"] = bench(lambda: c1_baseline_stoch(d))
    res["c1_baseline_det"] = bench(lambda: c1_baseline_det(d))
    res["c1_fused_best"] = bench(lambda: c1_fused(d, BLOCK_K=best1["BLOCK_K"], num_warps=best1["warps"]))
    res["c2_baseline_stoch"] = bench(lambda: c2_baseline_stoch(d))
    res["c2_baseline_det"] = bench(lambda: c2_baseline_det(d))
    res["c2_fused_best"] = bench(lambda: c2_fused(d, BLOCK=best2["BLOCK"], num_warps=best2["warps"]))

    res["c1_speedup_vs_stoch"] = round(res["c1_baseline_stoch"]["med_ms"] / res["c1_fused_best"]["med_ms"], 2)
    res["c1_speedup_vs_det"] = round(res["c1_baseline_det"]["med_ms"] / res["c1_fused_best"]["med_ms"], 2)
    res["c2_speedup_vs_stoch"] = round(res["c2_baseline_stoch"]["med_ms"] / res["c2_fused_best"]["med_ms"], 2)
    res["c2_speedup_vs_det"] = round(res["c2_baseline_det"]["med_ms"] / res["c2_fused_best"]["med_ms"], 2)

    res["numerics"] = numerics(d)

    # 带宽口径（簇字节数）
    b1 = S * K * 2  # bf16 每-pass 字节
    fused1_bytes = b1 * 2 + b1 * 2 + b1  # 读 x + 写 h + 写 res(相位A)，读 h + 写 q(相位B, fp8=1B)
    res["c1_fused_gbps"] = round(fused1_bytes / res["c1_fused_best"]["med_ms"] / 1e6, 1)
    f2 = S * FFN * 2
    fused2_bytes = f2 * 2 + f2 + f2 + f2 // 2
    res["c2_fused_gbps"] = round(fused2_bytes / res["c2_fused_best"]["med_ms"] / 1e6, 1)

    with open("/tmp/a2_bench_result.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("c1_sweep", "c2_sweep", "q_sweep")}, indent=1))

if __name__ == "__main__":
    main()
