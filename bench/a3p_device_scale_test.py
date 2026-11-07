#!/usr/bin/env python3
"""A3' device-scale 验证（主机2）：
1) 逐位一致：新 _quant_fp8_amax_kernel（scale 留 device）vs 旧两阶段
   （phase A -> host 语义的 _make_scale -> _quant_fp8_kernel）逐位相等；
   簇 1、簇 2 两条链都验；含退化输入（全零 h -> amax=0 -> scale 落到 1e-12 下限）。
2) bench：fused_rms_mod_quant / fused_swiglu_quant（新）vs 手搓旧两阶段，
   加速比不得劣化（容差 3%）。
"""
import sys

import torch
import triton

sys.path.insert(0, "/home/nvidia/a3p")
from ck_fused_mod_norm_quant import (
    F8_E4M3_MAX, _make_scale, _quant_fp8_amax_kernel, _quant_fp8_kernel,
    _rms_mod_res_amax_kernel, _swiglu_amax_kernel,
    fused_rms_mod_quant, fused_swiglu_quant,
)

dev = "cuda"
S, K, FFN = 40960, 5376, 14336
torch.manual_seed(0)
results = {}


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
    return ts[len(ts) // 2]


# ------------------------------------------------ 模块级旧两阶段实现（对照臂）
def old_rms_mod_quant(x, w, s, b, eps=1e-5, zero_amax=False):
    Sk, Kk = x.shape
    h = torch.empty_like(x); res = torch.empty_like(x)
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    n_chunk = triton.cdiv(Kk, 2048)
    _rms_mod_res_amax_kernel[(Sk,)](x, w, s, b, h, res, amax, Kk, eps,
                                    BLOCK_K=2048, N_CHUNK=n_chunk, num_warps=8)
    if zero_amax:
        amax.zero_()
    scale = _make_scale(amax)
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    n = q.numel()
    _quant_fp8_kernel[(triton.cdiv(n, 4096),)](h, q, scale, F8_E4M3_MAX, n,
                                               BLOCK=4096, num_warps=4)
    return q, scale, h, res


def old_swiglu_quant(gate, up, zero_amax=False):
    act = torch.empty_like(gate)
    amax = torch.zeros((), dtype=torch.float32, device=gate.device)
    n = gate.numel()
    _swiglu_amax_kernel[(triton.cdiv(n, 4096),)](gate, up, act, amax, n,
                                                 BLOCK=4096, num_warps=8)
    if zero_amax:
        amax.zero_()
    scale = _make_scale(amax)
    q = torch.empty(gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    _quant_fp8_kernel[(triton.cdiv(n, 4096),)](act, q, scale, F8_E4M3_MAX, n,
                                               BLOCK=4096, num_warps=4)
    return q, scale, act


def new_rms_mod_quant(x, w, s, b, eps=1e-5, zero_amax=False):
    # 同 fused_rms_mod_quant，但支持 zero_amax 以测 scale 下限路径
    Sk, Kk = x.shape
    h = torch.empty_like(x); res = torch.empty_like(x)
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    n_chunk = triton.cdiv(Kk, 2048)
    _rms_mod_res_amax_kernel[(Sk,)](x, w, s, b, h, res, amax, Kk, eps,
                                    BLOCK_K=2048, N_CHUNK=n_chunk, num_warps=8)
    if zero_amax:
        amax.zero_()
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    n = q.numel()
    _quant_fp8_amax_kernel[(triton.cdiv(n, 4096),)](h, q, amax, F8_E4M3_MAX, n,
                                                    SCALE_MIN=1e-12, BLOCK=4096,
                                                    num_warps=4)
    scale = _make_scale(amax)
    return q, scale, h, res


def new_swiglu_quant(gate, up, zero_amax=False):
    act = torch.empty_like(gate)
    amax = torch.zeros((), dtype=torch.float32, device=gate.device)
    n = gate.numel()
    _swiglu_amax_kernel[(triton.cdiv(n, 4096),)](gate, up, act, amax, n,
                                                 BLOCK=4096, num_warps=8)
    if zero_amax:
        amax.zero_()
    q = torch.empty(gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    _quant_fp8_amax_kernel[(triton.cdiv(n, 4096),)](act, q, amax, F8_E4M3_MAX, n,
                                                    SCALE_MIN=1e-12, BLOCK=4096,
                                                    num_warps=4)
    scale = _make_scale(amax)
    return q, scale, act


# ---------------------------------------------------------------- 数值逐位
d = {
    "x": torch.randn(S, K, device=dev, dtype=torch.bfloat16) * 0.5,
    "w": torch.randn(K, device=dev, dtype=torch.bfloat16).abs() + 0.5,
    "s": torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1,
    "b": torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1,
    "gate": torch.randn(S, FFN, device=dev, dtype=torch.bfloat16),
    "up": torch.randn(S, FFN, device=dev, dtype=torch.bfloat16),
}

for zero in (False, True):
    tag = "amax=0" if zero else "normal"
    qo, so, ho, _ = old_rms_mod_quant(d["x"], d["w"], d["s"], d["b"], zero_amax=zero)
    qn, sn, hn, _ = new_rms_mod_quant(d["x"], d["w"], d["s"], d["b"], zero_amax=zero)
    c1_eq = bool((qo.view(torch.uint8) == qn.view(torch.uint8)).all().item())
    c1_sc = bool(torch.equal(so, sn)) and so.device.type == "cuda"
    c1_h = bool(torch.equal(ho, hn))

    q2o, s2o, ao = old_swiglu_quant(d["gate"], d["up"], zero_amax=zero)
    q2n, s2n, an = new_swiglu_quant(d["gate"], d["up"], zero_amax=zero)
    c2_eq = bool((q2o.view(torch.uint8) == q2n.view(torch.uint8)).all().item())
    c2_sc = bool(torch.equal(s2o, s2n))
    c2_a = bool(torch.equal(ao, an))
    print(f"[{tag}] c1 q bitexact={c1_eq} scale_eq={c1_sc} h_eq={c1_h} | "
          f"c2 q bitexact={c2_eq} scale_eq={c2_sc} act_eq={c2_a}")
    results[f"c1_bitexact_{tag}"] = c1_eq and c1_sc and c1_h
    results[f"c2_bitexact_{tag}"] = c2_eq and c2_sc and c2_a

# ---------------------------------------------------------------- bench
fns = {
    "c1_old": lambda: old_rms_mod_quant(d["x"], d["w"], d["s"], d["b"]),
    "c1_new": lambda: new_rms_mod_quant(d["x"], d["w"], d["s"], d["b"]),
    "c1_new_api": lambda: fused_rms_mod_quant(d["x"], d["w"], d["s"], d["b"]),
    "c2_old": lambda: old_swiglu_quant(d["gate"], d["up"]),
    "c2_new": lambda: new_swiglu_quant(d["gate"], d["up"]),
    "c2_new_api": lambda: fused_swiglu_quant(d["gate"], d["up"]),
}
ms = {k: round(bench(f), 3) for k, f in fns.items()}
for k, v in ms.items():
    print(f"{k:14s} {v:8.3f} ms")
r1 = ms["c1_old"] / ms["c1_new"]
r2 = ms["c2_old"] / ms["c2_new"]
print(f"c1 old/new = {r1:.3f}x  c2 old/new = {r2:.3f}x")
results.update(ms_c1_old=ms["c1_old"], ms_c1_new=ms["c1_new"],
               ms_c2_old=ms["c2_old"], ms_c2_new=ms["c2_new"],
               api_eq_c1=abs(ms["c1_new_api"] - ms["c1_new"]) < 0.05,
               api_eq_c2=abs(ms["c2_new_api"] - ms["c2_new"]) < 0.05)
ok = (all(results[f"c1_bitexact_{t}"] and results[f"c2_bitexact_{t}"]
          for t in ("normal", "amax=0"))
      and r1 > 0.97 and r2 > 0.97
      and results["api_eq_c1"] and results["api_eq_c2"])
print("A3'-HOST2:", "PASS" if ok else "FAIL", results)
