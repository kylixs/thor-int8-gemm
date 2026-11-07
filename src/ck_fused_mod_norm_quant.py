# SPDX-License-Identifier: Apache-2.0
"""A2 融合层 Triton 原型（簇1/簇2）——rms_norm+modulation+residual+amax 单 kernel
＋fp8 tensorwise 量化；SwiGLU+amax 单 kernel＋量化。

语义对齐 comfy_kitchen 现役 fp8 tensorwise 管线（_scaled_mm_v2 路线）：
  scale = amax(|h|) / F8_E4M3_MAX;  q = clamp(h/scale, ±448) -> fp8 RTN
差异（有意）：随机数改确定性 round-to-nearest（A1 发现每步 0.42s 纯 GPU RNG），
因此数值校验可走逐位对比（同 h 输入下与 eager 参考逐位相等）。

用法（独立于 GEMM patch，可单挂）：
    from ck_fused_mod_norm_quant import fused_rms_mod_quant, fused_swiglu_quant
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

F8_E4M3_MAX = 448.0


# ---------------------------------------------------------------- 簇 1 phase A
@triton.jit
def _rms_mod_res_amax_kernel(
    x_ptr, w_ptr, s_ptr, b_ptr,          # in: 激活、norm 权重、mod scale、mod shift
    h_ptr, res_ptr, amax_ptr,            # out: 调制后隐藏态(bf16)、残差(bf16)、全局 amax(fp32)
    K, eps,
    BLOCK_K: tl.constexpr, N_CHUNK: tl.constexpr,
):
    """一行一个 program：x -> rms_norm(w) -> *s + b -> h；res = h + x；
    行内 |h| 最大值 atomic_max 进全局 amax。"""
    row = tl.program_id(0)
    base = row.to(tl.int64) * K

    ss = tl.zeros([BLOCK_K], dtype=tl.float32)
    for i in tl.static_range(N_CHUNK):
        offs = i * BLOCK_K + tl.arange(0, BLOCK_K)
        m = offs < K
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        ss += x * x
    inv = tl.math.rsqrt(tl.sum(ss) / K + eps)

    rowmax = tl.zeros([1], dtype=tl.float32)
    for i in tl.static_range(N_CHUNK):
        offs = i * BLOCK_K + tl.arange(0, BLOCK_K)
        m = offs < K
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        s = tl.load(s_ptr + offs, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
        h = x * inv * w * s + b
        tl.store(h_ptr + base + offs, h.to(tl.bfloat16), mask=m)
        tl.store(res_ptr + base + offs, (h + x).to(tl.bfloat16), mask=m)
        rowmax = tl.maximum(rowmax, tl.max(tl.abs(h))[None])
    tl.atomic_max(amax_ptr, tl.max(rowmax))


# ---------------------------------------------------------------- 簇 2 phase A
@triton.jit
def _swiglu_amax_kernel(
    g_ptr, u_ptr, act_ptr, amax_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """act = silu(gate) * up（bf16 写出）＋ 分块 |act| amax atomic。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    a = g * tl.sigmoid(g) * u
    tl.store(act_ptr + offs, a.to(tl.bfloat16), mask=m)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(a)))


# ---------------------------------------------------------------- phase B 共用
@triton.jit
def _quant_fp8_kernel(
    h_ptr, q_ptr, scale_ptr,
    lp_max, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    h = tl.load(h_ptr + offs, mask=m, other=0.0).to(tl.float32)
    scaled = h / tl.load(scale_ptr)
    clamped = tl.minimum(tl.maximum(scaled, -lp_max), lp_max)
    tl.store(q_ptr + offs, clamped.to(tl.float8e4nv), mask=m)


@triton.jit
def _quant_fp8_amax_kernel(
    h_ptr, q_ptr, amax_ptr,
    lp_max, n_elements,
    SCALE_MIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A3' device-scale 版：phase B 直接读 device 上的 amax 指针，
    kernel 内算 scale = max(amax/448, 1e-12) 再除/钳位/RTN——host 全程不碰数，
    消除两阶段设计里每个量化位点一次的 GPU->CPU 同步。
    数学与 _quant_fp8_kernel 逐位一致（fp32 除法＋同 clamp 语义）。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    amax = tl.load(amax_ptr)
    scale = tl.maximum(amax / lp_max, SCALE_MIN)
    h = tl.load(h_ptr + offs, mask=m, other=0.0).to(tl.float32)
    scaled = h / scale
    clamped = tl.minimum(tl.maximum(scaled, -lp_max), lp_max)
    tl.store(q_ptr + offs, clamped.to(tl.float8e4nv), mask=m)


def _make_scale(amax: torch.Tensor) -> torch.Tensor:
    """amax(fp32 标量) -> scale，避免除 0。device 上的异步小算子，仅供
    下游 QuantizedTensor.Params 等需要 scale 张量的场合；phase B 量化
    不再依赖它（走 _quant_fp8_amax_kernel）。"""
    return (amax / F8_E4M3_MAX).clamp_min(1e-12)


def fused_rms_mod_quant(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, b: torch.Tensor,
    eps: float = 1e-5,
    BLOCK_K: int = 2048, num_warps: int = 8,      # A2 扫描最优（512/1024/2048×4/8 warps 差 <1%）
    QBLOCK: int = 4096, q_num_warps: int = 4,
):
    """簇 1：rms_norm+modulation+residual+amax(1 kernel) ＋ fp8 量化(1 kernel)。
    返回 (q fp8, scale, h bf16, res bf16)。"""
    S, K = x.shape
    h = torch.empty_like(x)
    res = torch.empty_like(x)
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    n_chunk = triton.cdiv(K, BLOCK_K)
    _rms_mod_res_amax_kernel[(S,)](
        x, w, s, b, h, res, amax, K, eps,
        BLOCK_K=BLOCK_K, N_CHUNK=n_chunk, num_warps=num_warps,
    )
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    n = q.numel()
    _quant_fp8_amax_kernel[(triton.cdiv(n, QBLOCK),)](
        h, q, amax, F8_E4M3_MAX, n, SCALE_MIN=1e-12, BLOCK=QBLOCK,
        num_warps=q_num_warps,
    )
    scale = _make_scale(amax)  # 仅供下游 Params；不阻塞量化
    return q, scale, h, res


def fused_swiglu_quant(
    gate: torch.Tensor, up: torch.Tensor,
    BLOCK: int = 4096, num_warps: int = 8,        # A2 扫描最优
    QBLOCK: int = 4096, q_num_warps: int = 4,
):
    """簇 2：SwiGLU+amax(1 kernel) ＋ fp8 量化(1 kernel)。返回 (q, scale, act bf16)。"""
    assert gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
    act = torch.empty_like(gate)
    amax = torch.zeros((), dtype=torch.float32, device=gate.device)
    n = gate.numel()
    _swiglu_amax_kernel[(triton.cdiv(n, BLOCK),)](
        gate, up, act, amax, n, BLOCK=BLOCK, num_warps=num_warps,
    )
    q = torch.empty(gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    _quant_fp8_amax_kernel[(triton.cdiv(n, QBLOCK),)](
        act, q, amax, F8_E4M3_MAX, n, SCALE_MIN=1e-12, BLOCK=QBLOCK,
        num_warps=q_num_warps,
    )
    scale = _make_scale(amax)  # 仅供下游 Params；不阻塞量化
    return q, scale, act
