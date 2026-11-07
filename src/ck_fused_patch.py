# SPDX-License-Identifier: Apache-2.0
"""A3 fused-layer .pth agent: hooks the A2 Triton fused kernels into ComfyUI's
MiniMax H3 fp8 DiT pipeline.

Loaded from a .pth in site-packages (see deploy notes in the repo). Inert unless
enabled — env CK_FUED... (CK_FUSED) or /etc/ck-fused.mode says so. Modes:
  off / unset - stock H3 block (bit-identical: nothing is imported or patched)
  on          - both clusters fused
  c1          - cluster 1 only (rms_norm+modulation -> pre-quantized fp8 input)
  c2          - cluster 2 only (SwiGLU+amax -> pre-quantized fp8 fc2 input)

What it replaces (comfy.ldm.minimax.model.DiTBlock.forward, H3 on fp8 weights):
  cluster 1: per-segment rms_norm(x,w) * (1+scale) + shift  +  tensorwise fp8
             quantization of h (amax over the WHOLE packed sequence, matching
             QuantizedTensor.from_float's single per-tensor scale) -> the fp8
             QuantizedTensor is handed to qkv_proj / fc1, which then skip their
             own from_float quantize and go straight to _scaled_mm.
             The gated-residual addcmul_ (_mod_gate) stays stock.
  cluster 2: fc1 out -> chunk -> fused SwiGLU+amax+fp8 quant -> fc2 on the
             pre-quantized activation.
Kernels are ck_fused_mod_norm_quant.py (A2, host2-validated; deterministic RTN,
quant bit-identical to comfy_kitchen quantize_per_tensor_fp8 on equal h).

Guards: comfy-kitchen must be the validated version; any ineligible shape
(3D/non-bf16/CPU input, per-token mod rows from inpaint masks, non-fp8 weights,
training mode) or any runtime failure falls back to the captured stock forward
for the rest of the process, with a one-shot warning on stderr.

The patch target is the class attribute `DiTBlock.forward` on the
`comfy.ldm.minimax.model` module (callers resolve it as module.DiTBlock each
time — no stale from-import bindings), same agent pattern as
src/ck_int8_patch_thor.py.
"""
import importlib.abc
import importlib.machinery
import os
import sys

MODE_FILE = "/etc/ck-fused.mode"
TARGET = "comfy.ldm.minimax.model"
VALIDATED_KITCHEN = ("0.2.40",)  # H3 fp8 pipeline verified on this build (A3 2026-11)
_TAG = "[ck-fused-patch]"


def _read_mode():
    m = os.environ.get("CK_FUSED")
    if m is not None and m != "":
        m = m.strip().lower()
        return "on" if m == "1" else m
    try:
        with open(MODE_FILE) as f:
            return f.read().strip().lower() or "off"
    except OSError:
        return "off"


def _kitchen_version():
    try:
        from importlib.metadata import version
        return version("comfy-kitchen")
    except Exception:
        return None


class _Fused:
    """Lazily-bound runtime state; built after comfy.ldm.minimax.model imports."""

    def __init__(self, model_mod, mode):
        import torch  # noqa: F401  (presence check only; kernels need CUDA torch+triton)
        import triton  # noqa: F401
        import ck_fused_mod_norm_quant as fk
        import comfy.model_management
        from comfy.quant_ops import QuantizedTensor, TensorCoreFP8E4M3Layout
        self.ok = True
        self.torch = torch
        self.fk = fk
        self.mm = comfy.model_management
        self.QuantizedTensor = QuantizedTensor
        self.Fp8Layout = TensorCoreFP8E4M3Layout
        self.model = model_mod
        self.c1 = mode in ("on", "c1")
        self.c2 = mode in ("on", "c2")
        self.broken = False
        self.warned = False

    # ------------------------------------------------------------ eligibility
    def eligible(self, block, x, mod_segments):
        if self.broken or self.mm.in_training:
            return False
        if x.dim() != 2 or not x.is_cuda or x.dtype != self.torch.bfloat16 or not x.is_contiguous():
            return False
        qt = self.QuantizedTensor
        if self.c1 and not (isinstance(block.attn.qkv_proj.weight, qt) and isinstance(block.mlp.fc1.weight, qt)):
            return False
        if self.c2 and not isinstance(block.mlp.fc2.weight, qt):
            return False
        # per-token mod rows (inpaint masks) give every row its own scale/shift
        # vector — the A2 kernel takes exactly one (s, b) per launch segment
        for _a, _b, row in mod_segments:
            if not isinstance(row, int):
                return False
        return True

    # ------------------------------------------------------------ cluster 1
    def norm_mod_quant(self, norm, x, shift, scale, mod_segments):
        """rms_norm(x,w)*(1+scale[row])+shift[row] per segment + whole-tensor
        fp8 quantization. Returns an fp8 QuantizedTensor standing in for h."""
        torch = self.torch
        fk = self.fk
        S, K = x.shape
        w = self.mm.cast_to(norm.weight, device=x.device)
        h = torch.empty_like(x)
        res = torch.empty_like(x)  # kernel writes it; unused (residual gate stays stock)
        amax = torch.zeros((), dtype=torch.float32, device=x.device)
        BLOCK_K = 2048
        n_chunk = -(-K // BLOCK_K)
        eps = float(norm.eps)
        for a, b, row in mod_segments:
            # match _mod_scale_shift exactly: cast to bf16 first, then 1+scale in bf16
            s = 1.0 + scale[row].to(x.dtype)
            sh = shift[row].to(x.dtype)
            fk._rms_mod_res_amax_kernel[(b - a,)](
                x[a:b], w, s, sh, h[a:b], res[a:b], amax, K, eps,
                BLOCK_K=BLOCK_K, N_CHUNK=n_chunk, num_warps=8,
            )
        qscale = (amax / fk.F8_E4M3_MAX).clamp_min(1e-12)
        q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        n = q.numel()
        fk._quant_fp8_kernel[(-(-n // 4096),)](
            h, q, qscale, fk.F8_E4M3_MAX, n, BLOCK=4096, num_warps=4,
        )
        params = self.Fp8Layout.Params(scale=qscale, orig_dtype=x.dtype, orig_shape=(S, K))
        return self.QuantizedTensor(q, "TensorCoreFP8E4M3Layout", params)

    # ------------------------------------------------------------ cluster 2
    def swiglu_linear(self, mlp, h_q):
        torch = self.torch
        fk = self.fk
        y = mlp.fc1(h_q)  # [S, 2*ffn] bf16 out of the fp8 GEMM
        gate, up = y.chunk(2, dim=-1)
        if not (gate.is_contiguous() and up.is_contiguous()):
            gate, up = gate.contiguous(), up.contiguous()
        q, qscale, _act = fk.fused_swiglu_quant(gate, up)
        params = self.Fp8Layout.Params(scale=qscale, orig_dtype=y.dtype, orig_shape=gate.shape)
        qt = self.QuantizedTensor(q, "TensorCoreFP8E4M3Layout", params)
        return mlp.fc2(qt)

    # ------------------------------------------------------------ block fwd
    def block_forward(self, block, x, t_emb, mod_segments, rope_freqs,
                      transformer_options, attention):
        model = self.model
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        attn = block.attn if attention is None else attention
        if self.c1:
            h1 = self.norm_mod_quant(block.norm1, x, shift_msa, scale_msa, mod_segments)
        else:
            h1 = model._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
        x = model._mod_gate(x, gate_msa, attn(h1, rope_freqs=rope_freqs,
                                               transformer_options=transformer_options), mod_segments)
        # h2 reads the residual stream AFTER the attention gate update (stock order)
        if self.c1:
            h2 = self.norm_mod_quant(block.norm2, x, shift_mlp, scale_mlp, mod_segments)
        else:
            h2 = model._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
        if self.c2:
            out = self.swiglu_linear(block.mlp, h2)
        else:
            out = block.mlp(h2)
        return model._mod_gate(x, gate_mlp, out, mod_segments)

    def fail(self, exc):
        import traceback
        self.broken = True
        if not self.warned:
            self.warned = True
            print(f"{_TAG} fused path failed ({exc!r:.200}); stock DiTBlock for the rest of this process\n"
                  + traceback.format_exc(), file=sys.stderr, flush=True)


def _apply(mod, mode):
    fused = [None]
    try:
        import comfy.model_management  # noqa: F401  fail fast if comfy core is unavailable
        if not all(hasattr(mod, n) for n in ("DiTBlock", "_mod_scale_shift", "_mod_gate", "MLP")):
            print(f"{_TAG} NOT applied: comfy.ldm.minimax.model layout not recognized; running stock",
                  file=sys.stderr, flush=True)
            return
        fused[0] = _Fused(mod, mode)
        orig_forward = mod.DiTBlock.forward
        f = fused[0]

        def ck_fused_forward(self, x, t_emb, mod_segments, rope_freqs,
                             transformer_options={}, attention=None):
            if f.ok and not f.broken and f.eligible(self, x, mod_segments):
                try:
                    return f.block_forward(self, x, t_emb, mod_segments, rope_freqs,
                                           transformer_options, attention)
                except Exception as exc:  # stay numerically safe: stock from here on
                    f.fail(exc)
            return orig_forward(self, x, t_emb, mod_segments, rope_freqs,
                                transformer_options=transformer_options, attention=attention)

        mod.DiTBlock.forward = ck_fused_forward
        print(f"{_TAG} comfy-kitchen {_kitchen_version()} H3 DiTBlock fused mode={mode} "
              f"(c1={f.c1} c2={f.c2})", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"{_TAG} NOT applied ({exc!r:.200}); running stock", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, mode):
        self.mode = mode

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        loader, mode = spec.loader, self.mode
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _apply(module, mode)

        loader.exec_module = exec_module
        return spec


def install():
    mode = _read_mode()
    if mode not in ("on", "c1", "c2"):
        if mode not in ("off", ""):
            print(f"{_TAG} unknown mode {mode!r}; running stock", file=sys.stderr, flush=True)
        return
    ver = _kitchen_version()
    if ver not in VALIDATED_KITCHEN:
        print(f"{_TAG} NOT applied: comfy-kitchen {ver} is not validated "
              f"({', '.join(VALIDATED_KITCHEN)}); running stock", file=sys.stderr, flush=True)
        return
    # the A2 kernel module ships next to this file; make sure it is importable
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    if TARGET in sys.modules:
        _apply(sys.modules[TARGET], mode)
    else:
        sys.meta_path.insert(0, _Finder(mode))


try:
    install()
except Exception as exc:  # never break interpreter start-up
    print(f"{_TAG} disabled: {exc!r}", file=sys.stderr)
