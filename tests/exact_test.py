#!/usr/bin/env python3
# P2：stock vs Triton 逐位校验（随机形状×多档 M/K/N/含 bias）
import os, sys, importlib.util
sys.path.insert(0,"/home/nvidia/comfyui-libs")
import torch
import comfy_kitchen.backends.cuda as bc
def run(M,K,N,bias):
    W=torch.randint(-127,127,(N,K),dtype=torch.int8,device="cuda")
    ws=torch.rand(N,device="cuda")+0.5
    b=torch.randn(N,device="cuda",dtype=torch.bfloat16) if bias else torch.empty(0,device="cuda",dtype=torch.bfloat16)
    X=torch.randn(M,K,device="cuda",dtype=torch.bfloat16).mul(8).clamp(-120,120)
    return bc.int8_linear(X,W,ws,bias=b)
cases=[(128,6144,6144,True),(8000,6144,6144,False),(32000,6144,16384,True),(96,16384,6144,False),(7777,6143,6144,True)]
ref=[run(*c) for c in cases]
spec=importlib.util.spec_from_file_location("ckp","/home/nvidia/thor-int8-gemm/src/ck_int8_patch_thor.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.environ["CK_INT8_MODE"]="triton"; m._apply(bc,"triton")
ok=0
for c,a in zip(cases,ref):
    b=run(*c)
    eq=torch.equal(a,b)
    print(c,"逐位一致" if eq else f"不一致 maxdiff={(a.float()-b.float()).abs().max().item()}")
    ok+=eq
print(f"P2: {ok}/{len(cases)} 全等")
