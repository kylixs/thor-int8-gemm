#!/usr/bin/env python3
# P1/P2：stock vs Triton patched——同 bench 口径（9 形状）＋逐位校验
import os, sys, time, json, statistics
sys.path.insert(0,"/home/nvidia/comfyui-libs")
import torch
import comfy_kitchen.backends.cuda as bc
def bench(tag):
    res=[]
    for (K,N) in [(6144,6144),(6144,16384),(16384,6144)]:
        W=torch.randint(-127,127,(N,K),dtype=torch.int8,device="cuda")
        s=torch.ones(N,device="cuda",dtype=torch.float32)
        for M in [8000,32000,112000]:
            X=torch.randn(M,K,device="cuda",dtype=torch.bfloat16).mul(8).clamp(-100,100)
            for _ in range(3): bc.int8_linear(X,W,s)
            torch.cuda.synchronize(); ts=[]
            for _ in range(6):
                t0=time.perf_counter(); bc.int8_linear(X,W,s); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
            t=statistics.median(ts)
            res.append({"tag":tag,"K":K,"N":N,"M":M,"ms":round(t*1000,2),"TOPS":round(2*M*N*K/t/1e12,1)})
            print(json.dumps(res[-1]),flush=True); del X
    return res
def exact(K=6144,N=6144,M=8000):
    W=torch.randint(-127,127,(N,K),dtype=torch.int8,device="cuda")
    xs=torch.rand(M,device="cuda")+0.5; ws=torch.rand(N,device="cuda")+0.5
    X=torch.randint(-127,127,(M,K),dtype=torch.int8,device="cuda").to(torch.bfloat16)
    a=bc.int8_linear(X,W,ws)  # rowwise act scale internally
    return a
stock=bench("stock")
# 挂 Triton 代理
import importlib.util
spec=importlib.util.spec_from_file_location("ckp","/home/nvidia/thor-int8-gemm/src/ck_int8_patch_thor.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.environ["CK_INT8_MODE"]="triton"
import comfy_kitchen.backends.cuda as bc2
m._apply(bc2, "triton")
tri=bench("triton")
json.dump({"stock":stock,"triton":tri},open("/tmp/thor_bench.json","w"),indent=1)
s=[r["TOPS"] for r in stock]; t=[r["TOPS"] for r in tri]
print(f"STOCK均值 {statistics.mean(s):.1f} TOPS  TRITON均值 {statistics.mean(t):.1f} TOPS  提升 x{statistics.mean(t)/statistics.mean(s):.2f}")
print("P1_DONE")
