#!/usr/bin/env python3
# P3：tile 参数扫描（BM×BN×BK×warps×stages×G），4 代表形状
import os, sys, time, json, statistics, itertools, importlib.util
sys.path.insert(0,"/home/nvidia/comfyui-libs")
import torch, triton, triton.language as tl
SHAPES=[(32000,6144,6144),(112000,6144,16384),(32000,6144,16384),(112000,16384,6144)]
@triton.jit
def _k(a,b,xs,ws,bias,c,M,N,K,sam,sak,sbn,sbk,scm,scn,ws_n,
       HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,G:tl.constexpr,EVEN:tl.constexpr,EVEN_K:tl.constexpr):
    pid=tl.program_id(0); npm=tl.cdiv(M,BM); npn=tl.cdiv(N,BN)
    gid=pid//(G*npn); fm=gid*G; gs=min(npm-fm,G)
    pm=fm+(pid%gs); pn=(pid%(G*npn))//gs
    rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
    rm64=rm.to(tl.int64); rn64=rn.to(tl.int64)
    A=a+rm64[:,None]*sam+rk[None,:]*sak; B=b+rn64[None,:]*sbn+rk[:,None]*sbk
    acc=tl.zeros((BM,BN),dtype=tl.int32)
    for k0 in range(0,K,BK):
        if EVEN: acc+=tl.dot(tl.load(A),tl.load(B))
        elif EVEN_K: acc+=tl.dot(tl.load(A,mask=rm[:,None]<M,other=0),tl.load(B,mask=rn[None,:]<N,other=0))
        else:
            km=(k0+rk)<K
            acc+=tl.dot(tl.load(A,mask=(rm[:,None]<M)&km[None,:],other=0),tl.load(B,mask=(rn[None,:]<N)&km[:,None],other=0))
        A+=BK*sak; B+=BK*sbk
    out=acc.to(tl.float32)*tl.load(xs+rm,mask=rm<M,other=0.)[:,None]
    out=out*tl.load(ws+rn*ws_n,mask=rn<N,other=0.)[None,:]
    if HAS_BIAS: out+=tl.load(bias+rn,mask=rn<N,other=0.).to(tl.float32)[None,:]
    C=c+rm64[:,None]*scm+rn64[None,:]*scn
    tl.store(C,out.to(c.dtype.element_ty),mask=(rm[:,None]<M)&(rn[None,:]<N))
def prep(M,K,N):
    x8=torch.randint(-127,127,(M,K),dtype=torch.int8,device="cuda")
    w8=torch.randint(-127,127,(N,K),dtype=torch.int8,device="cuda")
    xs=torch.rand(M,device="cuda",dtype=torch.float32)+.5
    ws=torch.rand(N,device="cuda",dtype=torch.float32)+.5
    out=torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
    return x8,w8,xs,ws,out
def trial(M,K,N,BM,BN,BK,w,s,G):
    x8,w8,xs,ws,out=prep(M,K,N)
    ek=K%BK==0; ev=ek and M%BM==0 and N%BN==0
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    def go():
        _k[grid](x8,w8,xs,ws,out,out,M,N,K,x8.stride(0),x8.stride(1),w8.stride(0),w8.stride(1),out.stride(0),out.stride(1),1,
                 HAS_BIAS=False,BM=BM,BN=BN,BK=BK,G=G,EVEN=ev,EVEN_K=ek,num_warps=w,num_stages=s)
    try:
        for _ in range(2): go()
        torch.cuda.synchronize(); ts=[]
        for _ in range(3):
            t0=time.perf_counter(); go(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        t=statistics.median(ts); return 2*M*N*K/t/1e12, t
    except Exception as e:
        return 0.0, 0.0
res=[]
GRID=list(itertools.product([64,128,256],[128,256],[32,64,128],[4,8],[3,4,5],[4,8]))
# 限量：挑合法组合（BM*BN≤32768、BK≥16）
GRID=[g for g in GRID if g[0]*g[1]<=32768]
print(f"{len(GRID)} 配置 × {len(SHAPES)} 形状",flush=True)
for i,(BM,BN,BK,w,s,G) in enumerate(GRID):
    rs=[]
    for (M,K,N) in SHAPES:
        tops,t=trial(M,K,N,BM,BN,BK,w,s,G); rs.append(tops)
    mean=sum(rs)/len(rs)
    res.append({"cfg":f"{BM}x{BN}x{BK}w{w}s{s}g{G}","mean_TOPS":round(mean,1),"per_shape":[round(x,1) for x in rs]})
    print(json.dumps(res[-1]),flush=True)
res.sort(key=lambda r:-r["mean_TOPS"])
json.dump(res[:10],open("/tmp/sweep_top.json","w"),indent=1)
print("P3_DONE top:",res[0])
