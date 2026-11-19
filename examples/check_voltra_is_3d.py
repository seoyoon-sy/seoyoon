"""VOLTRA 해 파일(C_E0.032_X0.04 t=0)이 3D 계산인지 확인 (CLAUDE.md 8.66): z 방향 온도 변화와 내부 노드 열 평형(2D 전도만 대 z 전도 포함)."""
import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"examples"))
import numpy as np
import steady3d_test as S
from hbcalc.presets import try3_model
D=os.path.join(ROOT,"voltra_results","calib_steady_sol")+os.sep
def load(f):
    V={}
    for l in open(f,"rb").read().decode("cp1252").replace("\r","").split("\n"):
        r=l.split()
        if r and r[0]=="N" and len(r)==5: V[(int(r[2]),int(r[3]),int(r[4]))]=float(r[1])
    return V
V=load(D+"C_E0.032_X0.040000.sol"); m=try3_model({"EPS":0.032,"XPS":0.04},snap_mm=1); nz=20
s,E,W3,Kc3,Kz,ext,hk,A2=S.steady3d(m,None,nz=nz); N=s.N
T=np.zeros(N*(nz+1)); have=np.zeros(N*(nz+1),bool)
for (i,j,k),v in V.items():
    if s.idx[i,j]>=0: T[k*N+s.idx[i,j]]=v; have[k*N+s.idx[i,j]]=True
print("모델 노드",N,"x 평면",nz+1,"= 해 파일 격자 노드",have.sum(),"개 일치" if have.all() else "불일치")
# 1) z 방향으로 온도가 변하는가 (평면별 최대 최소 차이)
T2=T.reshape(nz+1,N); span=T2.max(0)-T2.min(0)
print(f"노드별 z 방향 온도 범위(최대 - 최소): 최대 {span.max():.3f} ℃, 중앙값 {np.median(span):.4f} ℃, 0.01 초과 노드 {(span>0.01).sum()} / {N}")
# 2) 내부 노드 열 평형: BC 와 닿지 않는 노드만
mat_nodes=np.zeros(N,bool)
cid=m.cid; nx,ny=cid.shape
interior=np.ones(N,bool)
for i in range(nx+1):
    for j in range(ny+1):
        if s.idx[i,j]<0: continue
        cs=[cid[a,b] for a in (i-1,i) for b in (j-1,j) if 0<=a<nx and 0<=b<ny]
        if any(c<0 for c in cs): interior[s.idx[i,j]]=False
mask=np.tile(interior,nz+1)
r2=(Kc3@T); r3=((Kc3+Kz)@T)
for name,planes in (("끝면 z=0",[0]),("z=1",[1]),("z=2",[2]),("중간 z=10",[10]),("z=19",[19])):
    sel=np.zeros(N*(nz+1),bool)
    for k in planes: sel[k*N:(k+1)*N]=True
    sel&=mask
    print(f"{name:10s} 내부 노드 {sel.sum()}개: 2D 전도만 쓴 잔차 RMS {np.sqrt((r2[sel]**2).mean()):.2e} W 최대 {abs(r2[sel]).max():.2e} | z 전도 포함 RMS {np.sqrt((r3[sel]**2).mean()):.2e} W 최대 {abs(r3[sel]).max():.2e}")
