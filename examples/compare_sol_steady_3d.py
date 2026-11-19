"""정상상태 해 파일(.sol, C_E0.032_X0.04 또는 X0.0001)과 3D 정상상태(z 20조각, 조각별 복사, z 전도)를 비교하고, 2D voltra 결과에 (3D half - 2D half) z 보정을 더한 추정을 낸다 (CLAUDE.md 8.65). 사용: python examples/compare_sol_steady_3d.py 0.04"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"examples"))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
import steady3d_test as S
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.radiation import grey_network
R=ROOT+"/"; D=R+"voltra_results/calib_steady_sol/"
bc=Boundary.from_voltra_files(*(R+g for g in ("wall_TO.FTE","Wall_TI.FTE","Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD")))
def load(f):
    V={}
    for l in open(f,"rb").read().decode("cp1252").replace("\r","").split("\n"):
        r=l.split()
        if r and r[0]=="N" and len(r)==5: V[(int(r[2]),int(r[3]),int(r[4]))]=float(r[1])
    return V
POS=(0,5,10,20,30,50,75,100,150,200,300,500)
X=float(sys.argv[1]); pre={0.04:"C_E0.032_X0.040000",0.0001:"C_E0.032_X0.00010000"}[X]
V=load(D+pre+".sol"); m=try3_model({"EPS":0.032,"XPS":X},snap_mm=1); nz=20
t0=time.time(); s,E,W3,Kc3,Kz,ext,hk,A2=S.steady3d(m,None,nz=nz); print("W3 build",round(time.time()-t0),"s",flush=True)
N=s.N; nE=len(E)
A3=np.array([A2[i]*hk[k] for i in range(nE) for k in range(nz+1)]); E3=[E[i] for i in range(nE) for k in range(nz+1)]
node3=np.array([E[i]["node"]+k*N for i in range(nE) for k in range(nz+1)]); zone3=np.array([E[i]["zone"] for i in range(nE) for k in range(nz+1)])
S3=sp.csr_matrix((np.ones(len(node3)),(node3,np.arange(len(node3)))),shape=(N*(nz+1),len(node3)))
zv={z:((bc.T_out[0],bc.h_out[0]) if m.zones[z].kind=="outdoor" else (bc.T_in[0],bc.h_in[0])) for z in s.zone_ids}
def solve(W):
    Gt,Gs,vs=grey_network(E3,W,A3,m.hrb); Gc=np.zeros(N*(nz+1)); b=np.zeros(N*(nz+1))
    for z,(Ta,h) in zv.items():
        sel=zone3==z; Az=S3[:,np.nonzero(sel)[0]]@A3[sel]; Gc+=h*Az; b+=h*Ta*Az+Ta*(S3@(Gs*sel))
    P=(Kc3+Kz+sp.diags(Gc)).tocsc(); lu=spl.splu(P)
    op=spl.LinearOperator((N*(nz+1),)*2,matvec=lambda x:P@x+S3@(Gt@(S3.T@x))); Mp=spl.LinearOperator((N*(nz+1),)*2,matvec=lambda x:lu.solve(x))
    T,info=spl.cg(op,b,M=Mp,rtol=1e-11,atol=0,maxiter=500); return T.reshape(nz+1,N)
po=[m.outputs[f"surf_{p}"][:2] for p in POS]
for nm,W in (("3D 컷오프 0",W3),("3D 컷오프 1e-4",np.where(((W3/A3[:,None])>=1e-4)&((W3/A3[:,None]).T>=1e-4),W3,0.0))):
    T2=solve(W); print(f"[{nm}] 표면 12위치(오른쪽 벽) 오차, z 평면별 (모사 - VOLTRA)  위치:",POS)
    for k in (10,5,2,0):
        e=[T2[k][s.idx[i,j]]-V[(i,j,k)] for i,j in po]; print(f"  z평면 {k:2d}:"," ".join(f"{v:+.3f}" for v in e))
    # 전체 노드(내부 포함) 중간면
    keys=[q for q in V if q[2]==10 and s.idx[q[0],q[1]]>=0]; d=np.array([T2[10][s.idx[q[0],q[1]]]-V[q] for q in keys]); print(f"  중간면 전체 노드 RMSE {np.sqrt((d**2).mean()):.4f} 평균 {d.mean():+.4f} 최대절대 {abs(d).max():.3f}")
print("VOLTRA 자신의 z 분포(오른쪽 벽 500 mm, 0 mm 표면):",[round(V[(po[11][0],po[11][1],k)],3) for k in (0,2,5,8,10)],[round(V[(po[0][0],po[0][1],k)],3) for k in (0,2,5,8,10)])
print("\n=== 추정: 2D voltra 결과 + (3D half - 2D half) z 보정 (중첩 근사) ===")
s2h=Solver(m,conv_lump="half"); s2h.run(bc,nsteps=1); T2h=s2h.T_all[0]
s2v=Solver(m); s2v.run(bc,nsteps=1); T2v=s2v.T_all[0]
keys=[q for q in V if q[2]==10 and s.idx[q[0],q[1]]>=0]
def rep(nm,Tn):
    d=np.array([Tn[s.idx[q[0],q[1]]]-V[q] for q in keys]); e=[Tn[s.idx[i,j]]-V[(i,j,10)] for i,j in po]
    print(f"  {nm:34s} 전체 RMSE {np.sqrt((d**2).mean()):.4f} 평균 {d.mean():+.4f} 최대 {abs(d).max():.3f} | 12위치 최대절대 {max(abs(np.array(e))):.3f} | "+" ".join(f"{v:+.3f}" for v in e))
rep("2D voltra (현재)",T2v)
for nm,W in (("컷오프 0",W3),("컷오프 1e-4",np.where(((W3/A3[:,None])>=1e-4)&((W3/A3[:,None]).T>=1e-4),W3,0.0))):
    T3=solve(W)[10]; rep(f"2D voltra + z보정({nm})",T2v+(T3-T2h))
