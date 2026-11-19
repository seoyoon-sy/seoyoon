"""정상상태 해 파일 3개(X0.04, X0.0001, X4.3)에 대해 형상계수 mid와 avg의 혼합 비중 z_blend별 온도장 오차 (CLAUDE.md 8.65). 사용: python examples/compare_sol_zblend.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
R=ROOT+"/"; D=R+"voltra_results/calib_steady_sol/"
bc=Boundary.from_voltra_files(*(R+g for g in ("wall_TO.FTE","Wall_TI.FTE","Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD")))
def load(f):
    V={}
    for l in open(f,"rb").read().decode("cp1252").replace("\r","").split("\n"):
        r=l.split()
        if r and r[0]=="N" and len(r)==5: V[(int(r[2]),int(r[3]),int(r[4]))]=float(r[1])
    return V
cases=[("C_E0.032_X0.040000",0.04,0),("C_E0.032_X0.00010000",0.0001,0),("C_E0.032_X4.30001",4.3,1)]
AL=(0,0.2,0.25,0.3,0.35,0.4,0.5)
res={}
for pre,X,step in cases:
    V=load(D+pre+".sol"); m=try3_model({"EPS":0.032,"XPS":X},snap_mm=1)
    sm=Solver(m,zmode="mid"); sa=Solver(m,zmode="avg"); Wm,Wa=sm.vf_faces['W'].copy(),sa.vf_faces['W'].copy()
    for a in AL:
        s=Solver(m,zmode="mid"); s._setup_voltra_surface((1-a)*Wm+a*Wa); s.run(bc,nsteps=step+1); T=s.T_all[step]
        keys=[k for k in V if k[2]==10 and s.idx[k[0],k[1]]>=0]; d=np.array([T[s.idx[k[0],k[1]]]-V[k] for k in keys])
        e0=T[s.idx[m.outputs['surf_0'][0],m.outputs['surf_0'][1]]]-V[(m.outputs['surf_0'][0],m.outputs['surf_0'][1],10)]
        e5=T[s.idx[m.outputs['surf_500'][0],m.outputs['surf_500'][1]]]-V[(m.outputs['surf_500'][0],m.outputs['surf_500'][1],10)]
        res[(X,a)]=(np.sqrt((d**2).mean()),d.mean(),abs(d).max(),e0,e5)
print("alpha(avg 비중) | 케이스별 RMSE / 평균 / 최대 / 0mm / 500mm")
for a in AL:
    print(f"{a:.2f} | "+" | ".join(f"X{X}: {r[0]:.4f} {r[1]:+.4f} {r[2]:.3f} {r[3]:+.3f} {r[4]:+.3f}" for X in (0.04,0.0001,4.3) for r in [res[(X,a)]]))
