"""격자 시험 L0 ~ L3(VOLTRA 정상상태)에 3D z 효과 추정(2D voltra + (3D half - 2D half))을 넣어 비교 (8.76). 사용: python examples/compare_mesh_study_3d.py 1.6 10 L0,L1,L2"""
import os,sys,time; ROOT="/home/user/seoyoon"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/examples")
import numpy as np
import steady3d_test as S3
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model, try3_excel_strip_props
from hbcalc.voltra_report import read_report
LV={"L0":(10,100,40),"L1":(5,50,20),"L2":(2.5,25,10),"L3":(1.25,12.5,5)}
P=(0,5,10,20,30,50,75,100,150,200,300,500)
bc=Boundary.from_voltra_files(*(os.path.join(ROOT,f) for f in ("wall_TO.FTE","Wall_TI.FTE","Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD")))
x=float(sys.argv[1]); nz=int(sys.argv[2]); lvs=sys.argv[3].split(",")
for lv in lvs:
    a,b,c=LV[lv]; cols,A=read_report(f"{ROOT}/voltra_results/mesh_study/MS_X{x:g}_{lv}.txt"); V=np.array(A[0][7:31][0::2])
    rho,cp=try3_excel_strip_props(x)
    m=try3_model({"EPS":0.032,"XPS":x},h_min_mm=a,h_max_mm=b,h_layer_mm=c,symmetric_mesh=True).with_properties(rho={"XPS":rho},cp={"XPS":cp})
    sv=Solver(m); sv.run(bc,nsteps=1); idx=[sv.idx[i,j] for i,j,*_ in [m.outputs[f"surf_{p}"] for p in P]]
    T2v=sv.T_all[0][idx]
    sh=Solver(m,conv_lump="half"); sh.run(bc,nsteps=1); T2h=sh.T_all[0][idx]
    t=time.time(); s,E,W3,Kc3,Kz,ext,hk,A2=S3.steady3d(m,None,nz=nz)
    zv={z:((bc.T_out[0],bc.h_out[0]) if m.zones[z].kind=="outdoor" else (bc.T_in[0],bc.h_in[0])) for z in s.zone_ids}
    T3=S3.solve3d(s,E,W3,Kc3,Kz,ext,hk,A2,zv,nz=nz,hrb=m.hrb)[0][nz//2][idx]; dt=time.time()-t
    est=T2v+(T3-T2h)
    f=lambda d:f"{abs(d).mean():.3f}/{abs(d).max():.3f} 0mm {d[0]:+.3f} 500mm {d[-1]:+.3f}"
    print(f"X{x:g} {lv} 2D노드 {s.N} nE {len(E)} nz {nz} 3D {dt:.0f}s | 2D-V {f(T2v-V)} | 3D추정-V {f(est-V)} | 500mm V {V[-1]:.2f} 2D {T2v[-1]:.3f} 3D추정 {est[-1]:.3f}",flush=True)
