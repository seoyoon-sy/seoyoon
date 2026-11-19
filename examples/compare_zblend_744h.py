"""N1 ~ N6 744시간 표면온도: 확산일사 차단(diffuse_obstruction)과 z_blend 네 조합 비교 (CLAUDE.md 8.64, 8.65). 사용: python examples/compare_zblend_744h.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"examples"))
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
import test_walls_def as T
from concurrent.futures import ProcessPoolExecutor
R=ROOT+"/"; N=744; mk=night_mask(N); POS=(0,5,10,20,30,50,75,100,150,200,300,500)
CFG={"차단끔 혼합0":(False,0.0),"차단켬 혼합0":(True,0.0),"차단끔 혼합0.4":(False,0.4),"차단켬 혼합0.4":(True,0.4)}
def job(a):
    key,mode,c=a; do,zb=CFG[c]
    ti="Wall_TI.FTE" if mode=="난방" else "Wall_TI_2.FTE"
    d=R+("voltra_results/test_walls/" if mode=="난방" else "voltra_results/test_walls_nonheating/")
    fj,fp=d+f"{key}_junction.txt",d+f"{key}_planar.txt"
    if not (os.path.exists(fj) and os.path.exists(fp)): return a,None
    bc=Boundary.from_voltra_files(*(R+g for g in ("wall_TO.FTE",ti,"Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD"))); bc.diffuse_obstruction=do
    b=T.build(key); _,v=read_report(fj); _,pa=read_report(fp); V=v[:N,7:][:,0::2]
    s=Solver(b["mj"],z_blend=zb); s.run(bc,nsteps=N); M=np.array([s.series(f"surf_{p}")[:N] for p in POS]).T
    e=(M-V)[mk]; sp=Solver(b["mp"]); sp.run(bc,nsteps=N); ep=(sp.series("surf")[:N]-pa[:N,5])[mk]
    dts=(sp.series("surf")[:N][mk].mean()-M[mk][:,0].mean())-(pa[:N,5][mk].mean()-V[mk][:,0].mean())
    return a,dict(mae=abs(e).mean(),mx=abs(e).max(),b0=e[:,0].mean(),mae0=abs(e[:,0]).mean(),pmae=abs(ep).mean(),pmx=abs(ep).max(),dts=dts)
if __name__=="__main__":
    jobs=[(k,m,c) for k in T.WALLS for m in ("난방","비난방") for c in CFG]
    with ProcessPoolExecutor(4) as ex: res=dict(ex.map(job,jobs))
    for m in ("난방","비난방"):
        print(f"== {m}  (벽체별: 접합부 MAE / 최대 / 0 mm 편향 / ΔTs 오차 | 평면 MAE / 최대)")
        for c in CFG:
            print(f" [{c}]")
            for k in T.WALLS:
                o=res.get((k,m,c))
                if o: print(f"    {k:22s} {o['mae']:.4f} / {o['mx']:.3f} / {o['b0']:+.4f} / {o['dts']:+.4f} | {o['pmae']:.4f} / {o['pmx']:.3f}")
