"""N6 원인 진단 (시험 벽체 6종 VOLTRA 정상상태 t=0 비교). 사용: python examples/diagnose_test_walls.py [grid]
전제: voltra_results/test_walls/_cmp.pkl (python examples/compare_test_wall_results.py 로 생성). _strip.py: 띠 면 대류 배분, 방사율 시험, _iter.py: 반복법 중단 시험."""
import sys, pickle, itertools
import os; ROOT_=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT_); sys.path.insert(0,ROOT_+'/examples')
import numpy as np
from hbcalc import Boundary, Solver
import test_walls_def as T
R=ROOT_+'/'
res=pickle.load(open(R+"voltra_results/test_walls/_cmp.pkl","rb"))
bc=Boundary.from_voltra_files(*(R+f for f in ("wall_TO.FTE","Wall_TI.FTE","Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD")),solar_shift_h=0.0)
POS=(0,5,10,20,30,50,75,100,150,200,300,500)
B={k:T.build(k) for k in res}
def dev(key,**kw):
    rad=kw.pop("rad",1.0); hs=kw.pop("hs",1.0)
    s=Solver(B[key]["mj"],**kw)
    if rad!=1.0: s.set_exchange_areas(s.rad_W*rad)
    s.run(bc,nsteps=1)
    M=np.array([s.series(f"surf_{p}")[0] for p in POS])
    return M-res[key]["junction"]["Vs"][0]
if __name__=="__main__":
    cfgs=[("low",dict(conv_lump="low")),("half",dict(conv_lump="half")),("(.5,1)",dict(conv_lump=(0.5,1.0))),("(1,.5)",dict(conv_lump=(1.0,0.5))),("(.75,.75)",dict(conv_lump=(0.75,0.75))),
          ("low avg",dict(conv_lump="low",zmode="avg")),("low inf",dict(conv_lump="low",zmode="inf")),("low rad.9",dict(conv_lump="low",rad=0.9)),("low rad1.1",dict(conv_lump="low",rad=1.1))]
    print("t=0 편차 0~150 mm 평균 / RMSE(12위치): 벽체별")
    for lab,kw in cfgs:
        out=[]
        for k in res:
            d=dev(k,**dict(kw)); out.append(f"{k[:2]} {d[:9].mean():+.3f}/{np.sqrt((d**2).mean()):.3f}")
        print(f"{lab:11s}"," | ".join(out))

def grid():
    ws=[0,0.25,0.5,0.75,1.0]; keys=list(res)
    print("wx(가로면=벽 표면) wy(세로면=내벽·띠 표면): 벽체별 RMSE(12위치) 0~150 mm 평균 편차")
    best=[]
    for wx in ws:
        for wy in ws:
            ds=[dev(k,conv_lump=(wx,wy)) for k in keys]
            rm=[np.sqrt((d**2).mean()) for d in ds]; mb=[d[:9].mean() for d in ds]
            best.append((max(rm),wx,wy,rm,mb))
    for mx,wx,wy,rm,mb in sorted(best)[:8]:
        print(f"wx {wx:.2f} wy {wy:.2f} 최대RMSE {mx:.3f} | RMSE "+" ".join(f"{r:.3f}" for r in rm)+" | 편차 "+" ".join(f"{m:+.3f}" for m in mb))
if len(sys.argv)>1 and sys.argv[1]=="grid": grid()
