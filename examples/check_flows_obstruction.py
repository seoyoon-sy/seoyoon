"""실외 열류 야간 오차(8.58)에 확산일사 장애물 차단(diffuse_obstruction)이 미치는 영향 (CLAUDE.md 8.73).
사용: python examples/check_flows_obstruction.py"""
import os, sys
ROOT="/home/user/seoyoon"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/examples")
import numpy as np
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
N=744; mk=night_mask(N); R=lambda *p: os.path.join(ROOT,*p)
def bc(obs):
    b=Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE","Wall_TI_2.FTE","Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD")),solar_shift_h=0.0)
    b.diffuse_obstruction=obs; return b
L=lambda eps=180:[Layer("concrete",200),Layer("EPS",eps),Layer("0.5B cement block",90),Layer("gypsum",10)]
cases={"A_EPS100":dict(layers=L(100)),"B_EPS250":dict(layers=L(250)),"C_partition250":dict(partition_mm=250),"D_strip20":dict(strip_mm=20,strip_extend_mm=120)}
for k,kw in cases.items():
    a=read_report(R("voltra_results",f"{k}.txt"))[1]; V={1:(a[:N,1],a[:N,2]),2:(a[:N,3],a[:N,4]),3:(a[:N,5],a[:N,6])}
    row=[]
    for obs in (False,True):
        s=Solver(build_tjunction(replace(try3_spec(snap_mm=1),**kw))); s.run(bc(obs),nsteps=N); f=s.zone_flows_into_out()
        res=[]
        for z in (1,2,3):
            vn=V[z][0]-V[z][1]; mn=(f[z][0]-f[z][1])[:N]
            e=mn-vn; vm=abs(vn[mk].mean())
            res.append((100*e[mk].mean()/vm, abs(e[mk]).mean(), abs(e[mk]).max(), 100*abs(e[mk]).mean()/vm, abs(e).mean(), abs(e).max()))
        row.append(res)

    for obs,res in zip(("끔","켬"),row):
        for z,nm in ((2,"왼쪽"),(3,"오른쪽"),(1,"실외")):
            r=res[z-1]; print(f"{k} 차단{obs} {nm}: 야간평균 {r[0]:+.2f}% | 야간 시간별 MAE {r[1]:.3f} W ({r[3]:.2f}%) 최대 {r[2]:.3f} W | 전체시간 MAE {r[4]:.3f} 최대 {r[5]:.3f} W",flush=True)
