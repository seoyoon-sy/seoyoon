import os, sys
ROOT="/home/user/seoyoon"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/examples")
import numpy as np
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
import test_walls_def as T
N=744; mk=night_mask(N)
R=lambda *p: os.path.join(ROOT,*p)
def bc(ti): return Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE",ti,"Wall_HO.FHT","Wall_HI.FHT","Wall_IG2.FSG","Wall_ID2.FSD")),solar_shift_h=0.0)
L=lambda eps=180:[Layer("concrete",200),Layer("EPS",eps),Layer("0.5B cement block",90),Layer("gypsum",10)]
cases={"A_EPS100":dict(layers=L(100)),"B_EPS250":dict(layers=L(250)),"C_partition250":dict(partition_mm=250),"D_strip20":dict(strip_mm=20,strip_extend_mm=120)}
def cmp(name, model, b, path):
    hdr,a=read_report(path)[0:2] if False else (None,read_report(path)[1])
    s=Solver(model); s.run(b,nsteps=N); f=s.zone_flows_into_out()
    # VOLTRA cols 2-7: color2 (outdoor) in,out ; color1 (left) in,out ; color7 (right) in,out  (index 1..6)
    V={1:(a[:N,1],a[:N,2]),2:(a[:N,3],a[:N,4]),3:(a[:N,5],a[:N,6])}
    print(name)
    for z,lab in ((1,"실외"),(2,"왼쪽실내"),(3,"오른쪽실내")):
        vi,vo=V[z]; mi,mo=f[z][0][:N],f[z][1][:N]
        vn,mn=vi-vo,mi-mo
        print(f"  {lab}: t=0 순 V {vn[0]:8.3f} M {mn[0]:8.3f} ({100*(mn[0]-vn[0])/abs(vn[0]):+.2f}%) | 야간평균 순 V {vn[mk].mean():8.3f} M {mn[mk].mean():8.3f} ({100*(mn[mk].mean()-vn[mk].mean())/abs(vn[mk].mean()):+.2f}%) | 시간별 순 MAE {abs(mn-vn).mean():.3f} W, 상관 {np.corrcoef(vn,mn)[0,1]:.5f}")
for k,kw in cases.items():
    cmp(k, build_tjunction(replace(try3_spec(snap_mm=1),**kw)), bc("Wall_TI_2.FTE"), R("voltra_results",f"{k}.txt"))

print("--- 실외 열류 시간 이동 시험 (A_EPS100)")
k="A_EPS100"; b=bc("Wall_TI_2.FTE"); a=read_report(R("voltra_results",f"{k}.txt"))[1]
s=Solver(build_tjunction(replace(try3_spec(snap_mm=1),**cases[k]))); s.run(b,nsteps=N+1) if False else s.run(b,nsteps=N)
f=s.zone_flows_into_out()[1]; mn=(f[0]-f[1])[:N]; vn=(a[:N,1]-a[:N,2])
t=np.arange(N)
for d in (-0.5,-0.25,0,0.25,0.5):
    m=np.interp(t-d,t,mn); print(f" 이동 {d:+.2f} h: 순 MAE {abs(m[1:-1]-vn[1:-1]).mean():.3f} W  야간평균 오차 {(m[mk]-vn[mk]).mean():+.3f} W")
print(" 낮(일사) 시각 평균 차 vs 야간:", (mn-vn)[~mk].mean(), (mn-vn)[mk].mean())
