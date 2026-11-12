"""VOLTRA 리포트(출력 노드 온도 + 색별 열류)와 hbcalc 비교. 출력 노드 위치는 .vtr 의 onod 를 그대로 읽는다.
사용: python examples/compare_vtr_outputs.py 파일.vtr 리포트.txt [실내온도파일]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.voltra_report import read_report
vtr, rep = sys.argv[1], sys.argv[2]; TI = sys.argv[3] if len(sys.argv) > 3 else "Wall_TI.FTE"
raw = open(vtr, "rb").read()
try: text = raw.decode("utf-8")
except UnicodeDecodeError: text = raw.decode("cp949", "replace")
L = text.replace("\r", "").split("\n")
onod = [tuple(int(v) for v in l.split("=")[1].split()) for l in L if l.startswith("onod=")]
m = model_from_vtr(vtr); cols, V = read_report(rep); n = V.shape[0]
nflow = len([c for c in cols if "Colour" in c]); noblk = len([c for c in cols if "Output block" in c])
T0 = 1 + nflow + noblk; Vt = V[:, T0:T0 + len(onod)]
print(f"{vtr}: 격자 {m.shape}, 출력 노드 {len(onod)}개, 리포트 {n}시각, 열 {len(cols)}개")
for k, o in enumerate(onod):
    print(f"  노드 {k+1}: 격자선 (x {o[1]}, y {o[2]}, z {o[3]}) = ({m.xe[o[1]]*1000:.0f}, {m.ye[o[2]]*1000:.0f}) mm  재료/경계 셀(위 아래): {[int(m.cid[min(o[1], m.cid.shape[0]-1), j]) for j in (o[2]-1, o[2])]}")
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", TI, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
night = np.zeros(n, bool)
for d in range(30): night[18 + 24 * d: 18 + 24 * d + 14] = True
night &= np.arange(n) < n - 1
outd = [z for z, v in m.zones.items() if v.kind == "outdoor"][0]
fz = [int(c.split("Colour")[1].split(",")[0]) for c in cols if "Colour" in c][::2]
for cl in ("low", "half"):
    s = Solver(m, conv_lump=cl); s.run(bc, nsteps=n); F = s.zone_flows_into_out()
    print(f"\n== hbcalc conv_lump={cl}")
    for k, o in enumerate(onod):
        h = s.node_series(o[1], o[2])[:n]; e = h - Vt[:, k]
        print(f"  노드 {k+1} (x {o[1]}): 야간 평균 VOLTRA {Vt[night, k].mean():.3f} hbcalc {h[night].mean():.3f} | 편차 평균 {e[night].mean():+.3f} 평균 절대 {np.abs(e[night]).mean():.3f} RMSE {np.sqrt((e[night]**2).mean()):.3f} 최대 {np.abs(e[night]).max():.3f} | t=0 편차 {e[0]:+.3f} | 상관 {np.corrcoef(h,Vt[:,k])[0,1]:.5f}")
    for j, z in enumerate(fz):
        vin, vout = V[:, 1 + 2 * j], V[:, 2 + 2 * j]; hin, hout = F[z][0][:n], F[z][1][:n]
        vnet = (vout - vin) if z == outd else (vin - vout); hnet = (hout - hin) if z == outd else (hin - hout)
        print(f"  색 {z:3d} 순 열류 t=0 VOLTRA {vnet[0]:.3f} hbcalc {hnet[0]:.3f} ({(hnet[0]/vnet[0]-1)*100:+.2f} %) | 야간 평균 {vnet[night].mean():.3f} 대 {hnet[night].mean():.3f} ({(hnet[night].mean()/vnet[night].mean()-1)*100:+.2f} %)")
