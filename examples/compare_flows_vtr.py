"""VOLTRA 리포트의 색별 열류(들어옴, 나감)와 hbcalc 비교. 사용: python examples/compare_flows_vtr.py 파일.vtr 리포트.txt [실내온도파일]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.voltra_report import read_report
vtr, rep = sys.argv[1], sys.argv[2]; TI = sys.argv[3] if len(sys.argv) > 3 else "Wall_TI.FTE"
m = model_from_vtr(vtr); cols, V = read_report(rep); n = V.shape[0]
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", TI, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
names = [c for c in cols if "Colour" in c]; zs = [int(c.split("Colour")[1].split(",")[0]) for c in names][::2]
outd = [z for z, v in m.zones.items() if v.kind == "outdoor"][0]
vn = {}
for k, z in enumerate(zs):
    vin, vout = V[:, 1 + 2 * k], V[:, 2 + 2 * k]
    vn[z] = (vin, vout)
hours = np.arange(n); night = np.zeros(n, bool)
for d in range(30): night[18 + 24 * d: 18 + 24 * d + 14] = True
night &= hours < n - 1
print(f"{vtr}  vs  {rep}  (시각 {n}개)")
for cl in ("low", "half"):
    s = Solver(m, conv_lump=cl); s.run(bc, nsteps=n); F = s.zone_flows_into_out()
    print(f"\n== hbcalc conv_lump={cl}")
    for z in zs:
        vin, vout = vn[z]; hin, hout = F[z][0][:n], F[z][1][:n]
        sign = -1 if z == outd else 1                      # 순 열류: 실외는 유출(out - in), 실내는 유입(in - out)
        vnet = (vout - vin) if z == outd else (vin - vout); hnet = (hout - hin) if z == outd else (hin - hout)
        e = hnet - vnet
        print(f"  색 {z:3d} ({m.zones[z].kind:7s}) t=0 순 열류 VOLTRA {vnet[0]:7.3f} hbcalc {hnet[0]:7.3f} ({(hnet[0]/vnet[0]-1)*100:+.2f} %) | 전체 시간 평균 VOLTRA {vnet.mean():7.3f} hbcalc {hnet.mean():7.3f} ({(hnet.mean()/vnet.mean()-1)*100:+.2f} %) | 야간 평균 오차 {e[night].mean():+.3f} W RMSE {np.sqrt((e[night]**2).mean()):.3f} W | 나감(실내) VOLTRA {vout.mean():.2f} 대 hbcalc {hout.mean():.2f} W" if z != outd else
              f"  색 {z:3d} ({m.zones[z].kind:7s}) t=0 순 유출 VOLTRA {vnet[0]:7.3f} hbcalc {hnet[0]:7.3f} ({(hnet[0]/vnet[0]-1)*100:+.2f} %) | 전체 시간 평균 VOLTRA {vnet.mean():7.3f} hbcalc {hnet.mean():7.3f} ({(hnet.mean()/vnet.mean()-1)*100:+.2f} %) | 야간 평균 오차 {e[night].mean():+.3f} W RMSE {np.sqrt((e[night]**2).mean()):.3f} W")
    rin = sum((F[z][0][:n] - F[z][1][:n]) for z in zs if z != outd); vin_ = sum((vn[z][0] - vn[z][1]) for z in zs if z != outd)
    print(f"  실내 두 방 합: 야간 평균 VOLTRA {vin_[night].mean():.3f} hbcalc {rin[night].mean():.3f} W ({(rin[night].mean()/vin_[night].mean()-1)*100:+.2f} %) | 좌우비 t=0 VOLTRA {(vn[zs[1]][0][0]-vn[zs[1]][1][0])/(vn[zs[2]][0][0]-vn[zs[2]][1][0]):.4f} hbcalc {(F[zs[1]][0][0]-F[zs[1]][1][0])/(F[zs[2]][0][0]-F[zs[2]][1][0]):.4f}")
