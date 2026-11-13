"""새 대류 배분 규칙(auto)이 억지 맞춤이 아닌지 보는 두 가지 점검 (독립 테스트 전).
1) 열류: calib_steady 14개 VOLTRA 정상상태 열류(실외 순 유출, 왼쪽, 오른쪽 실내 순 유입, 좌우비)와 hbcalc half, low, 새 auto 비교.
   표면온도만 맞추려고 w 를 만지면 열류와 좌우 대칭이 깨질 수 있다(8.19, 8.28에서 low 는 좌우비 0.99로 깨짐).
2) w 민감도: 케이스별 규칙 예측 w 에서 +-0.05, +-0.1 벗어났을 때 12위치 표면온도 RMSE(정상상태)가 얼마나 변하는지.
사용: python examples/check_w_flows_sensitivity.py"""
import os, sys, re, glob, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.voltra_report import read_report
from hbcalc.lump_rule import strip_weight
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
D = pickle.load(open(os.path.join(ROOT, "results", "accuracy", "calib_w_curves.pkl"), "rb")); WG = np.round(np.arange(0.2, 1.0001, 0.025), 4)
rm = lambda d: np.sqrt((d ** 2).mean())
def at(c, w):
    i = min(max(np.searchsorted(WG, w), 1), len(WG) - 1); a = (w - WG[i - 1]) / (WG[i] - WG[i - 1]); return (1 - a) * c["dev"][i - 1] + a * c["dev"][i]
print("1) 정상상태 열류 W: 실외 순유출 / 왼쪽 순유입 / 오른쪽 순유입 / 좌우비(왼/오른).  VOLTRA 대 hbcalc 오차 %")
agg = {k: [] for k in ("half", "low", "auto")}
for f in sorted(glob.glob(os.path.join(ROOT, "voltra_results", "calib_steady", "C_E*.txt"))):
    name = os.path.basename(f)[:-4]; e, x = map(float, re.match(r"C_E([\d.]+)_X([\d.]+)", name).groups())
    a = read_report(f)[1][0]; Vo, Vl, Vr = a[2] - a[1], a[3] - a[4], a[5] - a[6]
    m = try3_model({"EPS": e, "XPS": x}, snap_mm=1); line = f"  {name}: VOLTRA {Vo:.2f} / {Vl:.2f} / {Vr:.2f} / {Vl / Vr:.4f} |"
    for cl in ("half", "low", "auto"):
        s = Solver(m, conv_lump=cl); s.run(bc, nsteps=1); F = s.zone_flows_into_out()
        o = [z for z in F if s.m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in F if s.m.zones[z].kind == "indoor")
        Mo, Ml, Mr = F[o][1][0] - F[o][0][0], F[ind[0]][0][0] - F[ind[0]][1][0], F[ind[1]][0][0] - F[ind[1]][1][0]
        agg[cl].append((100 * (Mo / Vo - 1), 100 * (Ml / Vl - 1), 100 * (Mr / Vr - 1), Ml / Mr)); line += f" {cl} {100 * (Mo / Vo - 1):+.2f}% 좌우비 {Ml / Mr:.4f} |"
    print(line, flush=True)
for cl in agg:
    A = np.array(agg[cl]); print(f"  [{cl}] 실외 순유출 오차 평균 {A[:, 0].mean():+.2f} % (절대 최대 {abs(A[:, 0]).max():.2f}), 왼쪽 {A[:, 1].mean():+.2f} %, 오른쪽 {A[:, 2].mean():+.2f} %, 좌우비 평균 {A[:, 3].mean():.4f} (범위 {A[:, 3].min():.4f} ~ {A[:, 3].max():.4f})")
print("\n2) w 민감도: 규칙 예측 w 에서 벗어난 만큼 12위치 표면온도 RMSE(정상상태 14개 평균, 최대)")
for dw in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
    r = [rm(at(c, min(max(strip_weight(c["x"], "auto") + dw, 0.2), 1.0))) for c in D.values()]
    z = [abs(at(c, min(max(strip_weight(c["x"], "auto") + dw, 0.2), 1.0))[0]) for c in D.values()]
    print(f"  w {dw:+.2f}: RMSE 평균 {np.mean(r):.4f} 최대 {np.max(r):.3f} | 0 mm 평균절대 {np.mean(z):.4f}")
r_half = [rm(at(c, 0.5)) for c in D.values()]; r_low = [rm(at(c, 1.0)) for c in D.values()]; print(f"  참고: half(w 0.5) RMSE 평균 {np.mean(r_half):.4f}, low(w 1.0) {np.mean(r_low):.4f}")
