"""VOLTRA .sol(정상상태 0000) 온도장 대 hbcalc 변형들: 대류 배분(half, low, 규칙 w)과 형상계수(해석 mid, VOLTRA .vwf) 조합 (CLAUDE.md 8.51).
사용: python examples/compare_sol_variants.py 접두어 EPS XPS"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.lump_rule import strip_weight
from hbcalc.voltra_vwf import read_vwf
from hbcalc.vwf_apply import exchange_areas_from_vwf
pre, E_, X_ = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
V = {}
for l in open(os.path.join(D, pre + "0000.sol"), "rb").read().decode("cp1252").replace("\r", "").split("\n"):
    r = l.split()
    if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); cid = m.cid; nx, ny = cid.shape
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
vwf = read_vwf(os.path.join(D, pre + ".vwf")); xs = np.round(m.xe * 1000, 3); yi = int(np.argmin(abs(np.round(m.ye * 1000, 3) - 1480)))
w = strip_weight(X_, "auto"); POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
cfgs = [("half", "half", False), ("half + VOLTRA 형상계수", "half", True), ("low", "low", False), (f"w={w:.3f}", (w, w), False), (f"w={w:.3f} + VOLTRA 형상계수", (w, w), True)]
print(f"{pre}0000 (정상상태 t=0), 격자 노드 z 중간 평면 {len([k for k in V if k[2] == 10])}개")
for nm, cl, use in cfgs:
    s = Solver(m, conv_lump=cl)
    if use: s.set_exchange_areas(exchange_areas_from_vwf(s, vwf))
    s.run(bc, nsteps=1); T0 = s.T_all[0]
    keys = [k for k in V if k[2] == 10 and s.idx[k[0], k[1]] >= 0]; d = np.array([T0[s.idx[k[0], k[1]]] - V[k] for k in keys])
    right = [T0[s.idx[m.outputs[f"surf_{p}"][0], m.outputs[f"surf_{p}"][1]]] - V[(m.outputs[f"surf_{p}"][0], m.outputs[f"surf_{p}"][1], 10)] for p in POS]
    left = []
    for p in (0, 50, 100, 200, 500):
        i = int(np.argmin(abs(xs - (1000 - p)))); left.append(T0[s.idx[i, yi]] - V[(i, yi, 10)])
    print(f"  [{nm}] 전체 RMSE {np.sqrt((d ** 2).mean()):.4f} 평균 {d.mean():+.4f} | 오른쪽 벽 0, 5, 10, 20, 30, 50, 100, 500 mm: "
          + " ".join(f"{right[POS.index(p)]:+.3f}" for p in (0, 5, 10, 20, 30, 50, 100, 500)) + f" (12위치 RMSE {np.sqrt(np.mean(np.square(right))):.3f}) | 왼쪽 벽 x=1000, 950, 900, 800, 500: " + " ".join(f"{v:+.3f}" for v in left), flush=True)
