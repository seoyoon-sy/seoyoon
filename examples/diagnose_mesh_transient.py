"""TRY3 비난방 케이스를 손격자(VOLTRA 격자)와 자동 격자(hbcalc)로 풀었을 때 hbcalc 자체의 차이(VOLTRA 결과 없이).
난방 시작 구간(h60 ~ h130)과 야간 전체에서 두 격자의 12위치 표면온도 차이, 그리고 시간 가중 theta(0.5, 1.0)에 따른 차이를 본다.
사용: python examples/diagnose_mesh_transient.py > 결과.txt"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model, try3_voltra_mesh_model
from hbcalc.validate import night_mask
from diagnose_theta import run_theta
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N)
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", "Wall_TI_2.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
def surf(s, out): return np.array([out[:, s.idx[s.m.outputs[f"surf_{p}"][0], s.m.outputs[f"surf_{p}"][1]]] for p in POS])
for lam in ({"EPS": 0.032, "XPS": 0.0001}, {"EPS": 0.032, "XPS": 0.005}, {"EPS": 0.032, "XPS": 0.035}, {"EPS": 0.032, "XPS": 4.3}):
    sh, sa = Solver(try3_voltra_mesh_model(lam)), Solver(try3_model(lam, snap_mm=1)); print("XPS", lam["XPS"], f"노드 수 손격자 {sh.N}, 자동 {sa.N}", flush=True)
    res = {}
    for th in (0.5, 1.0):
        res[("h", th)] = surf(sh, run_theta(sh, bc, N, th)); res[("a", th)] = surf(sa, run_theta(sa, bc, N, th))
    for th in (0.5, 1.0):
        d = res[("h", th)] - res[("a", th)]; w = d[:, 60:131]
        print(f"  theta {th}: 손격자 - 자동 격자 | h60~130 0~50 mm 평균 {w[:6].mean():+.3f} 최대절대 {abs(w[:6]).max():.3f} | 야간 12위치 MAE {abs(d[:, mk]).mean():.4f} 최대 {abs(d[:, mk]).max():.3f}", flush=True)
    for g in ("h", "a"):
        d = res[(g, 1.0)] - res[(g, 0.5)]; print(f"  {'손격자' if g == 'h' else '자동격자'}: theta 1.0 - 0.5 | h67 0 mm {d[0, 67]:+.3f} h60~130 0~50 mm 최대절대 {abs(d[:6, 60:131]).max():.3f} 야간 MAE {abs(d[:, mk]).mean():.4f}", flush=True)
