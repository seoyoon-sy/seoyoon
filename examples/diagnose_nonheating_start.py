"""비난방 케이스에서 난방 시작 구간(1/3 18시경, 67시간째 이후)의 큰 시간별 오차 원인 진단.
실내온도를 시간 안에서 어떻게 다루는지(시각 이동, 계단) 바꿔 보며 VOLTRA 대비 오차 변화를 본다. 입력: results/_abs_cache.pkl
사용: python examples/diagnose_nonheating_start.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import parse_sheet, EXCEL_LABELS, SIM_POS_VOLTRA_ORDER
R = lambda *p: os.path.join(ROOT, *p)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb")); NH = 200
bc0 = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", "Wall_TI_2.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
names = [n for (c, n) in cache if c == "비난방"]
pick = ["단열재 조정 0.01+결로 방지 단열재 11.2", "결로 방지 단열재 0.0001", "결로 방지 단열재 0.005", "결로 방지 단열재 0.035", "결로 방지 단열재 1.6", "단열재 조정 나등급(0.04)"]
def shifted(x, d):                       # x(t + d) 선형보간 (d 시간)
    t = np.arange(len(x)); return np.interp(t + d, t, x)
def variant(kind):
    Ti = bc0.T_in.copy()
    if kind == "base": pass
    elif kind.startswith("shift"): Ti = shifted(Ti, float(kind[5:]))
    elif kind == "step_prev": Ti = np.r_[Ti[0], Ti[:-1]]
    return replace(bc0, T_in=Ti)
kinds = ["base", "shift+0.25", "shift+0.4", "shift+0.5", "shift+0.6", "shift+0.75"]
print("케이스 x 변형: 0~150 mm 위치 평균 시간별 오차(모사 - VOLTRA)의 구간 평균 / 최대 절대, 구간 h66~h130")
for n in pick:
    eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    s = Solver(try3_voltra_mesh_model(lam)); V = cache[("비난방", n)]["V_surf"][:, :NH]
    print(n)
    for k in kinds:
        s.run(variant(k), nsteps=NH); M = np.array([s.series(f"surf_{p}")[:NH] for p in SIM_POS_VOLTRA_ORDER])
        e = (M - V)[:9, 66:131]; print(f"   {k:10s} 평균 {e.mean():+.3f}  평균절대 {abs(e).mean():.3f}  최대절대 {abs(e).max():.3f}  h67 0mm {(M-V)[0,67]:+.3f}  h90 0mm {(M-V)[0,90]:+.3f}", flush=True)
