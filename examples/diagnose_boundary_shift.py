"""접합부 열교 깊이(평면 - 0 mm)의 시간 지연 진단. 시간별 오차가 깊이의 시간변화율에 비례하는 것(0.5 ~ 1 시간 지연)이 경계값 시각 어긋남인지 시험한다.
실외온도 T_out 만, 실내온도 T_in 만, 둘 다를 delta 시간만큼 앞당겨(T(t + delta)) 야간 시간별 오차(MAE, 최대)를 본다. 일사는 건드리지 않는다.
사용: python examples/diagnose_boundary_shift.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import parse_sheet, SIM_POS_VOLTRA_ORDER, night_mask
R = lambda *p: os.path.join(ROOT, *p)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb")); N = 744; mk = night_mask(N)
bcs = {c: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", t, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD"))) for c, t in (("난방", "Wall_TI.FTE"), ("비난방", "Wall_TI_2.FTE"))}
sh = lambda x, d: np.interp(np.arange(len(x)) + d, np.arange(len(x)), x)
cases = [("난방", "결로 방지 단열재 10"), ("난방", "결로 방지 단열재 0.005 + 단열재 조정 0.024"), ("비난방", "단열재 조정 0.01+결로 방지 단열재 11.2"), ("비난방", "결로 방지 단열재 0.005")]
if __name__ == "__main__":
    for c, n in cases:
        eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
        if xps is not None: lam["XPS"] = xps
        s = Solver(try3_voltra_mesh_model(lam)); V = cache[(c, n)]["V_surf"][:, :N]; b0 = bcs[c]; print(c, n)
        for what in ("T_out", "T_in", "둘 다"):
            for d in (0.0, 0.25, 0.5, 0.75, 1.0):
                kw = {}
                if what in ("T_out", "둘 다"): kw["T_out"] = sh(b0.T_out, d)
                if what in ("T_in", "둘 다"): kw["T_in"] = sh(b0.T_in, d)
                if c == "난방" and what == "T_in": continue
                s.run(replace(b0, **kw)); M = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)[:, mk]
                print(f"   {what:5s} 앞당김 {d:4.2f} h: 야간 MAE {abs(e).mean():.4f}  0 mm MAE {abs(e[0]).mean():.4f}  최대 {abs(e).max():.3f}", flush=True)
