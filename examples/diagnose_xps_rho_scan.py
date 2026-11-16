"""TRY3 손격자 케이스에서 XPS 판 밀도를 바꿔 가며 오차가 가장 작은 지점을 찾는 진단 (diagnose_material_rhoc.py 후속).
진단용이며 이 값을 기본값에 넣지 않는다. 실제 값은 사용자의 VOLTRA 입력 파일에서 확인해야 한다.
사용: python examples/diagnose_xps_rho_scan.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import parse_sheet, SIM_POS_VOLTRA_ORDER, night_mask
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb"))
bcs = {c: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", t, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD"))) for c, t in (("난방", "Wall_TI.FTE"), ("비난방", "Wall_TI_2.FTE"))}
cases = [("난방", "결로 방지 단열재 10"), ("난방", "결로 방지 단열재 0.005 + 단열재 조정 0.024"), ("비난방", "결로 방지 단열재 0.005"), ("비난방", "단열재 조정 0.01+결로 방지 단열재 11.2")]
for c, n in cases:
    eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    base = try3_voltra_mesh_model(lam); V = cache[(c, n)]["V_surf"][:, :N]; print(c, n, flush=True)
    for rho in (2200, 1100, 550, 275, 100, 40):
        s = Solver(base.with_properties(rho={"XPS": rho})); s.run(bcs[c]); M = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)[:, mk]
        print(f"   XPS 밀도 {rho:5d}  MAE {abs(e).mean():.4f}  0mm MAE {abs(e[0]).mean():.4f}  최대 {abs(e).max():.3f}  0~150 mm 평균편차 {e[:9].mean():+.4f}", flush=True)
