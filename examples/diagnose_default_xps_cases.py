"""XPS 열전도율을 바꾸지 않은 기본(1.6) 케이스들의 XPS 판 밀도가 25인지 2200(템플릿)인지 진단. 비난방에서 이 케이스들만 오차가 2배(8.56 이후 관찰).
각 케이스를 XPS 밀도 25와 2200으로 계산해 야간 시간별 오차를 비교한다. 진단용이며 기본값에 넣지 않는다.
사용: python examples/diagnose_default_xps_cases.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import parse_sheet, SIM_POS_VOLTRA_ORDER, night_mask
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb"))
bcs = {c: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", t, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD"))) for c, t in (("난방", "Wall_TI.FTE"), ("비난방", "Wall_TI_2.FTE"))}
sel = [k for k in cache if parse_sheet(k[1])[1] is None or abs((parse_sheet(k[1])[1] or 0) - 1.6) < 1e-9]
print("대상 케이스:", [(c, n[:30]) for c, n in sel], flush=True)
for c, n in sel:
    eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    V = cache[(c, n)]["V_surf"][:, :N]; out = []
    for rho in (25.0, 2200.0):
        s = Solver(try3_voltra_mesh_model(lam).with_properties(rho={"XPS": rho})); s.run(bcs[c]); M = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)[:, mk]
        out.append(f"밀도 {rho:6.0f}: MAE {abs(e).mean():.4f} 0mm MAE {abs(e[0]).mean():.4f} 0mm 편차 {e[0].mean():+.4f} 최대 {abs(e).max():.3f} 0.05초과 {100 * (abs(e) > 0.05).mean():.2f} %")
    print(f"{c} {n[:34]:34s} | " + " | ".join(out), flush=True)
