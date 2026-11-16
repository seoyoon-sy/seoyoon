"""결로 방지재를 조절하지 않은 케이스(이름에 결로 방지 단열재가 없거나 1.6)는 판 물성이 콘크리트와 같다(사용자 확인 2026-11-16): 열전도율 1.6, 밀도 2200, 비열 795.
이 케이스들을 (a) 판 밀도 25, 비열 1300, (b) 콘크리트 물성(2200, 795)으로 계산해 오차를 비교한다.
사용: python examples/diagnose_default_xps_cases2.py"""
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
for c, n in sel:
    eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    V = cache[(c, n)]["V_surf"][:, :N]; out = []
    for tag, rho, cp in (("판 25/1300", 25.0, 1300.0), ("콘크리트 2200/795", 2200.0, 795.0)):
        s = Solver(try3_voltra_mesh_model(lam).with_properties(rho={"XPS": rho}, cp={"XPS": cp})); s.run(bcs[c]); M = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)[:, mk]
        out.append(f"{tag}: MAE {abs(e).mean():.4f} 0mm편차 {e[0].mean():+.4f} 최대 {abs(e).max():.3f} 0.05초과 {100 * (abs(e) > 0.05).mean():.2f} %")
    print(f"{c} {n[:30]:30s} | " + " | ".join(out), flush=True)
