"""XPS 판 밀도를 25로 정정한 뒤(8.56) 다른 재료(콘크리트, EPS, 블록, 석고)의 밀도, 비열이 사용자의 TRY3 VOLTRA 입력과 맞는지 재진단.
재료 하나씩 밀도 또는 비열을 0.7배, 1.4배로 바꿔 야간 시간별 오차(12위치 MAE, 0 mm 평균 편차, 최대)가 줄면 그 재료의 입력값이 다를 가능성이 있다. 진단용이며 기본값에 넣지 않는다.
사용: python examples/diagnose_material_rhoc2.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model, TRY3_EXCEL_XPS_RHO
from hbcalc.validate import parse_sheet, SIM_POS_VOLTRA_ORDER, night_mask
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb"))
bcs = {c: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", t, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD"))) for c, t in (("난방", "Wall_TI.FTE"), ("비난방", "Wall_TI_2.FTE"))}
cases = [("난방", "결로 방지 단열재 10"), ("난방", "결로 방지 단열재 0.005 + 단열재 조정 0.024"), ("비난방", "결로 방지 단열재 1.6"), ("비난방", "단열재 조정 0.01+결로 방지 단열재 11.2")]
for c, n in cases:
    eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    base = try3_voltra_mesh_model(lam).with_properties(rho={"XPS": TRY3_EXCEL_XPS_RHO}); V = cache[(c, n)]["V_surf"][:, :N]
    names = {m.name: m for m in base.materials.values()}; print(c, n, {k: (v.rho, v.cp) for k, v in names.items()}, flush=True)
    def run(model):
        s = Solver(model); s.run(bcs[c]); M = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)[:, mk]; return abs(e).mean(), e[0].mean(), abs(e).max()
    r = run(base); print(f"   기준                  MAE {r[0]:.4f}  0mm 편차 {r[1]:+.4f}  최대 {r[2]:.3f}", flush=True)
    for nm, mat in names.items():
        if nm == "XPS": continue
        for what in ("rho", "cp"):
            for sc in (0.7, 1.4):
                kw = {what: {nm: getattr(mat, what) * sc}}; r = run(base.with_properties(**kw)); print(f"   {nm:18s} {what:3s} x{sc:3.1f}  MAE {r[0]:.4f}  0mm 편차 {r[1]:+.4f}  최대 {r[2]:.3f}", flush=True)
