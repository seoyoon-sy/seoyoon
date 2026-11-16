"""TRY3 손격자 케이스의 시간 변화 오차가 재료 열용량(밀도 x 비열) 값 때문인지 진단 (정상상태는 열용량과 무관해서 정상상태 비교로는 못 가린다).
재료 하나씩 밀도를 0.5배, 2배로 바꿔 야간 시간별 오차(12위치 MAE, 0 mm MAE, 최대)가 어떻게 변하는지 본다. 진단용이며 이 값을 기본값에 넣지 않는다.
사용: python examples/diagnose_material_rhoc.py"""
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
    base = try3_voltra_mesh_model(lam); V = cache[(c, n)]["V_surf"][:, :N]
    names = {m.name: m for m in base.materials.values()}; print(c, n, {k: (v.rho, v.cp) for k, v in names.items()}, flush=True)
    def run(model):
        s = Solver(model); s.run(bcs[c]); M = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)[:, mk]; return abs(e).mean(), abs(e[0]).mean(), abs(e).max(), e[:9].mean()
    r = run(base); print(f"   기준            MAE {r[0]:.4f}  0mm MAE {r[1]:.4f}  최대 {r[2]:.3f}  0~150 mm 평균편차 {r[3]:+.4f}", flush=True)
    for nm, mat in names.items():
        for sc in (0.5, 2.0):
            r = run(base.with_properties(rho={nm: mat.rho * sc})); print(f"   {nm:18s} 밀도 x{sc:3.1f}  MAE {r[0]:.4f}  0mm MAE {r[1]:.4f}  최대 {r[2]:.3f}  0~150 mm 평균편차 {r[3]:+.4f}", flush=True)
