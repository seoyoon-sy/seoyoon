"""3D 솔버(확산일사 차단 켬, z 50 mm x 20)를 추가 벽체로 검증 (CLAUDE.md 8.78). 비교 기준은 2D 현재 기본값(차단 끔).
대상: N1 ~ N6 접합부와 평면(난방, 비난방), 변형 A ~ D 접합부(비난방), 독립 T1 ~ T8 접합부(난방 8, 비난방 T1 ~ T4).
VOLTRA Report Output 대비 야간 시간별 12위치 표면온도. 결과: results/accuracy/solver3d_extra.csv"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/examples")
import numpy as np, pandas as pd
from dataclasses import replace
from multiprocessing import Pool
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.solver3d import Solver3D
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
import test_walls_def as TW

R = lambda *p: os.path.join(ROOT, *p)
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); N = 744; mk = night_mask(N)
TI = {"난방": "Wall_TI.FTE", "비난방": "Wall_TI_2.FTE"}
Lv = lambda eps=180: [Layer("concrete", 200), Layer("EPS", eps), Layer("0.5B cement block", 90), Layer("gypsum", 10)]
VAR = {"A_EPS100": dict(layers=Lv(100)), "B_EPS250": dict(layers=Lv(250)), "C_partition250": dict(partition_mm=250), "D_strip20": dict(strip_mm=20, strip_extend_mm=120)}


def bc(cond, obs):
    return Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", TI[cond], "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0, diffuse_obstruction=obs)


def model(group, key, kind):
    if group == "N":
        b = TW.build(key); return b["mj"] if kind == "junction" else b["mp"]
    if group == "변형":
        return build_tjunction(replace(try3_spec(snap_mm=1), **VAR[key]))
    import eval_independent_test as EI
    return EI.model_of(key)[0]


def report(group, key, cond, kind):
    if group == "N":
        return R("voltra_results", "test_walls" if cond == "난방" else "test_walls_nonheating", f"{key}_{kind}.txt")
    if group == "변형":
        return R("voltra_results", f"{key}.txt")
    return R("voltra_results", "independent_test", f"{key}_{'heating' if cond == '난방' else 'nonheating'}.txt")


def job(a):
    group, key, cond, kind = a
    A = read_report(report(group, key, cond, kind))[1][:N]
    m = model(group, key, kind)
    names = ["surf"] if kind == "planar" else [f"surf_{p}" for p in POS]
    V = A[:, 5:6] if kind == "planar" else A[:, 7:][:, 0::2]
    out = {}
    s2 = Solver(m); s2.run(bc(cond, False), nsteps=N); out["2D"] = np.array([s2.series(n)[:N] for n in names]).T
    s3 = Solver3D(m, nz=20); s3.run(bc(cond, True), store="outputs"); out["3D"] = np.array([s3.series(n)[:N] for n in names]).T
    print(group, key, cond, kind, flush=True)
    return (group, key, cond, kind), V, out


if __name__ == "__main__":
    jobs = [("N", k, c, kd) for k in TW.WALLS for c in TI for kd in ("junction", "planar")]
    jobs += [("변형", k, "비난방", "junction") for k in VAR]
    jobs += [("독립", f"T{i}", c, "junction") for i in range(1, 9) for c in TI if c == "난방" or i <= 4]
    jobs = [j for j in jobs if os.path.exists(report(*j))]
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=1)
    rows = []
    D = {k: (V, out) for k, V, out in res}
    for (g, key, cond, kind), (V, out) in D.items():
        for sc, M in out.items():
            d = (M - V)[mk]; ab = np.abs(d)
            r = dict(그룹=g, 벽체=key, 조건=cond, 종류=kind, 설정=sc, MAE=ab.mean(), 최대=ab.max(), 영mm_편향=d[:, 0].mean(), 초과005=(ab > 0.05).mean() * 100)
            if kind == "junction":
                r["평면500_편향"] = d[:, -1].mean()
                P = D.get((g, key, cond, "planar"))
                if P is not None:     # ΔTs = 평면 벽체 - 접합부 0 mm
                    r["dTs_오차"] = (P[1][sc][mk, 0].mean() - M[mk, 0].mean()) - (P[0][mk, 0].mean() - V[mk, 0].mean())
                elif g == "독립":
                    r["dTs_오차"] = np.nan
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(R("results", "accuracy", "solver3d_extra.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
