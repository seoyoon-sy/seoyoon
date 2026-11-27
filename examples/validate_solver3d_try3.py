"""3D 솔버(확산일사 차단 켬)를 TRY3 엑셀 69케이스(난방 39, 비난방 30)로 검증 (CLAUDE.md 8.77).
VOLTRA 엑셀 대비 야간 시간별 12위치 표면온도. 사용: python examples/validate_solver3d_try3.py 20  (z 조각 수)
결과: results/accuracy/solver3d_try3_nz{nz}.csv (케이스별), _pos.csv (위치별)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from multiprocessing import Pool
from hbcalc import Boundary
from hbcalc.solver3d import Solver3D
from hbcalc.validate import *
from hbcalc.presets import try3_voltra_mesh_model, try3_excel_strip_props

R = ROOT + "/"; POS = EXCEL_LABELS; SIMPOS = SIM_POS_VOLTRA_ORDER; ALLP = tuple(sorted(set(POS) | set(SIMPOS)))
CONDS = (("난방", "Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"), ("비난방", "Wall_TI_2.FTE", "Non_heating_Tsi.xlsx"))
NZ = int(sys.argv[1]) if len(sys.argv) > 1 else 20
mk = night_mask(744)


def job(a):
    cond, ti, xlsx, nm = a
    df = pd.read_excel(R + xlsx, sheet_name=nm)
    Vs = np.array([df[f"{p}mm 표면온도(℃)"].values[:744] for p in POS]).T
    bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + ti, R + "Wall_HO.FHT", R + "Wall_HI.FHT", R + "Wall_IG2.FSG", R + "Wall_ID2.FSD",
                                    solar_shift_h=0.0, diffuse_obstruction=True)
    eps, xps = parse_sheet(nm)
    lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    m = try3_voltra_mesh_model(lam, offsets_mm=ALLP)
    rho, cp = try3_excel_strip_props(xps); m = m.with_properties(rho={"XPS": rho}, cp={"XPS": cp})
    s = Solver3D(m, nz=NZ); s.run(bc, store="outputs")
    M = np.array([s.series(f"surf_{p}")[:744] for p in SIMPOS]).T
    d = (M - Vs)[mk]; ab = np.abs(d)
    row = dict(조건=cond, 케이스=nm, MAE=ab.mean(), 최대=ab.max(), 영mm_편향=d[:, 0].mean(), 평면500_편향=d[:, -1].mean(),
               dTs_오차=d[:, -1].mean() - d[:, 0].mean(), 초과005=(ab > 0.05).mean() * 100)
    pos = [dict(조건=cond, 케이스=nm, 위치=p, 편향=d[:, i].mean(), MAE=ab[:, i].mean(), 최대=ab[:, i].max()) for i, p in enumerate(POS)]
    print(cond, nm, round(row["MAE"], 4), round(row["최대"], 3), flush=True)
    return row, pos


if __name__ == "__main__":
    jobs = []
    for cond, ti, xlsx in CONDS:
        _, names = load_cases(R + xlsx); jobs += [(cond, ti, xlsx, nm) for nm in names]
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame([r for r, _ in res]); df.to_csv(R + f"results/accuracy/solver3d_try3_nz{NZ}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([x for _, ps in res for x in ps]).to_csv(R + f"results/accuracy/solver3d_try3_nz{NZ}_pos.csv", index=False, encoding="utf-8-sig")
    for cond in ("난방", "비난방"):
        x = df[df.조건 == cond]
        print(f"{cond} nz={NZ} n={len(x)}: 케이스MAE 평균 {x.MAE.mean():.4f} 최대 {x.MAE.max():.4f} | 점 최대 {x.최대.max():.3f} | 0mm 편향 {x.영mm_편향.min():+.3f}~{x.영mm_편향.max():+.3f} | "
              f"500 편향 {x.평면500_편향.min():+.3f}~{x.평면500_편향.max():+.3f} | ΔTs 평균절대 {x.dTs_오차.abs().mean():.4f} 최대 {x.dTs_오차.abs().max():.4f} | 0.05초과 {x.초과005.mean():.2f}%")
