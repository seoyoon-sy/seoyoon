"""TRY3 엑셀 케이스(난방 39, 비난방 30)에서 확산일사 장애물 차단 끔 대 켬 비교 (8.74).
VOLTRA 엑셀 대비 야간 시간별 12위치 표면온도(모사 - VOLTRA), 케이스별로 낸다. 평균만 보고하지 않고 최대를 함께 적는다.
결과: results/accuracy/diffuse_obstruction_try3.csv (케이스별), diffuse_obstruction_try3_pos.csv (위치별)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from multiprocessing import Pool
from hbcalc import Boundary
from hbcalc.validate import *

R = ROOT + "/"; OUT = R + "results/accuracy/"
POS = EXCEL_LABELS; SIMPOS = SIM_POS_VOLTRA_ORDER
CONDS = (("난방", "Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"),
         ("비난방", "Wall_TI_2.FTE", "Non_heating_Tsi.xlsx"))


def job(a):
    cond, ti, xlsx, nm, flag = a
    bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + ti, R + "Wall_HO.FHT", R + "Wall_HI.FHT",
                                    R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0, diffuse_obstruction=flag)
    s, m = run_case(nm, bc, mesh="voltra", positions=sorted(set(POS) | set(SIMPOS)))
    df = pd.read_excel(R + xlsx, sheet_name=nm)
    ss = {p: s.series(f"surf_{p}") for p in set(POS) | set(SIMPOS)}
    bb = {p: s.series(f"back_{p}") for p in set(POS) | set(SIMPOS)}
    d, dw = compare_case(df, ss, bb, POS, SIMPOS)
    return (cond, nm, flag, d)


if __name__ == "__main__":
    jobs = []
    for cond, ti, xlsx in CONDS:
        _, names = load_cases(R + xlsx)
        jobs += [(cond, ti, xlsx, nm, f) for nm in names for f in (False, True)]
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=1)
    rows, prow = [], []
    for cond, nm, flag, d in res:
        a = np.abs(d)
        dts = d[:, POS.index(500)].mean() - d[:, POS.index(0)].mean()
        rows.append(dict(조건=cond, 케이스=nm, 차단=flag, MAE=a.mean(), 최대=a.max(), 영mm_MAE=a[:, 0].mean(), 영mm_편향=d[:, 0].mean(),
                         평면500_편향=d[:, POS.index(500)].mean(), dTs_오차=dts, 초과005=(a > 0.05).mean() * 100))
        for i, pz in enumerate(POS):
            prow.append(dict(조건=cond, 케이스=nm, 차단=flag, 위치=pz, 편향=d[:, i].mean(), MAE=a[:, i].mean(), 최대=a[:, i].max()))
    df = pd.DataFrame(rows); df.to_csv(OUT + "diffuse_obstruction_try3.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prow).to_csv(OUT + "diffuse_obstruction_try3_pos.csv", index=False, encoding="utf-8-sig")
    for cond in ("난방", "비난방"):
        for f in (False, True):
            x = df[(df.조건 == cond) & (df.차단 == f)]
            print(f"{cond} 차단={f} n={len(x)}  케이스MAE 평균 {x.MAE.mean():.4f} 최대 {x.MAE.max():.4f} | 점 최대 {x.최대.max():.3f} | "
                  f"0mm 편향 평균 {x.영mm_편향.mean():+.4f} 범위 {x.영mm_편향.min():+.3f}~{x.영mm_편향.max():+.3f} | 500mm 편향 평균 {x.평면500_편향.mean():+.4f} "
                  f"| ΔTs오차 평균절대 {x.dTs_오차.abs().mean():.4f} 최대 {x.dTs_오차.abs().max():.4f} 평균 {x.dTs_오차.mean():+.4f} | 0.05초과 {x.초과005.mean():.2f}%")
