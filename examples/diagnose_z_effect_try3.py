"""진단(8.75): 확산일사 차단을 켰을 때 드러나는 접합부 음의 오차가 z 방향(3D) 효과로 설명되는지 TRY3 엑셀 69케이스로 확인.
케이스마다 3D 정상상태(z 100 mm 조각 10개, z 전도 + 조각별 복사, 대류 half)의 중간면과 2D half 정상상태의 차이 c(위치)를
단위 온도차(실내 1, 실외 0)로 구해, 2D 기본 결과(차단 켬)에 시간별로 더한다(중첩 근사, 맞춘 값 없음).
시간 변화 처리 3가지: inst = c x (Ti - To)(t), d24 = c x 24시간 이동평균(Ti - To), surf = c/(1-Ts_unit) x (Ti - Ts_2D(t)) 위치별.
결과: results/accuracy/z_effect_try3.csv"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/examples")
import numpy as np, pandas as pd
from multiprocessing import Pool
import steady3d_test as S3
from hbcalc import Boundary, Solver
from hbcalc.validate import *
from hbcalc.presets import try3_voltra_mesh_model, try3_excel_strip_props

R = ROOT + "/"; POS = EXCEL_LABELS; SIMPOS = SIM_POS_VOLTRA_ORDER; ALLP = tuple(sorted(set(POS) | set(SIMPOS)))
CONDS = (("난방", "Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"), ("비난방", "Wall_TI_2.FTE", "Non_heating_Tsi.xlsx"))
NZ = 10; mk = night_mask(744)


def zcorr_unit(m, hin, hout):
    s, E, W3, Kc3, Kz, ext, hk, A2 = S3.steady3d(m, None, nz=NZ)
    zv = {z: ((0.0, hout) if m.zones[z].kind == "outdoor" else (1.0, hin)) for z in s.zone_ids}
    T3 = S3.solve3d(s, E, W3, Kc3, Kz, ext, hk, A2, zv, nz=NZ, hrb=m.hrb)[0][NZ // 2]
    b = Boundary.constant(2, T_out=0.0, T_in=1.0, h_out=hout, h_in=hin)
    s2 = Solver(m, conv_lump="half"); s2.run(b, nsteps=1); T2 = s2.T_all[0]
    o = [m.outputs[f"surf_{p}"] for p in SIMPOS]
    idx = [s.idx[i, j] for i, j, *_ in o]
    return np.array([T3[k] - T2[k] for k in idx]), np.array([T2[k] for k in idx])


def job(a):
    cond, ti, xlsx, nm = a
    df = pd.read_excel(R + xlsx, sheet_name=nm)
    Vs = np.array([df[f"{p}mm 표면온도(℃)"].values[:744] for p in POS]).T
    out = []
    for flag in (False, True):
        bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + ti, R + "Wall_HO.FHT", R + "Wall_HI.FHT",
                                        R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0, diffuse_obstruction=flag)
        s, m = run_case(nm, bc, mesh="voltra", positions=ALLP)
        M = np.array([s.series(f"surf_{p}")[:744] for p in SIMPOS]).T
        out.append((flag, M, bc))
    c, tsu = zcorr_unit(m, bc.h_in[:744].mean(), bc.h_out[:744].mean())
    dT = (bc.T_in - bc.T_out)[:744]
    d24 = pd.Series(dT).rolling(24, min_periods=1).mean().values
    rows = []
    for flag, M, _ in out:
        var = {"none": M, "inst": M + np.outer(dT, c), "d24": M + np.outer(d24, c),
               "surf": M + (bc.T_in[:744, None] - M) * (c / (1 - tsu))[None, :]}
        for k, Mv in var.items():
            d = (Mv - Vs)[mk]; a_ = np.abs(d)
            rows.append(dict(조건=cond, 케이스=nm, 차단=flag, 보정=k, MAE=a_.mean(), 최대=a_.max(), 영mm_편향=d[:, 0].mean(),
                             평면500_편향=d[:, -1].mean(), dTs_오차=d[:, -1].mean() - d[:, 0].mean(), 초과005=(a_ > 0.05).mean() * 100,
                             c0=c[0], c500=c[-1]))
    return rows


if __name__ == "__main__":
    jobs = []
    for cond, ti, xlsx in CONDS:
        _, names = load_cases(R + xlsx); jobs += [(cond, ti, xlsx, nm) for nm in names]
    with Pool(4) as p:
        res = sum(p.map(job, jobs, chunksize=1), [])
    df = pd.DataFrame(res); df.to_csv(R + "results/accuracy/z_effect_try3.csv", index=False, encoding="utf-8-sig")
    for cond in ("난방", "비난방"):
        for flag in (False, True):
            for k in ("none", "inst", "d24", "surf"):
                x = df[(df.조건 == cond) & (df.차단 == flag) & (df.보정 == k)]
                print(f"{cond} 차단={int(flag)} {k:4s}: MAE 평균 {x.MAE.mean():.4f} 케이스최대 {x.MAE.max():.4f} 점최대 {x.최대.max():.3f} | "
                      f"0mm 편향 {x.영mm_편향.min():+.3f}~{x.영mm_편향.max():+.3f} | 500 편향 {x.평면500_편향.min():+.3f}~{x.평면500_편향.max():+.3f} | "
                      f"ΔTs 평균절대 {x.dTs_오차.abs().mean():.4f} 최대 {x.dTs_오차.abs().max():.4f} | 0.05초과 {x.초과005.mean():.2f}%")
    x = df[(df.차단 == True) & (df.보정 == "none")]
    print("z 보정 단위값(실내외 1도 차이당) 0 mm", x.c0.min().round(4), "~", x.c0.max().round(4), " 500 mm", x.c500.min().round(4), "~", x.c500.max().round(4))
