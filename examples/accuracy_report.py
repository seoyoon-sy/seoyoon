"""현재 기본 설정(conv_lump='low', 일사 보정 0, VOLTRA 격자)의 정확도 집계.
VOLTRA 엑셀(난방 39케이스, 비난방 30케이스)과 야간(744시간 중 14시간 x 30일) 시간별 표면온도를 비교한다.
결과: results/accuracy/*.csv, results/accuracy/raw.pkl"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.planar import planar_model
from hbcalc.presets import TRY3_MATERIALS
from hbcalc.validate import *

R = ROOT + "/"; OUT = R + "results/accuracy/"
mk = night_mask(744)
POS = EXCEL_LABELS
SIMPOS = SIM_POS_VOLTRA_ORDER            # 엑셀 열 i 는 모사 위치 SIMPOS[i] 와 대응(8.4)
tab = pd.read_excel(R + "정상상태_정답.xlsx"); tab.columns = ["case", "Phi", "Phi2", "P2D", "psi"]


def key(nm):
    e, x = parse_sheet(nm)
    return (e if e is not None else 0.032, x if x is not None else 1.6)


psi_by = {key(r.case): r.psi for _, r in tab.iterrows() if not r.case.startswith("Model A")}


def planar_night(e, bc):
    mats = dict(TRY3_MATERIALS); mats["EPS"] = replace(mats["EPS"], lam=e)
    pm = planar_model([("concrete", 200), ("EPS", 180), ("0.5B cement block", 90), ("gypsum", 10)], mats, hrb=5.25)
    s = Solver(pm); s.run(bc, nsteps=744)
    return s.series("surf")[:744][mk].mean()


raw = {}
for cond, ti, xlsx in (("난방", "Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"),
                       ("비난방", "Wall_TI_2.FTE", "Non_heating_Tsi.xlsx")):
    bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + ti, R + "Wall_HO.FHT", R + "Wall_HI.FHT",
                                    R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0)
    xl, names = load_cases(R + xlsx)
    for nm in names:
        s, m = run_case(nm, bc, mesh="voltra", positions=sorted(set(POS) | set(SIMPOS)))
        df = xl.parse(nm)
        ss = {p: s.series(f"surf_{p}") for p in set(POS) | set(SIMPOS)}
        bb = {p: s.series(f"back_{p}") for p in set(POS) | set(SIMPOS)}
        d, dw = compare_case(df, ss, bb, POS, SIMPOS)             # (야간시각, 12위치) 모사 - VOLTRA
        S = np.array([df[f"{p}mm 표면온도(℃)"].values[:744] for p in POS]).T[mk]
        e = key(nm)[0]
        raw[(cond, nm)] = dict(d=d, dw=dw, S=S, planar=planar_night(e, bc), key=key(nm))
        print(cond, nm, round(np.abs(d).mean(), 4), flush=True)
pickle.dump(raw, open(OUT + "raw.pkl", "wb"))
