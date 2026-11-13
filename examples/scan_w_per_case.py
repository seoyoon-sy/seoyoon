"""TRY3 난방 케이스 중 결로 방지 단열재(XPS) 열전도율 0.5 미만인 케이스마다 대류 배분 가중치 w(모든 면 같은 값)를 바꿔
VOLTRA 엑셀 야간 시간별 표면온도 평균절대오차(12위치)가 최소인 w를 찾는다 (8.36). 결과 results/accuracy/w_per_case.csv"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary
from hbcalc.validate import *
R = ROOT + "/"; POS = EXCEL_LABELS; SIMPOS = SIM_POS_VOLTRA_ORDER
bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + "Wall_TI.FTE", R + "Wall_HO.FHT", R + "Wall_HI.FHT", R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0)
xl, names = load_cases(R + "voltra_30cases_night420_통합_v2.xlsx")
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0); rows = []
for nm in names:
    e, x = parse_sheet(nm)
    if x is None or x >= 0.5: continue
    df = xl.parse(nm)
    for w in WS:
        s, m = run_case(nm, bc, mesh="voltra", positions=sorted(set(POS) | set(SIMPOS)), solver_kw=dict(conv_lump=(w, w)))
        ss = {p: s.series(f"surf_{p}") for p in set(POS) | set(SIMPOS)}; bb = {p: s.series(f"back_{p}") for p in set(POS) | set(SIMPOS)}
        d, dw = compare_case(df, ss, bb, POS, SIMPOS)
        rows.append(dict(케이스=nm, XPS=x, w=w, MAE=np.abs(d).mean(), 편향_0mm=d[:, 0].mean(), MAE_0mm=np.abs(d[:, 0]).mean()))
    print(nm, flush=True)
d = pd.DataFrame(rows); d.to_csv(R + "results/accuracy/w_per_case.csv", index=False, encoding="utf-8-sig")
b = d.loc[d.groupby("케이스").MAE.idxmin()].sort_values("XPS"); print(b.round(4).to_string(index=False))
