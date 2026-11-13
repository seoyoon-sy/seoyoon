"""Model A 를 피팅 케이스에 넣어 본다. 정의(.vtr/.trc)가 없어서 EPS 두께와 열전도율, 결로 방지 단열재 열전도율을 가정해 훑는다.
H1: EPS 160 mm, 0.024 / H2: EPS 180 mm, 0.027 (정상상태 표 Φ2D 7.2223 W 에서 역산한 저항이 같음, 8.33). 격자는 hbcalc 자동 격자(snap 1 mm).
비교: 야간 12위치 시간별 평균절대오차와 0 mm 편향(모사 - VOLTRA). conv_lump 는 half, low, auto."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.geometry import Layer
from hbcalc.presets import try3_model
from hbcalc.validate import EXCEL_LABELS, SIM_POS_VOLTRA_ORDER, night_mask, compare_case

R = ROOT + "/"
bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + "Wall_TI.FTE", R + "Wall_HO.FHT", R + "Wall_HI.FHT", R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0)
df = pd.ExcelFile(R + "voltra_30cases_night420_통합_v2.xlsx").parse("Model A")
POS = sorted(set(EXCEL_LABELS) | set(SIM_POS_VOLTRA_ORDER))
hyp = {"H1(160, 0.024)": (160, 0.024), "H2(180, 0.027)": (180, 0.027)}
rows = []
for hn, (t, lam) in hyp.items():
    for x in (0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5, 1.0, 2.0, 4.3):
        for cl in ("half", "low", "auto"):
            m = try3_model({"EPS": lam, "XPS": x}, layers=[Layer("concrete", 200), Layer("EPS", t), Layer("0.5B cement block", 90), Layer("gypsum", 10)],
                           snap_mm=1, offsets_mm=tuple(POS))
            s = Solver(m, conv_lump=cl); s.run(bc)
            ss = {p: s.series(f"surf_{p}") for p in POS}; bb = {p: s.series(f"back_{p}") for p in POS}
            d, _ = compare_case(df, ss, bb, EXCEL_LABELS, SIM_POS_VOLTRA_ORDER)
            rows.append(dict(hyp=hn, xps=x, lump=cl, mae=np.abs(d).mean(), bias0=d[:, 0].mean(), bias_all=d.mean()))
            print(rows[-1], flush=True)
out = pd.DataFrame(rows); out.to_csv(R + "results/accuracy/model_a_scan.csv", index=False)
print(out.pivot_table(index=["hyp", "xps"], columns="lump", values="mae").round(3))
print(out.pivot_table(index=["hyp", "xps"], columns="lump", values="bias0").round(3))
