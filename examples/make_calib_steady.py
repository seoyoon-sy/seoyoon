"""대류 배분 보정(w) 검정용 VOLTRA 정상상태 .vtr 14개 (hbcalc 자동 격자 snap 1 mm, TRY3 구성: 콘크리트 200, EPS 180, 블록 90, 석고 10, 난방 20도, calc 3600).
결로 방지 단열재(XPS 판) 열전도율 10개(EPS 0.032) + EPS 0.024, 0.06 x XPS 0.0001, 0.005 4개. 기대값 expected_hbcalc_steady.csv (half, low, auto).
사용: python examples/make_calib_steady.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.write_vtr import write_vtr
OUT = os.path.join(ROOT, "voltra_inputs", "calib_steady"); os.makedirs(OUT, exist_ok=True)
TPL = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr"); POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
CASES = [(0.032, x) for x in (0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.1, 0.3, 1.0, 4.3)] + [(0.024, 0.0001), (0.06, 0.0001), (0.024, 0.005), (0.06, 0.005)]
rows = []
for e, x in CASES:
    name = f"C_E{e:g}_X{x:g}"
    m = try3_model({"EPS": e, "XPS": x}, snap_mm=1)
    write_vtr(m, TPL, os.path.join(OUT, name + ".vtr"), calc_s=3600, indoor_bc="heating")
    for cl in ("half", "low", "auto"):
        s = Solver(m, conv_lump=cl); s.run(bc, nsteps=1)
        r = {"케이스": name, "EPS": e, "XPS": x, "격자": f"{m.shape[0]}x{m.shape[1]}", "대류배분": cl}
        for p in POS: r[f"표면_{p}mm"] = s.series(f"surf_{p}")[0]
        rows.append(r)
    print(name, m.shape, flush=True)
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "expected_hbcalc_steady.csv"), index=False, encoding="utf-8-sig")
