"""균일 격자 VOLTRA 시험 입력 (CLAUDE.md 8.68 후속). 접합부 표면온도의 격자 의존성을 VOLTRA와 hbcalc가 같은 균일 격자에서
어떻게 다른지 보기 위한 정상상태(calc 3600, 난방 20 도) .vtr 와 hbcalc 기대값.
  U10: 균일 10 mm. 출력 위치가 격자선에 있어야 하므로 10 mm 배수 12개 (0, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500)
  U5 : 균일 5 mm. 기존 12개 위치 (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
케이스: 결로 방지 판 열전도율 1.6(강함, 판 물성 콘크리트와 같음), 0.0001(약함, 판 밀도 25 비열 1300). EPS 0.032.
출력: voltra_inputs/mesh_study_uniform/*.vtr, expected_hbcalc_steady.csv (t=0 표면온도, 같은 위치)
사용: python examples/make_uniform_mesh_cases.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model, try3_excel_strip_props
from hbcalc.write_vtr import write_vtr

OUT = os.path.join(ROOT, "voltra_inputs", "mesh_study_uniform"); os.makedirs(OUT, exist_ok=True)
POS = {10: (0, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500), 5: (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)}
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
rows = []
for mm in (10, 5):
    for x in (1.6, 0.0001):
        rho, cp = try3_excel_strip_props(x)
        m = try3_model({"EPS": 0.032, "XPS": x}, mesh_mm=mm, offsets_mm=POS[mm]).with_properties(rho={"XPS": rho}, cp={"XPS": cp})
        nm = f"U{mm}_X{x:g}"
        write_vtr(m, os.path.join(ROOT, "TRY3_Plus_QTest2.vtr"), os.path.join(OUT, nm + ".vtr"), calc_s=3600, indoor_bc="heating")
        s = Solver(m); s.run(bc, nsteps=2)
        r = {"파일": nm, "격자": f"{m.shape[0]}x{m.shape[1]}", "노드": s.N}
        for p in POS[mm]: r[f"t0_표면_{p}mm"] = s.series(f"surf_{p}")[0]
        rows.append(r); print(nm, r["격자"], r["노드"], flush=True)
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "expected_hbcalc_steady.csv"), index=False, encoding="utf-8-sig")
