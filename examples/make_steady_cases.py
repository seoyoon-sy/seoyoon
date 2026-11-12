"""격자 의존성 분리용 정상상태 시험 .vtr (G1 ~ G4): 좌우 대칭 격자 4종. 실내온도는 난방 20 도 고정(Wall_TI.FTE), calc=3600(첫 스텝이 정상상태).
같은 조건 hbcalc 첫 스텝(정상상태) 표면온도와 영역 열류를 expected_steady_hbcalc.csv 로 낸다.
사용: python examples/make_steady_cases.py [출력폴더]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.compat_trisco import read_trc
from hbcalc.write_vtr import write_vtr

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "vtr_cases_steady")
os.makedirs(OUT, exist_ok=True)
TPL = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr")
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
# 격자는 TRISCO 시험 T1 ~ T4(trisco_cases/*.trc)와 똑같이 맞춘다 (같은 격자로 VOLTRA 와 TRISCO 를 비교하기 위해)
CASES = {"G1_fine_sym_strong": ("대칭 세밀 격자, 강한 열교 (XPS 1.6)", "T1_fine_sym_strong", 1.6),
         "G2_fine_sym_weak": ("대칭 세밀 격자, 약한 열교 (XPS 0.0001)", "T2_fine_sym_weak", 0.0001),
         "G3_default_sym_strong": ("VOLTRA 굵기 규칙 격자(5 mm 시작) 대칭, 강한 열교", "T3_default_sym_strong", 1.6),
         "G4_default_sym_weak": ("VOLTRA 굵기 규칙 격자 대칭, 약한 열교", "T4_default_sym_weak", 0.0001)}
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
rows = []
for key, (desc, trc, xps) in CASES.items():
    xw, yw = read_trc(os.path.join(ROOT, "trisco_cases", trc + ".trc"))[:2]
    spec = replace(try3_spec(), x_edges_mm=tuple(np.r_[0, np.cumsum(xw)]), y_edges_mm=tuple(np.r_[0, np.cumsum(yw)]))
    m = build_tjunction(spec).with_properties(lam={"XPS": xps})
    write_vtr(m, TPL, os.path.join(OUT, f"{key}.vtr"), calc_s=3600, indoor_bc="heating")
    for cl in ("half", "low"):
        s = Solver(m, conv_lump=cl); s.run(bc, nsteps=1); F = s.zone_flows_into_out()
        o = [z for z in F if s.m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in F if s.m.zones[z].kind == "indoor")
        r = {"케이스": key, "설명": desc, "격자": f"{m.shape[0]}x{m.shape[1]}", "노드": s.N, "대류배분": cl,
             "실외유출W": F[o][1][0] - F[o][0][0], "왼쪽실내유입W": F[ind[0]][0][0] - F[ind[0]][1][0], "오른쪽실내유입W": F[ind[1]][0][0] - F[ind[1]][1][0]}
        for p in POS: r[f"표면_{p}mm"] = s.series(f"surf_{p}")[0]
        rows.append(r)
    print(key, m.shape, flush=True)
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "expected_steady_hbcalc.csv"), index=False, encoding="utf-8-sig")
print("완료:", OUT, "(실내온도 Wall_TI.FTE 20도 고정)")
