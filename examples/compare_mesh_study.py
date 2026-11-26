"""격자 시험(대칭 격자 L0 ~ L3) VOLTRA 정상상태 결과와 hbcalc 비교 (CLAUDE.md 8.72).
입력: voltra_results/mesh_study/MS_X{1.6,0.0001}_L{0..3}.txt (VOLTRA Report Output, t=0 행), 있는 파일만 처리.
출력: results/mesh_independence/mesh_study_vs_voltra.csv
사용: python examples/compare_mesh_study.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model, try3_excel_strip_props
from hbcalc.voltra_report import read_report
LEVELS = {"L0": (10, 100, 40), "L1": (5, 50, 20), "L2": (2.5, 25, 10), "L3": (1.25, 12.5, 5)}
P = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
rows, prof = [], {}
for x in (1.6, 0.0001):
    for lv, (a, b, c) in LEVELS.items():
        f = os.path.join(ROOT, "voltra_results", "mesh_study", f"MS_X{x:g}_{lv}.txt")
        if not os.path.exists(f): continue
        cols, A = read_report(f); v = A[0]; surf = v[7:31][0::2]
        rho, cp = try3_excel_strip_props(x)
        m = try3_model({"EPS": 0.032, "XPS": x}, h_min_mm=a, h_max_mm=b, h_layer_mm=c, symmetric_mesh=True).with_properties(rho={"XPS": rho}, cp={"XPS": cp})
        s = Solver(m); s.run(bc, nsteps=1); h = np.array([s.series(f"surf_{p}")[0] for p in P]); z = s.zone_flows_into_out()
        d = h - surf; prof[(x, lv)] = (surf, h)
        rows.append({"X": x, "격자": lv, "노드": s.N, "VOLTRA_0mm": surf[0], "hbcalc_0mm": round(h[0], 4), "VOLTRA_500mm": surf[-1], "hbcalc_500mm": round(h[-1], 4),
                     "차이_평균절대": round(abs(d).mean(), 4), "차이_최대절대": round(abs(d).max(), 4), "차이_0mm": round(d[0], 4), "차이_500mm": round(d[-1], 4),
                     "VOLTRA_ΔTs": round(surf[-1] - surf[0], 3), "hbcalc_ΔTs": round(h[-1] - h[0], 3),
                     "VOLTRA_실외유출W": v[2], "hbcalc_실외유출W": round(z[1][1][0], 3)})
        print(rows[-1], flush=True)
os.makedirs(os.path.join(ROOT, "results", "mesh_independence"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "mesh_independence", "mesh_study_vs_voltra.csv"), index=False, encoding="utf-8-sig")
for x in (1.6, 0.0001):
    ks = [k for k in prof if k[0] == x]
    if len(ks) < 2: continue
    print(f"\nX{x:g} 격자 단계 변화(L0 기준 이후 단계 - 이전 단계), 12위치 중 0, 50, 150, 500 mm")
    for k0, k1 in zip(ks[:-1], ks[1:]):
        for nm, j in (("VOLTRA", 0), ("hbcalc", 1)):
            dd = prof[k1][j] - prof[k0][j]
            print(f"  {k0[1]} -> {k1[1]} {nm:6s}: " + " ".join(f"{dd[i]:+.3f}" for i in (0, 5, 8, 11)))
