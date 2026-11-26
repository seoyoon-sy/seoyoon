"""균일 격자(10 mm, 5 mm) 정상상태: VOLTRA 리포트(t=0 행)와 hbcalc 비교 (CLAUDE.md 8.71).
입력: voltra_results/mesh_study_uniform/U{10,5}_X{1.6,0.0001}.txt (VOLTRA Report Output)
출력: results/mesh_independence/uniform_vs_voltra.csv, results/fig_mesh_uniform_vs_voltra.png
사용: python examples/compare_uniform_mesh.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for _f in fm.findSystemFonts():
    if "wqy" in _f.lower() or "WenQuanYi" in _f: fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model, try3_excel_strip_props
from hbcalc.voltra_report import read_report
POS = {10: (0, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500), 5: (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)}
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
rows, prof = [], {}
for x in (1.6, 0.0001):
    for mm in (10, 5):
        nm = f"U{mm}_X{x:g}"
        cols, A = read_report(os.path.join(ROOT, "voltra_results", "mesh_study_uniform", nm + ".txt"))
        v = A[0]; surf = v[7:31][0::2]               # 출력 노드 1, 3, 5, ... = 표면 (짝수 번은 석고보드 벽체쪽)
        rho, cp = try3_excel_strip_props(x)
        m = try3_model({"EPS": 0.032, "XPS": x}, mesh_mm=mm, offsets_mm=POS[mm]).with_properties(rho={"XPS": rho}, cp={"XPS": cp})
        s = Solver(m); s.run(bc, nsteps=2)
        h = np.array([s.series(f"surf_{p}")[0] for p in POS[mm]]); z = s.zone_flows_into_out()
        d = h - surf; prof[nm] = (POS[mm], surf, h)
        rows.append({"파일": nm, "노드": s.N, "VOLTRA_0mm": surf[0], "hbcalc_0mm": round(h[0], 4), "VOLTRA_500mm": surf[-1], "hbcalc_500mm": round(h[-1], 4),
                     "차이_평균절대": round(abs(d).mean(), 4), "차이_최대절대": round(abs(d).max(), 4),
                     "VOLTRA_실외유출W": v[2], "hbcalc_실외유출W": round(z[1][1][0], 3), "VOLTRA_ΔTs": round(surf[-1] - surf[0], 3), "hbcalc_ΔTs": round(h[-1] - h[0], 3)})
        print(rows[-1], flush=True)
os.makedirs(os.path.join(ROOT, "results", "mesh_independence"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "mesh_independence", "uniform_vs_voltra.csv"), index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for j, x in enumerate((1.6, 0.0001)):
    for mm, c in ((10, "tab:blue"), (5, "tab:red")):
        p, sv, hb = prof[f"U{mm}_X{x:g}"]; xs = np.arange(len(p))
        ax[0, j].plot(xs, sv, "-o", color=c, ms=4, label=f"VOLTRA 균일 {mm} mm"); ax[0, j].plot(xs, hb, "--s", color=c, ms=3, alpha=.7, label=f"hbcalc 균일 {mm} mm")
        ax[1, j].plot(xs, hb - sv, "-o", color=c, ms=4, label=f"균일 {mm} mm")
        ax[1, j].set_xticks(xs); ax[1, j].set_xticklabels([str(q) for q in p], fontsize=8)
    ax[0, j].set_title(f"결로 방지 판 열전도율 {x:g}: 표면온도(정상상태 t=0)"); ax[0, j].set_xticks(xs); ax[0, j].set_xticklabels([str(q) for q in p], fontsize=8)
    ax[1, j].set_title("hbcalc - VOLTRA"); ax[1, j].axhline(0, color="gray", lw=.8)
    for a in ax[:, j]: a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel("접합부로부터 거리 [mm] (U10과 U5의 위치 일부가 다름)")
    ax[0, j].set_ylabel("표면온도 [℃]"); ax[1, j].set_ylabel("차이 [℃]")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "fig_mesh_uniform_vs_voltra.png"), dpi=140)
