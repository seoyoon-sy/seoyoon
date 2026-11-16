"""실외온도를 delta 시간 앞당겼을 때: (1) 평면 벽체(N1 ~ N6, 난방 744시간 VOLTRA 리포트) 오차가 나빠지는지, (2) 접합부 오차 최소가 어디인지(큰 이동량 포함).
사용: python examples/diagnose_boundary_shift2.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.voltra_report import read_report
from hbcalc.validate import night_mask
import test_walls_def as T
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N)
b0 = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
sh = lambda x, d: np.interp(np.arange(len(x)) + d, np.arange(len(x)), x)
print("난방 N1 ~ N6: 실외온도 앞당김 delta 별 야간 MAE. 평면 표면 / 접합부 12위치 (모사 - VOLTRA)")
for key in ("N1_light", "N3_thin", "N5_noblock", "N6_noblock_insstrip"):
    b = T.build(key); ap = read_report(R("voltra_results", "test_walls", f"{key}_planar.txt"))[1][:N]; aj = read_report(R("voltra_results", "test_walls", f"{key}_junction.txt"))[1][:N, 7:]
    Vj = aj[:, 0::2]; sj = Solver(b["mj"]); sp_ = Solver(b["mp"]); print(key)
    for d in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        bc = replace(b0, T_out=sh(b0.T_out, d))
        sp_.run(bc); ep = (sp_.series("surf")[:N] - ap[:, 5])[mk]
        sj.run(bc); POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); Mj = np.array([sj.series(f"surf_{p}")[:N] for p in POS]).T; ej = (Mj - Vj)[mk]
        print(f"   앞당김 {d:3.1f} h: 평면 MAE {abs(ep).mean():.4f} (편차 {ep.mean():+.4f})  접합부 MAE {abs(ej).mean():.4f}  0 mm MAE {abs(ej[:, 0]).mean():.4f}  최대 {abs(ej).max():.3f}", flush=True)
