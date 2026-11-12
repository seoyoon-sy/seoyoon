"""시험 벽체(N1 ~ N6) VOLTRA 결과(리포트 txt)와 hbcalc 비교.
사용: python examples/compare_test_wall_results.py [결과폴더]   (폴더에 {벽체}_junction.txt, {벽체}_planar.txt)
조건: 난방 20도 고정(Wall_TI.FTE), 일사 보정 0, 기본 설정(conv_lump='low'), VOLTRA 와 같은 격자. 야간(18시 ~ 07시) 기준."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
import test_walls_def as T

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_results", "test_walls")
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
N = 744; mk = night_mask(N)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
res = {}
for key in T.WALLS:
    fj, fp = (os.path.join(DIR, f"{key}_{k}.txt") for k in ("junction", "planar"))
    b = T.build(key); r = {}
    if os.path.exists(fp):
        _, a = read_report(fp); s = Solver(b["mp"]); s.run(bc, nsteps=N)
        r["planar"] = dict(Vs=a[:N, 5], Vb=a[:N, 6], Ms=s.series("surf")[:N], Mb=s.series("back")[:N])
    if os.path.exists(fj):
        _, a = read_report(fj); V = a[:N, 7:]
        s = Solver(b["mj"]); s.run(bc, nsteps=N)
        Ms = np.array([s.series(f"surf_{p}")[:N] for p in POS]).T; Mb = np.array([s.series(f"back_{p}")[:N] for p in POS]).T
        r["junction"] = dict(Vs=V[:, 0::2], Vb=V[:, 1::2], Ms=Ms, Mb=Mb)
    if r: res[key] = r
pickle.dump(res, open(os.path.join(DIR, "_cmp.pkl"), "wb"))
pd.set_option("display.width", 230)
rows = []
for key, r in res.items():
    row = {"벽체": key}
    if "planar" in r:
        d = r["planar"]; e = (d["Ms"] - d["Vs"])[mk]; ew = (d["Mb"] - d["Vb"])[mk]
        row.update({"평면 표면 편차": e.mean(), "평면 표면 MAE": np.abs(e).mean(), "평면 벽체쪽 MAE": np.abs(ew).mean(), "평면 t=0 편차": d["Ms"][0] - d["Vs"][0]})
    if "junction" in r:
        d = r["junction"]; e = (d["Ms"] - d["Vs"])[mk]; ew = (d["Mb"] - d["Vb"])[mk]
        row.update({"접합부 표면 MAE": np.abs(e).mean(), "접합부 RMSE": np.sqrt((e ** 2).mean()), "접합부 벽체쪽 MAE": np.abs(ew).mean(),
                    "0 mm 편차": e[:, 0].mean(), "0 mm MAE": np.abs(e[:, 0]).mean(), "최대 절대오차": np.abs(e).max()})
    if "planar" in r and "junction" in r:
        dv = r["planar"]["Vs"][mk].mean() - r["junction"]["Vs"][mk, 0].mean(); dm = r["planar"]["Ms"][mk].mean() - r["junction"]["Ms"][mk, 0].mean()
        row.update({"ΔTs VOLTRA": dv, "ΔTs hbcalc": dm, "ΔTs 오차": dm - dv})
    rows.append(row)
tab = pd.DataFrame(rows).set_index("벽체").round(3)
print(tab.T.to_string())
tab.to_csv(os.path.join(DIR, "comparison_summary.csv"), encoding="utf-8-sig")
print("\n접합부 위치별 야간 평균 편차(모사-VOLTRA, 표면) ℃, 위치 mm:", POS)
for key, r in res.items():
    if "junction" in r:
        d = r["junction"]; print(f"  {key:20s}", np.round((d["Ms"] - d["Vs"])[mk].mean(0), 3))
print("접합부 위치별 야간 평균 절대오차:")
for key, r in res.items():
    if "junction" in r:
        d = r["junction"]; print(f"  {key:20s}", np.round(np.abs(d["Ms"] - d["Vs"])[mk].mean(0), 3))
