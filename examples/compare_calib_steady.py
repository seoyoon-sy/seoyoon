"""보정용 VOLTRA 정상상태(calib_steady) 결과와 hbcalc 첫 스텝(정상상태, 일사 없음) 비교, 케이스별 최적 대류 배분 w 찾기.
voltra_results/calib_steady/{이름}.txt 가 있는 케이스만 처리. 모사는 half, low, auto와 모든 면 같은 w(0 ~ 1)를 시험.
출력: 표면온도 12위치 RMSE와 0 mm 편차(모사 - VOLTRA), 최적 w. 결과 표 results/accuracy/calib_steady_w.csv"""
import os, sys, re, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.voltra_report import read_report
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
WS = [round(w, 2) for w in np.arange(0.0, 1.0001, 0.1)]
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "voltra_results", "calib_steady", "C_E*.txt"))):
    name = os.path.basename(f)[:-4]; e, x = map(float, re.match(r"C_E([\d.]+)_X([\d.]+)", name).groups())
    cols, arr = read_report(f); row0 = arr[0]
    V = row0[7:31].reshape(12, 2)[:, 0]                           # 출력 노드 1, 3, 5, ... = 표면 12위치
    m = try3_model({"EPS": e, "XPS": x}, snap_mm=1)
    res = {}
    for key in ["half", "low", "auto"] + WS:
        s = Solver(m, conv_lump=key if isinstance(key, str) else (key, key)); s.run(bc, nsteps=1)
        S = np.array([s.series(f"surf_{p}")[0] for p in POS]); d = S - V
        res[key] = (np.sqrt((d ** 2).mean()), d[0], d[:8].mean())
    wbest = min(WS, key=lambda w: res[w][0])
    r = dict(케이스=name, EPS=e, XPS=x, w_best=wbest, rmse_best=res[wbest][0], 편차0mm_best=res[wbest][1])
    for k in ("half", "low", "auto"): r[f"rmse_{k}"], r[f"편차0mm_{k}"], r[f"편차0_100mm_{k}"] = res[k]
    r["w_auto"] = None
    rows.append(r)
    print(f"{name}: VOLTRA 0mm {V[0]:.3f}  최적 w {wbest} (RMSE {res[wbest][0]:.3f}, 0mm {res[wbest][1]:+.3f}) | half {res['half'][0]:.3f}/{res['half'][1]:+.3f}  low {res['low'][0]:.3f}/{res['low'][1]:+.3f}  auto {res['auto'][0]:.3f}/{res['auto'][1]:+.3f}", flush=True)
    print("   w별 RMSE:", " ".join(f"{w}:{res[w][0]:.3f}" for w in WS), flush=True)
df = pd.DataFrame(rows).drop(columns="w_auto").round(4)
os.makedirs(os.path.join(ROOT, "results", "accuracy"), exist_ok=True)
df.to_csv(os.path.join(ROOT, "results", "accuracy", "calib_steady_w.csv"), index=False, encoding="utf-8-sig")
