"""자동 격자(hbcalc 생성 .vtr용)의 대류 배분 규칙 w(결로 방지 단열재 열전도율)를 calib_steady 14개 VOLTRA 정상상태 결과로 맞춘다.
1) 케이스마다 모든 면 같은 w를 0.2 ~ 1.0(0.025 단위)로 훑어 12위치 표면온도 RMSE 곡선 저장(results/accuracy/calib_w_curves.pkl).
2) 후보 규칙(현재 auto 규칙 = 손격자 기준, 새 표)을 케이스별 RMSE와 0 mm 편차로 비교. 새 표는 EPS 0.032 열 10개로 맞추고 EPS 0.024, 0.06 4개는 표본 밖으로 평가,
   EPS 0.032 열은 한 점씩 빼고 맞춰 예측하는 교차 검증(LOO)도 낸다.
사용: python examples/fit_strip_weight_auto_mesh.py"""
import os, sys, re, glob, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.voltra_report import read_report
from hbcalc.lump_rule import strip_weight
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
PK = os.path.join(ROOT, "results", "accuracy", "calib_w_curves.pkl")
WG = np.round(np.arange(0.2, 1.0001, 0.025), 4)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
if os.path.exists(PK):
    D = pickle.load(open(PK, "rb"))
else:
    D = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "voltra_results", "calib_steady", "C_E*.txt"))):
        name = os.path.basename(f)[:-4]; e, x = map(float, re.match(r"C_E([\d.]+)_X([\d.]+)", name).groups())
        V = read_report(f)[1][0][7:31].reshape(12, 2)[:, 0]
        m = try3_model({"EPS": e, "XPS": x}, snap_mm=1)
        dev = []
        for w in WG:
            s = Solver(m, conv_lump=(float(w), float(w))); s.run(bc, nsteps=1)
            dev.append(np.array([s.series(f"surf_{p}")[0] for p in POS]) - V)
        s = Solver(m, conv_lump="auto"); s.run(bc, nsteps=1)
        D[name] = dict(e=e, x=x, dev=np.array(dev), auto=np.array([s.series(f"surf_{p}")[0] for p in POS]) - V)
        print(name, flush=True)
    pickle.dump(D, open(PK, "wb"))
rm = lambda d: np.sqrt((d ** 2).mean(axis=-1))
def at(c, w):                                   # 임의 w 에서 편차(선형 보간)
    i = np.searchsorted(WG, w); i = min(max(i, 1), len(WG) - 1); a = (w - WG[i - 1]) / (WG[i] - WG[i - 1])
    return (1 - a) * c["dev"][i - 1] + a * c["dev"][i]
print("케이스별 최적 w(0.025 단위): RMSE, 0 mm 편차")
for n, c in D.items():
    r = rm(c["dev"]); k = int(r.argmin()); c["wb"] = float(WG[k])
    print(f"  {n}: w {WG[k]:.3f}  RMSE {r[k]:.3f}  0mm {c['dev'][k][0]:+.3f}   현재 auto RMSE {rm(c['auto']):.3f} (0mm {c['auto'][0]:+.3f})")
base = [n for n in D if D[n]["e"] == 0.032]; oth = [n for n in D if D[n]["e"] != 0.032]
lx = lambda n: np.log10(D[n]["x"])
def table(names):                               # (log10 λ, 최적 w) 점들. 최적 w는 단조(비감소)로 만들고 1.0 상한
    pts = sorted((lx(n), D[n]["wb"]) for n in names)
    xs, ws = zip(*pts); ws = np.maximum.accumulate(np.minimum(ws, 1.0)); return np.array(xs), ws
def pred(tab, x): return float(np.interp(np.log10(x), tab[0], tab[1]))
print("\n새 표(EPS 0.032 열 10점): ", ", ".join(f"{10 ** a:g}:{b:.3f}" for a, b in zip(*table(base))))
res = {}
for n in D:
    c = D[n]; tab = table([m for m in base if m != n]) if n in base else table(base)   # 기저 케이스는 자기 자신을 뺀 표(LOO)
    w = pred(tab, c["x"]); d = at(c, w); res[n] = (rm(d), d[0], w)
print("\n케이스별(새 표: EPS 0.032 열은 LOO, 그 밖은 표본 밖) 예측 w, RMSE, 0 mm 편차  |  현재 auto")
for n in D:
    print(f"  {n}: w {res[n][2]:.3f}  RMSE {res[n][0]:.3f}  0mm {res[n][1]:+.3f}  |  auto RMSE {rm(D[n]['auto']):.3f}  0mm {D[n]['auto'][0]:+.3f}")
for lab, names in (("EPS 0.032 열 10개(LOO)", base), ("EPS 0.024, 0.06 4개(표본 밖)", oth), ("전체 14개", list(D))):
    print(f"{lab}: 새 표 평균 RMSE {np.mean([res[n][0] for n in names]):.4f}, 0 mm 평균절대 {np.mean([abs(res[n][1]) for n in names]):.4f}, 최대 RMSE {max(res[n][0] for n in names):.3f} | "
          f"현재 auto {np.mean([rm(D[n]['auto']) for n in names]):.4f}, {np.mean([abs(D[n]['auto'][0]) for n in names]):.4f}, {max(rm(D[n]['auto']) for n in names):.3f}")
