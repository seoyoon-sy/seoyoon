"""격자 자투리, 급한 칸 크기비가 결과에 주는 영향 시험 (독립 테스트 T1 벽체, 난방 20도, 정상상태 t=0, 12위치 표면온도, 솔버 voltra).
A: 현재 자동 격자, B: y 방향 자투리와 급한 칸 비를 완만하게 고친 격자, C: A의 모든 칸을 반으로 쪼갠 격자(수렴 참고), D: C를 한 번 더 쪼갠 격자.
사용: python examples/diagnose_grid_sensitivity.py"""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.model import Material
from hbcalc.presets import try3_spec
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); R = lambda *p: os.path.join(ROOT, *p)
NAMES = dict(concrete="concrete", EPS="EPS", block="0.5B cement block", gypsum="gypsum", XPS="XPS")
row = next(r for r in csv.DictReader(open(R("voltra_inputs", "independent_test", "wall_definitions.csv"), encoding="utf-8-sig")) if r["벽체"] == "T1")
th = [float(row[k]) for k in ("콘크리트_mm", "EPS_mm", "블록_mm", "석고_mm")]
mats = {NAMES[k]: Material(NAMES[k], float(row[f"{k}_열전도율"]), float(row[f"{k}_밀도"]), float(row[f"{k}_비열"])) for k in NAMES}
layers = [Layer(NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
spec = replace(try3_spec(snap_mm=1), layers=layers, partition_mm=float(row["내벽_mm"]), strip_mm=float(row["판_두께_mm"]), strip_extend_mm=float(row["판_돌출_mm"]), materials=mats)
mA = build_tjunction(spec); xe = mA.xe / 1e-3; ye = mA.ye / 1e-3
print("A y 칸:", np.diff(ye).round(1).tolist()); print("A x 칸:", np.diff(xe).round(1).tolist())
def ratios(e):
    d = np.diff(e); r = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:]); return r.max(), int((r > 1.5).sum())
print("A 인접 칸 크기비 최대, 1.5 초과 개수: x", ratios(xe), " y", ratios(ye))
# B: 블록 구간(y 1380~1480), 판 끝 앞 자투리, 판 끝 뒤를 같은 합으로 다시 나눔
def replace_run(e, lo, hi, cells):
    assert abs(sum(cells) - (hi - lo)) < 1e-9, (sum(cells), hi - lo)
    inner = [v for v in e if lo < v < hi]; assert lo in e and hi in e
    return np.array(sorted(set([v for v in e if v <= lo or v >= hi] + list(lo + np.cumsum(cells)[:-1]))))
yB = ye.copy(); y_in = sum(th)  # 실내 마감면에서 벗어난 값은 Yo 기준이라 아래에서 맞춘다
Yo = ye[np.argmax(np.abs(ye - 1000) < 1e-6)] if np.any(np.abs(ye - 1000) < 1e-6) else None
print("Yo(1000)?", Yo, " y_in =", 1000 + sum(th))
b0, b1 = 1000 + th[0] + th[1], 1000 + th[0] + th[1] + th[2]
yB = replace_run(yB, b0, b1, [20, 18, 16, 14, 12, 10, 10][:7] if th[2] == 100 else [th[2]])
t0 = 1000 + sum(th) + float(row["판_돌출_mm"])
nxt = [v for v in yB if v > t0][:2]
print("판 끝", t0, "그 앞 칸", np.diff(yB[np.searchsorted(yB, t0) - 3: np.searchsorted(yB, t0) + 1]), "뒤", nxt)
yB = replace_run(yB, t0 - 91, t0, [30, 30, 31])                  # 판 끝 앞 30, 37, 24 -> 30, 30, 31 (37 -> 24 급변 제거)
def refine(e, k=2): return np.concatenate([e[i] + (e[i + 1] - e[i]) * np.arange(k) / k for i in range(len(e) - 1)] + [e[-1:]])
def model(xe_, ye_):
    return build_tjunction(replace(spec, x_edges_mm=tuple(xe_), y_edges_mm=tuple(ye_)))
bc = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
def surf(m):
    s_ = Solver(m, conv_lump="voltra"); s_.run(bc, nsteps=1); return np.array([s_.series(f"surf_{p}")[0] for p in POS]), m.shape
res = {}
for nm, (xx, yy) in {"A 현재": (xe, ye), "B y만 완만": (xe, yB), "C 2배 세분": (refine(xe), refine(ye)), "D 4배 세분": (refine(xe, 4), refine(ye, 4))}.items():
    res[nm], sh = surf(model(xx, yy)); print(nm, "격자", sh, "인접비 y", ratios(yy)[0].round(2) if nm != "A 현재" else "", flush=True)
ref = res["D 4배 세분"]
print("12위치 모사 온도 (도):", POS)
for nm, v in res.items(): print(f"  {nm:10s}", np.round(v, 3).tolist())
print("D(4배 세분) 대비 차이 (도), 최대절대 / RMSE")
for nm, v in res.items():
    d = v - ref; print(f"  {nm:10s} 최대 {abs(d).max():.3f} RMSE {np.sqrt((d**2).mean()):.3f}  0 mm {d[0]:+.3f}  :", np.round(d, 3).tolist())
d = res["B y만 완만"] - res["A 현재"]; print("B - A (도):", np.round(d, 3).tolist(), " 최대", abs(d).max().round(3))
