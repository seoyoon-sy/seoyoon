"""VOLTRA식 표면 처리(conv_lump='voltra', CLAUDE.md 8.52)를 경험 보정(auto)과 비교: 정상상태 t=0 표면온도 12위치(VOLTRA 리포트).
대상: calib_steady 14개(자동 격자 TRY3 구성), N1 ~ N6(난방 744시간 리포트 첫 행), N5b, N6b, N5w, T3nb. 사용: python examples/compare_voltra_scheme_steady.py"""
import os, sys, re, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_model, try3_spec
from hbcalc.voltra_report import read_report
import test_walls_def as T
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); R = lambda *p: os.path.join(ROOT, *p)
bc = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
def swap(key, strip):
    text, th, part, st, ext, props = T.WALLS[key]; props = dict(props); props["XPS"] = strip
    mats = {T.NAMES[k]: T.M(T.NAMES[k], *v) for k, v in props.items()}; layers = [Layer(T.NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    return build_tjunction(replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=st, strip_extend_mm=ext, materials=mats))
cases = []
for f in sorted(glob.glob(R("voltra_results", "calib_steady", "C_E*.txt"))):
    n = os.path.basename(f)[:-4]; e, x = map(float, re.match(r"C_E([\d.]+)_X([\d.]+)", n).groups()); cases.append((n, try3_model({"EPS": e, "XPS": x}, snap_mm=1), f))
for k in T.WALLS: cases.append((k, T.build(k)["mj"], R("voltra_results", "test_walls", f"{k}_junction.txt")))
st5, st6 = T.WALLS["N5_noblock"][5]["XPS"], T.WALLS["N6_noblock_insstrip"][5]["XPS"]
t3 = build_tjunction(try3_spec(snap_mm=1, layers=[Layer("concrete", 200), Layer("EPS", 180), Layer("gypsum", 10)])).with_properties(lam={"XPS": 0.04})
cases += [("N5b", swap("N5_noblock", st6), R("voltra_results", "strip_split", "N5b_strip0.04.txt")), ("N6b", swap("N6_noblock_insstrip", st5), R("voltra_results", "strip_split", "N6b_strip0.5.txt")),
          ("N5w", swap("N5_noblock", (0.005, 40, 1400)), R("voltra_results", "next_steady", "N5w_strip0.005.txt")), ("T3nb", t3, R("voltra_results", "next_steady", "T3nb_strip0.04.txt"))]
print("정상상태 t=0, 12위치 표면온도 RMSE / 0 mm 편차 (모사 - VOLTRA): half | auto(w 규칙) | voltra(표면 처리)")
agg = {k: [] for k in ("half", "auto", "voltra")}
for n, m, f in cases:
    V = read_report(f)[1][0][7:31].reshape(12, 2)[:, 0]; line = f"  {n:20s}"
    for cl in ("half", "auto", "voltra"):
        s = Solver(m, conv_lump=cl); s.run(bc, nsteps=1); d = np.array([s.series(f"surf_{p}")[0] for p in POS]) - V; agg[cl].append((np.sqrt((d ** 2).mean()), d[0], abs(d).max()))
        line += f" | {np.sqrt((d ** 2).mean()):.3f} / {d[0]:+.3f}"
    print(line, flush=True)
for cl in agg:
    a = np.array(agg[cl]); print(f"  [{cl}] 평균 RMSE {a[:, 0].mean():.4f}, 최대 RMSE {a[:, 0].max():.3f}, 0 mm 평균절대 {abs(a[:, 1]).mean():.4f}, 12위치 최대 오차의 최댓값 {a[:, 2].max():.3f}")
