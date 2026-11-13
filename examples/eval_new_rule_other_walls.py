"""새 대류 배분 규칙(자동 격자용 표, lump_rule 'auto')을 규칙 맞춤에 쓰지 않은 다른 구성의 벽체로 확인한다 (이전 표 'hand' 와 비교).
1) 정상상태(t=0): N1 ~ N6(난방 744시간 리포트의 첫 행), N5b, N6b(strip_split), N5w, T3nb(next_steady). 12위치 표면온도 RMSE와 0 mm 편차.
2) 744시간 야간 시간별 평균절대오차와 0 mm 편차: N1 ~ N6 난방, 비난방 접합부.
주의: 이 벽체들은 이전에 진단에 쓰였다(완전한 독립 테스트는 아니다). 완전한 독립 테스트는 voltra_inputs/independent_test (T1 ~ T8).
사용: python examples/eval_new_rule_other_walls.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.lump_rule import strip_weight
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
import test_walls_def as T
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); N = 744; mk = night_mask(N)
R = lambda *p: os.path.join(ROOT, *p)
mkbc = lambda ti: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
bcH, bcN = mkbc("Wall_TI.FTE"), mkbc("Wall_TI_2.FTE")
def xps_lam(m): return [x.lam for x in m.materials.values() if x.name == "XPS"][0]
def run(m, bc, rule, n):
    cl = (strip_weight(xps_lam(m), rule),) * 2
    s = Solver(m, conv_lump=cl); s.run(bc, nsteps=n); return np.array([s.series(f"surf_{p}")[:n] for p in POS]).T
def swap(key, strip):
    text, th, part, st, ext, props = T.WALLS[key]; props = dict(props); props["XPS"] = strip
    mats = {T.NAMES[k]: T.M(T.NAMES[k], *v) for k, v in props.items()}
    layers = [Layer(T.NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    return build_tjunction(replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=st, strip_extend_mm=ext, materials=mats))
st5, st6 = T.WALLS["N5_noblock"][5]["XPS"], T.WALLS["N6_noblock_insstrip"][5]["XPS"]
t3 = build_tjunction(try3_spec(snap_mm=1, layers=[Layer("concrete", 200), Layer("EPS", 180), Layer("gypsum", 10)])).with_properties(lam={"XPS": 0.04})
steady = [(k, T.build(k)["mj"], R("voltra_results", "test_walls", f"{k}_junction.txt")) for k in T.WALLS]
steady += [("N5b_strip0.04", swap("N5_noblock", st6), R("voltra_results", "strip_split", "N5b_strip0.04.txt")),
           ("N6b_strip0.5", swap("N6_noblock_insstrip", st5), R("voltra_results", "strip_split", "N6b_strip0.5.txt")),
           ("N5w_strip0.005", swap("N5_noblock", (0.005, 40, 1400)), R("voltra_results", "next_steady", "N5w_strip0.005.txt")),
           ("T3nb_strip0.04", t3, R("voltra_results", "next_steady", "T3nb_strip0.04.txt"))]
print("1) 정상상태 t=0, 12위치 표면온도: RMSE / 0 mm 편차 (모사 - VOLTRA).  이전 표(hand) 대 새 표(auto)")
rows = []
for name, m, f in steady:
    V = read_report(f)[1][0][7:31].reshape(12, 2)[:, 0]; lam = xps_lam(m); out = {}
    for rule in ("hand", "auto"):
        d = run(m, bcH, rule, 1)[0] - V; out[rule] = (np.sqrt((d ** 2).mean()), d[0], strip_weight(lam, rule))
    rows.append(dict(벽체=name, 판_열전도율=lam, w_이전=out["hand"][2], RMSE_이전=out["hand"][0], d0_이전=out["hand"][1], w_새것=out["auto"][2], RMSE_새것=out["auto"][0], d0_새것=out["auto"][1]))
    print(f"  {name}: 판 {lam:g}  이전 w {out['hand'][2]:.2f} {out['hand'][0]:.3f} / {out['hand'][1]:+.3f}   새 w {out['auto'][2]:.2f} {out['auto'][0]:.3f} / {out['auto'][1]:+.3f}", flush=True)
df = pd.DataFrame(rows); print(f"  평균 RMSE 이전 {df.RMSE_이전.mean():.4f} 새것 {df.RMSE_새것.mean():.4f}, 최대 이전 {df.RMSE_이전.max():.3f} 새것 {df.RMSE_새것.max():.3f}")
print("\n2) 744시간 야간 시간별 평균절대오차 / 0 mm 편차: 이전 표 대 새 표")
rows2 = []
for cond, d_, bc in (("난방", "test_walls", bcH), ("비난방", "test_walls_nonheating", bcN)):
    for k in T.WALLS:
        m = T.build(k)["mj"]; V = read_report(R("voltra_results", d_, f"{k}_junction.txt"))[1][:N, 7:][:, 0::2]; out = {}
        for rule in ("hand", "auto"):
            e = (run(m, bc, rule, N) - V)[mk]; out[rule] = (np.abs(e).mean(), e[:, 0].mean())
        rows2.append(dict(조건=cond, 벽체=k, MAE_이전=out["hand"][0], d0_이전=out["hand"][1], MAE_새것=out["auto"][0], d0_새것=out["auto"][1]))
        print(f"  {cond} {k}: 이전 {out['hand'][0]:.3f} / {out['hand'][1]:+.3f}   새것 {out['auto'][0]:.3f} / {out['auto'][1]:+.3f}", flush=True)
d2 = pd.DataFrame(rows2); print(f"  평균 MAE 이전 {d2.MAE_이전.mean():.4f} 새것 {d2.MAE_새것.mean():.4f}")
os.makedirs(R("results", "accuracy"), exist_ok=True)
df.round(4).to_csv(R("results", "accuracy", "new_rule_other_walls_steady.csv"), index=False, encoding="utf-8-sig"); d2.round(4).to_csv(R("results", "accuracy", "new_rule_other_walls_744h.csv"), index=False, encoding="utf-8-sig")
