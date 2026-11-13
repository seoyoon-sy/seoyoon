"""VOLTRA식 표면 처리(conv_lump='voltra')를 744시간 야간 시간별 표면온도로 검증 (CLAUDE.md 8.52): 경험 보정(auto)과 비교.
대상: TRY3 난방 39, 비난방 30(VOLTRA 엑셀, 손격자), N1 ~ N6 난방과 비난방(자동 격자), 변형 A ~ D 비난방. 지표: 12위치 야간 시간별 평균절대오차와 0 mm 편차.
사용: python examples/compare_voltra_scheme_744h.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.validate import *
from hbcalc.voltra_report import read_report
import test_walls_def as T
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N); POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
mkbc = lambda ti: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
bcs = {"난방": mkbc("Wall_TI.FTE"), "비난방": mkbc("Wall_TI_2.FTE")}
rows = []
def rec(grp, name, cl, e):
    rows.append(dict(그룹=grp, 케이스=name, 방식=cl, MAE=np.abs(e).mean(), d0=e[:, 0].mean(), 최대=np.abs(e).max()))
for cond, xlsx in (("난방", "voltra_30cases_night420_통합_v2.xlsx"), ("비난방", "Non_heating_Tsi.xlsx")):
    xl, names = load_cases(R(xlsx)); allp = sorted(set(EXCEL_LABELS) | set(SIM_POS_VOLTRA_ORDER))
    for nm in names:
        df = xl.parse(nm)
        for cl in ("auto", "voltra"):
            s, m = run_case(nm, bcs[cond], mesh="voltra", positions=allp, solver_kw=dict(conv_lump=cl))
            d, dw = compare_case(df, {p: s.series(f"surf_{p}") for p in allp}, {p: s.series(f"back_{p}") for p in allp}, EXCEL_LABELS, SIM_POS_VOLTRA_ORDER); rec(f"TRY3 {cond}", nm, cl, d)
        print(cond, nm, flush=True)
L = lambda eps=180: [Layer("concrete", 200), Layer("EPS", eps), Layer("0.5B cement block", 90), Layer("gypsum", 10)]
for key in T.WALLS:
    b = T.build(key)
    for cond, d_ in (("난방", "test_walls"), ("비난방", "test_walls_nonheating")):
        V = read_report(R("voltra_results", d_, f"{key}_junction.txt"))[1][:N, 7:][:, 0::2]
        for cl in ("auto", "voltra"):
            s = Solver(b["mj"], conv_lump=cl); s.run(bcs[cond], nsteps=N); e = (np.array([s.series(f"surf_{p}")[:N] for p in POS]).T - V)[mk]; rec(f"추가 벽체 {cond}", key, cl, e)
    print(key, flush=True)
VAR = {"A_EPS100": dict(layers=L(100)), "B_EPS250": dict(layers=L(250)), "C_partition250": dict(partition_mm=250), "D_strip20": dict(strip_mm=20, strip_extend_mm=120)}
for key, kw in VAR.items():
    V = read_report(R("voltra_results", f"{key}.txt"))[1][:N, 7:][:, 0::2]; m = build_tjunction(replace(try3_spec(snap_mm=1), **kw))
    for cl in ("auto", "voltra"):
        s = Solver(m, conv_lump=cl); s.run(bcs["비난방"], nsteps=N); rec("변형 A ~ D 비난방", key, cl, (np.array([s.series(f"surf_{p}")[:N] for p in POS]).T - V)[mk])
df = pd.DataFrame(rows); df.to_csv(R("results", "accuracy", "voltra_scheme_744h.csv"), index=False, encoding="utf-8-sig")
g = df.groupby(["그룹", "방식"]).agg(케이스=("MAE", "size"), MAE평균=("MAE", "mean"), MAE최대=("MAE", "max"), d0평균절대=("d0", lambda x: np.abs(x).mean()), 시간최대오차=("최대", "max")).round(4)
print(g.to_string()); print("\n전체 평균 MAE:", df.groupby("방식").MAE.mean().round(4).to_dict(), "  0.05 이하 케이스:", df[df.MAE <= 0.05].groupby("방식").size().to_dict(), "/", df.groupby("방식").size().to_dict())
