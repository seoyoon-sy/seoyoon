"""독립 테스트 T1 ~ T8 평가 (CLAUDE.md 8.44 뒤에 만든 새 벽체, 규칙 고정 후 처음 평가).
VOLTRA 결과(voltra_results/independent_test/{이름}_{heating,nonheating}.txt, Report Output)와 hbcalc 5가지 설정을 비교한다.
설정: half, low, 이전 표(손격자용 w 표), 새 표(자동 격자용 w 표, conv_lump='auto'), voltra(표면 처리 규칙, 맞춘 값 없음).
이 결과를 본 뒤에는 규칙을 다시 맞추지 않는다(다시 맞추면 독립 평가가 아니게 된다).
야간(18 ~ 07시) 744시간 중 야간 시간별 값, 12위치 표면온도. 0 mm 편향 = 모사 - VOLTRA 의 야간 평균(ΔTs 오차 = -0 mm 편향, 같은 평면 벽체 가정).
사용: python examples/eval_independent_test.py   (결과 results/accuracy/independent_test.csv, 원자료 independent_test.pkl)"""
import os, sys, re, csv, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.model import Material
from hbcalc.presets import try3_spec
from hbcalc.lump_rule import strip_weight
from hbcalc.voltra_report import read_report
from hbcalc.validate import night_mask
R = lambda *p: os.path.join(ROOT, *p)
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); N = 744; mk = night_mask(N)
NAMES = dict(concrete="concrete", EPS="EPS", block="0.5B cement block", gypsum="gypsum", XPS="XPS")
FILES = {"heating": "Wall_TI.FTE", "nonheating": "Wall_TI_2.FTE"}
SCHEMES = ["half", "low", "이전 표(손격자용)", "새 표(자동 격자용)", "voltra"]
bcs = {c: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD"))) for c, ti in FILES.items()}
defs = {r["벽체"]: r for r in csv.DictReader(open(R("voltra_inputs", "independent_test", "wall_definitions.csv"), encoding="utf-8-sig"))}
def model_of(name):
    row = defs[name]; th = [float(row[k]) for k in ("콘크리트_mm", "EPS_mm", "블록_mm", "석고_mm")]
    mats = {NAMES[k]: Material(NAMES[k], *(float(row[f"{k}_{q}"]) for q in ("열전도율", "밀도", "비열"))) for k in NAMES if f"{k}_열전도율" in row and row[f"{k}_열전도율"] != ""}
    layers = [Layer(NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    spec = replace(try3_spec(snap_mm=1), layers=layers, partition_mm=float(row["내벽_mm"]), strip_mm=float(row["판_두께_mm"]),
                   strip_extend_mm=float(row["판_돌출_mm"]), materials=mats)
    return build_tjunction(spec), float(row["XPS_열전도율"])
def check_grid(name, cond, m):
    t = open(R("voltra_inputs", "independent_test", f"{name}_{cond}.vtr"), "rb").read().decode("cp949", "replace")
    for ax, e in (("xwid", m.xe), ("ywid", m.ye)):
        v = np.array([float(x) for x in re.findall(ax + r"=([\d.]+)", t)]); mine = np.diff(e) * 1000
        if len(v) != len(mine) or abs(v - mine).max() > 1e-3: return False
    return True
def run(m, lam_xps, cond, scheme):
    cl = {"half": "half", "low": "low", "voltra": "voltra", "새 표(자동 격자용)": "auto"}.get(scheme)
    if scheme == "이전 표(손격자용)":
        w = strip_weight(lam_xps, "hand"); cl = "low" if w >= 1.0 else (w, w)
    s = Solver(m, conv_lump=cl); s.run(bcs[cond], nsteps=N)
    return np.array([s.series(f"surf_{p}")[:N] for p in POS]).T          # (N, 12)
if __name__ == "__main__":
    out = R("results", "accuracy", "independent_test.pkl")
    data = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    for name in defs:
        m, lx = model_of(name)
        for cond in FILES:
            f = R("voltra_results", "independent_test", f"{name}_{cond}.txt")
            if not os.path.exists(f): continue
            if (name, cond) in data: continue
            ok = check_grid(name, cond, m)
            V = read_report(f)[1][:N, 7:][:, 0::2]
            res = {"V": V, "grid_ok": ok, "xps": lx}
            for sc in SCHEMES: res[sc] = run(m, lx, cond, sc)
            data[(name, cond)] = res; pickle.dump(data, open(out, "wb")); print(name, cond, "격자 일치" if ok else "격자 불일치!", flush=True)
    rows = []
    for (name, cond), d in sorted(data.items()):
        V = d["V"]
        for sc in SCHEMES:
            e = d[sc] - V; en = e[mk]
            rows.append(dict(벽체=name, 조건=cond, XPS=d["xps"], 설정=sc, 야간MAE=abs(en).mean(), 야간RMSE=np.sqrt((en ** 2).mean()), **{"0mm편향": en[:, 0].mean(), "0mm_MAE": abs(en[:, 0]).mean()},
                             최대절대=abs(en).max(), **{"0.05초과%": 100 * (abs(en) > 0.05).mean(), "0.1초과%": 100 * (abs(en) > 0.1).mean()},
                             t0_RMSE=np.sqrt((e[0] ** 2).mean()), **{"t0_0mm": e[0, 0]}, 격자일치=d["grid_ok"]))
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(R("results", "accuracy", "independent_test.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for cond in FILES:
        s = df[df.조건 == cond]
        if s.empty: continue
        print(f"\n== {cond}: 벽체 {s.벽체.nunique()}개, 설정별 평균 (야간 시간별 12위치)")
        print(s.groupby("설정", sort=False)[["야간MAE", "야간RMSE", "0mm편향", "0mm_MAE", "최대절대", "0.05초과%", "0.1초과%", "t0_RMSE"]].mean().round(4).to_string())
        print(f"-- 벽체별 야간 MAE ({cond})"); print(s.pivot(index=["벽체", "XPS"], columns="설정", values="야간MAE")[SCHEMES].round(4).to_string())
        print(f"-- 벽체별 0 mm 편향 ({cond})"); print(s.pivot(index=["벽체", "XPS"], columns="설정", values="0mm편향")[SCHEMES].round(3).to_string())
        print(f"-- 벽체별 최대 절대오차 ({cond})"); print(s.pivot(index=["벽체", "XPS"], columns="설정", values="최대절대")[SCHEMES].round(3).to_string())
