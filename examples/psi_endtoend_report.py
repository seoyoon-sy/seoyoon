"""psi_endtoend.py 결과 요약: 시나리오별 LOCO(30fold x 30야간)와 독립 테스트(9케이스) 오차율, R²."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
D = os.path.join(ROOT, "results", "psi_endtoend")
lo = pd.read_pickle(os.path.join(D, "loco.pkl")); te = pd.read_pickle(os.path.join(D, "test.pkl"))
for d in (lo, te): d["err"] = (d.pred - d.actual).abs() / d.actual.abs() * 100
te["bias"] = (te.pred - te.actual) / te.actual * 100
pd.set_option("display.width", 220)
def tab(df, name):
    g = df.groupby(["shift", "scen", "model"]).apply(lambda x: pd.Series({"N": len(x), "평균오차율(%)": x.err.mean(), "중앙값(%)": x.err.median(), "R2": r2_score(x.actual, x.pred)}), include_groups=False).reset_index()
    return g
for name, df in (("LOCO (30fold x 30야간 = 900)", lo), ("독립 테스트 (9케이스, 케이스별 야간 평균 예측)", te)):
    g = tab(df, name)
    print(f"\n=========== {name}: 시나리오 x 모델 평균오차율(%) / R²")
    for sh in ("shift0", "shift-0.25"):
        sub = g[g["shift"] == sh]
        piv_e = sub.pivot(index="scen", columns="model", values="평균오차율(%)").round(2); piv_r = sub.pivot(index="scen", columns="model", values="R2").round(3)
        print(f"--- 일사 보정 {sh}"); print("오차율(%)\n", piv_e.to_string()); print("R²\n", piv_r.to_string())
# 핵심: 모델 5개 평균, A 대비 증가
print("\n=========== 요약: 5개 모델 평균 오차율(%) (독립 테스트 9케이스 / LOCO)")
for nm, df in (("독립테스트", te), ("LOCO", lo)):
    g = df.groupby(["shift", "scen"]).err.mean().unstack(0).round(2); print(nm); print(g.to_string())
print("\n독립 테스트 케이스별 부호 있는 오차(%) (시나리오 B, 학습 VOLTRA 시험 모사), 5개 모델 평균:")
b = te[te.scen.str.startswith("B")].groupby(["shift", "case"]).bias.mean().unstack(0).round(2); print(b.to_string())
