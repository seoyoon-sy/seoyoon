"""모사 표면온도가 선형열관류율(Ψ) 예측 성능에 주는 영향: 사용자 학습 코드(training/psi_model_v6_3.py)를 그대로 통과시킨 종단 시험(examples/psi_endtoend.py) 결과를 그림과 표로 정리.
시나리오: A 학습 VOLTRA 시험 VOLTRA(기준), B 학습 VOLTRA 시험 모사, C 학습 모사 시험 VOLTRA, D 학습 모사 시험 모사. 일사 보정 0, 2변수(C, A) 모델 5개.
정답 Ψ는 정상상태_정답.xlsx(TRISCO). 독립 테스트 10케이스(Model A 포함), LOCO(30fold).
출력: results/fig_psi_effect.png, results/accuracy/psi_effect_summary.csv
사용: python examples/plot_psi_effect.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb"
D = os.path.join(ROOT, "results", "psi_endtoend")
lo = pd.read_pickle(os.path.join(D, "loco.pkl")); te = pd.read_pickle(os.path.join(D, "test.pkl"))
lo = lo[lo["shift"] == "shift0"].copy(); te = te[te["shift"] == "shift0"].copy()
for d in (lo, te): d["err"] = (d.pred - d.actual).abs() / d.actual.abs() * 100; d["abse"] = (d.pred - d.actual).abs(); d["se"] = (d.pred - d.actual) ** 2
MODELS = ["2차_회귀", "ANN", "KNN", "Random_Forest", "SVR"]; MLAB = ["2차 회귀", "ANN", "KNN", "RF", "SVR"]
SCEN = {"A": "A: 학습 VOLTRA, 시험 VOLTRA (기준)", "B": "B: 학습 VOLTRA, 시험 모사", "C": "C: 학습 모사, 시험 VOLTRA", "D": "D: 학습 모사, 시험 모사"}
COL = {"A": "#59636e", "B": "#2f6fd0", "C": "#e8590c", "D": "#2b9348"}
sc = lambda d, k: d[d.scen.str.startswith(k)]
def per_model(d, k):
    x = sc(d, k); e = [x[x.model == m].err.mean() for m in MODELS]; return e
# ---- 요약 표 계산
rows = []
for k in SCEN:
    xt, xl = sc(te, k), sc(lo, k)
    r2 = [r2_score(xt[xt.model == m].actual, xt[xt.model == m].pred) for m in MODELS]
    rows.append(dict(시나리오=SCEN[k], 독립_오차율=xt.err.mean(), LOCO_오차율=xl.err.mean(), RF_독립_오차율=xt[xt.model == "Random_Forest"].err.mean(), RF_독립_R2=r2[3], 평균_독립_R2=np.mean(r2),
                     독립_Ψ_MAE=np.mean([xt[xt.model == m].abse.mean() for m in MODELS]), 독립_Ψ_RMSE=np.mean([np.sqrt(xt[xt.model == m].se.mean()) for m in MODELS]),
                     LOCO_R2=np.mean([r2_score(xl[xl.model == m].actual, xl[xl.model == m].pred) for m in MODELS])))
S = pd.DataFrame(rows); S["A 대비 독립(%p)"] = S.독립_오차율 - S.독립_오차율.iloc[0]; S["A 대비 LOCO(%p)"] = S.LOCO_오차율 - S.LOCO_오차율.iloc[0]
S.round(4).to_csv(os.path.join(ROOT, "results", "accuracy", "psi_effect_summary.csv"), index=False, encoding="utf-8-sig")
# ---- 그림
fig = plt.figure(figsize=(16, 10), facecolor=SURF); gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.845, bottom=0.05, hspace=0.36, wspace=0.16)
fig.text(0.03, 0.955, "모사 표면온도가 선형열관류율(Ψ) 예측 성능에 주는 영향", fontsize=22, weight="bold", color=INK)
fig.text(0.03, 0.915, "사용자 학습 코드(v6.3)를 그대로 통과. VOLTRA 대신 모사를 쓰거나 섞었을 때의 변화를 모델별로 본다 (일사 보정 0). 모델 평균은 쓰지 않는다", fontsize=12.5, color=MUTED)
def bars(ax, d, title):
    w = 0.19; x = np.arange(5)
    for i, k in enumerate(SCEN):
        v = per_model(d, k); ax.bar(x + (i - 1.5) * w, v, w, color=COL[k], label=SCEN[k] if title.startswith("독립") else None, zorder=3)
        if i in (0, 3): pass
    for xi, m in enumerate(MLAB):
        a, dd = per_model(d, "A")[xi], per_model(d, "D")[xi]
    ax.set_xticks(x); ax.set_xticklabels(MLAB, fontsize=11.5, color=INK); ax.set_ylabel("Ψ 예측 평균 오차율 (%)", fontsize=11, color=MUTED); ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_title(title, loc="left", fontsize=13.5, color=INK); ax.set_facecolor(SURF)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax1 = fig.add_subplot(gs[0, 0]); bars(ax1, te, "독립 테스트 (10케이스): 모델별 평균 오차율"); ax1.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=1)
ax2 = fig.add_subplot(gs[0, 1]); bars(ax2, lo, "LOCO (30fold, 케이스 단위 교차검증): 모델별 평균 오차율")
for a in (ax1, ax2): a.set_ylim(0, max(a.get_ylim()[1], 8.5))
# 정답 대 예측 (RF 한 모델, 독립 테스트)
ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(SURF)
mk = {"A": "o", "B": "s", "C": "^", "D": "D"}; sz = {"A": 90, "B": 55, "C": 55, "D": 30}
for k in SCEN:
    g = sc(te[te.model == "Random_Forest"], k).groupby("case").agg(actual=("actual", "first"), pred=("pred", "first")).sort_values("actual")
    ax3.scatter(g.actual, g.pred, marker=mk[k], s=sz[k], facecolor="none" if k in "ABC" else COL[k], edgecolor=COL[k], lw=1.6, zorder=4 if k == "D" else 3, label=SCEN[k])
lim = [0.3, 0.85]; ax3.plot(lim, lim, color=INK, lw=1); ax3.set_xlim(lim); ax3.set_ylim(lim); ax3.set_aspect("equal"); ax3.grid(color=GRID, lw=0.7)
ax3.set_xlabel("정답 Ψ (TRISCO 정상상태, W/(m·K))", fontsize=11, color=MUTED); ax3.set_ylabel("예측 Ψ (RF)", fontsize=11, color=MUTED)
ax3.set_title("독립 테스트 10케이스: 정답 대 예측, RF만 (모델 평균 없음)", loc="left", fontsize=13.5, color=INK)
for sp in ("top", "right"): ax3.spines[sp].set_visible(False)
ax3.legend(frameon=False, fontsize=9.5, loc="upper left")
# 표
axT = fig.add_subplot(gs[1, 1]); axT.axis("off"); axT.set_xlim(0, 1); axT.set_ylim(0, 1)
axT.text(0, 1.0, "모델별 결과 (평균 아님, 독립 테스트 10케이스)", fontsize=13.5, color=INK, va="top")
cols = [("모델", 0.0, "left"), ("A 케이스\n최대오차 %", 0.37, "right"), ("B - A\n최대 변화 %p", 0.58, "right"), ("C - A\n최대 변화 %p", 0.79, "right"), ("D - A\n최대 변화 %p", 1.0, "right")]
y0 = 0.86
for nme, x, ha in cols: axT.text(x, y0, nme, ha=ha, va="center", fontsize=10.5, weight="bold", color=INK, linespacing=1.15)
axT.plot([0, 1], [y0 - 0.075] * 2, color=INK, lw=1)
def pv(m, k): x = te[(te.model == m) & (te.scen.str[0] == k)].set_index("case"); return x.pred, x.actual
for i, m in enumerate(MODELS):
    pa, ac = pv(m, "A"); y = y0 - 0.15 - i * 0.105
    vals = [MLAB[i], f"{((pa - ac).abs() / ac * 100).max():.1f}"] + [f"{((pv(m, k)[0] - pa).abs() / ac * 100).max():.2f}" for k in "BCD"]
    for (nme, x, ha), t in zip(cols, vals): axT.text(x, y, t, ha=ha, va="center", fontsize=11, color=INK)
    axT.plot([0, 1], [y - 0.052] * 2, color=GRID, lw=0.7)
notes = ["케이스 한 개의 예측이 모사 때문에 가장 많이 달라진 크기이다(그 케이스가 어느 것인지는",
         "fig_psi_delta_by_model.png). 값이 큰 곳은 KNN, RF의 C(학습 모사, 시험 VOLTRA)로",
         "결로 방지 0.0001 + 조정 0.024 케이스이다. 자세한 표: results/accuracy/psi_effect_by_model.csv",
         "한계: 독립 테스트 10케이스, 시드 한 번. A 자체의 케이스별 편향(조정 0.07 약 +7 ~ +17 %)은",
         "학습모델 자체의 것이고 A에도 똑같이 있다. 점 하나는 30야간 평균 예측이다."]
for j, t in enumerate(notes): axT.text(0, 0.22 - j * 0.045, t, fontsize=9.3, color=MUTED, va="top")
fig.savefig(os.path.join(ROOT, "results", "fig_psi_effect.png"), dpi=130, facecolor=SURF); plt.close(fig)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); print(S.round(4).to_string(index=False))
