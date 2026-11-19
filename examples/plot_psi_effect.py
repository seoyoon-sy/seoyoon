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
MODELS = ["2차_회귀", "ANN", "KNN", "Random_Forest", "SVR"]; MLAB = ["2차 회귀", "ANN", "KNN", "RF", "SVR", "5개 평균"]
SCEN = {"A": "A: 학습 VOLTRA, 시험 VOLTRA (기준)", "B": "B: 학습 VOLTRA, 시험 모사", "C": "C: 학습 모사, 시험 VOLTRA", "D": "D: 학습 모사, 시험 모사"}
COL = {"A": "#59636e", "B": "#2f6fd0", "C": "#e8590c", "D": "#2b9348"}
sc = lambda d, k: d[d.scen.str.startswith(k)]
def per_model(d, k):
    x = sc(d, k); e = [x[x.model == m].err.mean() for m in MODELS]; return e + [np.mean(e)]
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
fig.text(0.03, 0.915, "사용자 학습 코드(v6.3)를 그대로 통과. VOLTRA 표면온도 대신 모사를 쓰거나 섞어도 예측 오차율이 거의 변하지 않는다 (일사 보정 0, 모델 5개)", fontsize=12.5, color=MUTED)
def bars(ax, d, title):
    w = 0.19; x = np.arange(6)
    for i, k in enumerate(SCEN):
        v = per_model(d, k); ax.bar(x + (i - 1.5) * w, v, w, color=COL[k], label=SCEN[k] if title.startswith("독립") else None, zorder=3)
        if i in (0, 3): pass
    for xi, m in enumerate(MLAB):
        a, dd = per_model(d, "A")[xi], per_model(d, "D")[xi]
    ax.set_xticks(x); ax.set_xticklabels(MLAB, fontsize=11.5, color=INK); ax.set_ylabel("Ψ 예측 평균 오차율 (%)", fontsize=11, color=MUTED); ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_title(title, loc="left", fontsize=13.5, color=INK); ax.set_facecolor(SURF)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.axvline(4.5, color=GRID, lw=1.2)
ax1 = fig.add_subplot(gs[0, 0]); bars(ax1, te, "독립 테스트 (10케이스): 모델별 평균 오차율"); ax1.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=1)
ax2 = fig.add_subplot(gs[0, 1]); bars(ax2, lo, "LOCO (30fold, 케이스 단위 교차검증): 모델별 평균 오차율")
for a in (ax1, ax2): a.set_ylim(0, max(a.get_ylim()[1], 8.5))
# 정답 대 예측 (5개 모델 평균 예측, 독립 테스트)
ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(SURF)
mk = {"A": "o", "B": "s", "C": "^", "D": "D"}; sz = {"A": 90, "B": 55, "C": 55, "D": 30}
for k in SCEN:
    g = sc(te, k).groupby("case").agg(actual=("actual", "first"), pred=("pred", "mean")).sort_values("actual")
    ax3.scatter(g.actual, g.pred, marker=mk[k], s=sz[k], facecolor="none" if k in "ABC" else COL[k], edgecolor=COL[k], lw=1.6, zorder=4 if k == "D" else 3, label=SCEN[k])
lim = [0.3, 0.85]; ax3.plot(lim, lim, color=INK, lw=1); ax3.set_xlim(lim); ax3.set_ylim(lim); ax3.set_aspect("equal"); ax3.grid(color=GRID, lw=0.7)
ax3.set_xlabel("정답 Ψ (TRISCO 정상상태, W/(m·K))", fontsize=11, color=MUTED); ax3.set_ylabel("예측 Ψ (5개 모델 평균)", fontsize=11, color=MUTED)
ax3.set_title("독립 테스트 10케이스: 정답 대 예측 (네 시나리오가 거의 겹친다)", loc="left", fontsize=13.5, color=INK)
for sp in ("top", "right"): ax3.spines[sp].set_visible(False)
ax3.legend(frameon=False, fontsize=9.5, loc="upper left")
# 표
axT = fig.add_subplot(gs[1, 1]); axT.axis("off"); axT.set_xlim(0, 1); axT.set_ylim(0, 1)
axT.text(0, 1.0, "성능 요약 (5개 모델 평균, %는 오차율)", fontsize=13.5, color=INK, va="top")
cols = [("시나리오 (학습 → 시험)", 0.0, "left"), ("독립 오차율", 0.43, "right"), ("A 대비", 0.535, "right"), ("LOCO 오차율", 0.70, "right"), ("A 대비", 0.805, "right"), ("RF R²", 0.895, "right"), ("Ψ RMSE", 1.0, "right")]
y0 = 0.88
for nme, x, ha in cols: axT.text(x, y0, nme, ha=ha, va="center", fontsize=10.5, weight="bold", color=INK)
axT.plot([0, 1], [y0 - 0.05] * 2, color=INK, lw=1)
SH = {"A": "A: VOLTRA → VOLTRA (기준)", "B": "B: VOLTRA → 모사", "C": "C: 모사 → VOLTRA", "D": "D: 모사 → 모사"}
for i, r in S.iterrows():
    y = y0 - 0.12 - i * 0.115; k = "ABCD"[i]
    axT.add_patch(plt.Rectangle((-0.005, y - 0.04), 0.012, 0.08, fc=COL[k], ec="none", transform=axT.transData, clip_on=False))
    vals = [SH[k], f"{r.독립_오차율:.2f} %", "-" if i == 0 else f"{r['A 대비 독립(%p)']:+.2f} %p", f"{r.LOCO_오차율:.2f} %", "-" if i == 0 else f"{r['A 대비 LOCO(%p)']:+.2f} %p", f"{r.RF_독립_R2:.3f}", f"{r.독립_Ψ_RMSE:.4f}"]
    for (nme, x, ha), t in zip(cols, vals): axT.text(x if nme.startswith("시나리오") is False else 0.02, y, t, ha=ha, va="center", fontsize=10, color=INK)
    axT.plot([0, 1], [y - 0.057] * 2, color=GRID, lw=0.7)
notes = ["오차율 = |예측 Ψ - 정답 Ψ| / 정답 Ψ x 100. 독립 테스트는 케이스별 야간 평균 예측,",
         "LOCO는 30케이스를 하나씩 빼며 학습한 야간 단위 예측. Ψ RMSE 단위 W/(m·K).",
         "모델: 사용자 코드의 2차 회귀, ANN, KNN, RF, SVR. 입력은 C(실내온도 - 실외온도),",
         "A(비열교부 - 열교부 표면온도). 정답 Ψ는 TRISCO 정상상태 표.",
         "한계: 독립 테스트 10케이스, 시드 한 번. 학습모델 자체의 케이스별 편향",
         "(예: 조정 0.07 케이스 약 +12 %)은 A에도 똑같이 있다.",
         "시나리오 간 차이(독립 -0.01 ~ +0.10 %p, LOCO 0.03 %p 이내)는",
         "표본 흔들림 범위일 수 있다."]
for j, t in enumerate(notes): axT.text(0, 0.28 - j * 0.037 - (0.012 if j >= 2 else 0) - (0.012 if j >= 4 else 0) - (0.012 if j >= 6 else 0), t, fontsize=9.3, color=MUTED, va="top")
fig.savefig(os.path.join(ROOT, "results", "fig_psi_effect.png"), dpi=130, facecolor=SURF); plt.close(fig)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); print(S.round(4).to_string(index=False))
