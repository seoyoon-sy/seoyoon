"""모사 표면온도가 Ψ 예측에 주는 영향을 모델별, 케이스별로 본다 (모델 평균이나 부호 있는 오차의 평균으로 가리지 않는다).
입력: results/psi_endtoend/test.pkl, loco.pkl (psi_endtoend.py 결과, 일사 보정 0, 독립 테스트 10케이스 Model A 포함)
출력: results/fig_psi_pred_by_model.png   모델 5개 x (정답 대 예측, 부호 있는 오차 %). A(VOLTRA만)와 D(모사만)만 보여 준다
      results/fig_psi_delta_by_model.png  모사로 바꿨을 때 예측이 달라진 양(B, C, D - A, 케이스 x 모델)
      results/accuracy/psi_effect_by_model.csv  모델별, 시나리오별 오차 (케이스 단위와 야간 단위의 평균절대, 최대절대)
사용: python examples/plot_psi_by_model.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb"
D = os.path.join(ROOT, "results", "psi_endtoend")
te = pd.read_pickle(os.path.join(D, "test.pkl")); te = te[te["shift"] == "shift0"].copy()
MODELS = ["2차_회귀", "ANN", "KNN", "Random_Forest", "SVR"]; MLAB = {"2차_회귀": "2차 회귀", "ANN": "ANN", "KNN": "KNN", "Random_Forest": "RF", "SVR": "SVR"}
SC = {"A": "A: 학습 VOLTRA, 시험 VOLTRA (기준)", "B": "B: 학습 VOLTRA, 시험 모사", "C": "C: 학습 모사, 시험 VOLTRA", "D": "D: 학습 모사, 시험 모사"}
COL = {"A": "#59636e", "B": "#2f6fd0", "C": "#e8590c", "D": "#2b9348"}; MK = {"A": "o", "B": "s", "C": "^", "D": "D"}; SZ = {"A": 95, "B": 60, "C": 60, "D": 28}
te["sc"] = te.scen.str[0]
short = lambda n: n.replace("결로 방지 단열재", "결로").replace("단열재 조정", "조정").replace(" + ", "+")
cases = sorted(te.case.unique(), key=lambda c: te[te.case == c].actual.iloc[0]); act = {c: te[te.case == c].actual.iloc[0] for c in cases}
def P(m, s):                                      # 케이스별 예측 (야간 평균 예측)
    x = te[(te.model == m) & (te.sc == s)].set_index("case").pred; return x.reindex(cases)
# ---------------- 표 (평균 대신 최대를 함께)
rows = []
for m in MODELS:
    for s in "ABCD":
        x = te[(te.model == m) & (te.sc == s)]; e = ((x.pred - x.actual).abs() / x.actual * 100)
        nn = np.concatenate([(np.array(r.pred_nights) - r.actual) / r.actual * 100 for r in x.itertuples()])
        dA = (P(m, s) - P(m, "A")) / pd.Series(act) * 100
        xa = te[(te.model == m) & (te.sc == "A")].set_index("case")
        nd = np.concatenate([np.abs((np.array(r.pred_nights) - np.array(xa.loc[r.Index, "pred_nights"])) / r.actual * 100) for r in x.set_index("case").itertuples()]) if s != "A" else np.zeros(1)
        rows.append({"모델": MLAB[m], "시나리오": s, "케이스단위 평균절대오차율 %": e.mean(), "케이스단위 최대절대오차율 %": e.max(), "최대인 케이스": short(x.loc[e.idxmax(), "case"]),
                     "야간단위 평균절대 %": np.abs(nn).mean(), "야간단위 최대절대 %": np.abs(nn).max(), "A 대비 케이스별 최대 변화 %p": 0 if s == "A" else dA.abs().max(), "A 대비 야간단위 평균절대 변화 %p": 0 if s == "A" else nd.mean(), "A 대비 야간단위 최대 변화 %p": 0 if s == "A" else nd.max(),
                     "A 대비 가장 변한 케이스": "" if s == "A" else short(dA.abs().idxmax())})
tab = pd.DataFrame(rows); tab.round(3).to_csv(os.path.join(ROOT, "results", "accuracy", "psi_effect_by_model.csv"), index=False, encoding="utf-8-sig")
# ---------------- 그림 1: 모델별 정답 대 예측과 부호 있는 오차
fig, axs = plt.subplots(2, 5, figsize=(21, 9.4), facecolor=SURF, gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.38, "wspace": 0.22, "left": 0.045, "right": 0.99, "top": 0.78, "bottom": 0.16})
fig.text(0.02, 0.955, "모델별로 본 정답 대 예측 (모델 평균 없음)", fontsize=20, weight="bold", color=INK)
fig.text(0.02, 0.915, "점 하나 = 케이스 하나의 야간 평균 예측. 위: 정답 Ψ 대 예측 Ψ. 아래: 같은 점의 부호 있는 오차 (예측 - 정답) / 정답, %. A와 D 점이 겹쳐 보이면 VOLTRA 대신 모사를 써도 차이가 작다는 뜻이다. B, C는 psi_effect_by_model.csv에 있다. 독립 테스트 10케이스(Model A 포함), 일사 보정 0", fontsize=11.5, color=MUTED)
xs = np.arange(len(cases)); lim = [0.3, 0.85]; SHOW1 = "AD"
for j, m in enumerate(MODELS):
    a1, a2 = axs[0, j], axs[1, j]
    for s in SHOW1:
        p = P(m, s).values; a = np.array([act[c] for c in cases])
        a1.scatter(a, p, marker=MK[s], s=SZ[s], facecolor="none" if s in "ABC" else COL[s], edgecolor=COL[s], lw=1.5, zorder=3, label=SC[s] if j == 0 else None)
        a2.scatter(xs + {"A": -0.14, "D": 0.14}[s], (p - a) / a * 100, marker=MK[s], s=SZ[s] * 0.8, facecolor="none" if s in "ABC" else COL[s], edgecolor=COL[s], lw=1.4, zorder=3)
    a1.plot(lim, lim, color=INK, lw=1); a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal"); a1.grid(color=GRID, lw=0.7); a1.set_axisbelow(True)
    a1.set_xlabel("정답 Ψ (TRISCO, W/(m·K))", fontsize=10, color=MUTED)
    if j == 0: a1.set_ylabel("예측 Ψ", fontsize=10.5, color=MUTED)
    ta = tab[(tab.모델 == MLAB[m])].set_index("시나리오")
    a1.set_title(f"{MLAB[m]}   (A 기준 케이스 최대오차 {ta.loc['A', '케이스단위 최대절대오차율 %']:.1f} %)", loc="left", fontsize=12.5, color=INK)
    a2.axhline(0, color=INK, lw=0.9); a2.set_xticks(xs); a2.set_xticklabels([short(c) for c in cases], rotation=70, ha="right", fontsize=8.2, color=INK); a2.set_ylim(-16, 16); a2.grid(axis="y", color=GRID, lw=0.7); a2.set_axisbelow(True)
    if j == 0: a2.set_ylabel("부호 있는 오차 (%)", fontsize=10.5, color=MUTED)
    for a_ in (a1, a2):
        a_.set_facecolor(SURF)
        for sp_ in ("top", "right"): a_.spines[sp_].set_visible(False)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, fontsize=11.5, bbox_to_anchor=(0.99, 0.88))
fig.savefig(os.path.join(ROOT, "results", "fig_psi_pred_by_model.png"), dpi=125, facecolor=SURF); plt.close(fig)
# ---------------- 그림 2: 모사로 바꾸면 예측이 얼마나 달라지나 (케이스 x 모델, 부호 있는 변화 %p)
fig, axs = plt.subplots(1, 3, figsize=(19, 6.8), facecolor=SURF, gridspec_kw={"wspace": 0.22, "left": 0.12, "right": 0.97, "top": 0.8, "bottom": 0.09})
fig.text(0.02, 0.945, "VOLTRA 대신 모사를 쓰면 케이스별 예측 Ψ가 얼마나 달라지나 (모델별, 평균 없음)", fontsize=20, weight="bold", color=INK)
fig.text(0.02, 0.89, "값 = (시나리오 예측 - A 예측) / 정답 x 100, %p. 파랑은 모사 쪽이 낮게, 빨강은 높게 예측. 독립 테스트 10케이스, 일사 보정 0", fontsize=11.5, color=MUTED)
for ax, s in zip(axs, "BCD"):
    M = np.array([[(P(m, s)[c] - P(m, "A")[c]) / act[c] * 100 for m in MODELS] for c in cases])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([MLAB[m] for m in MODELS], fontsize=11); ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([short(c) for c in cases] if s == "B" else [""] * len(cases), fontsize=10)
    for i in range(M.shape[0]):
        for j in range(5): ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=9.5, color="white" if abs(M[i, j]) > 1.8 else INK)
    ax.set_title(f"{SC[s]} - A", loc="left", fontsize=12.5, color=INK); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
cb = fig.colorbar(im, ax=axs, fraction=0.018, pad=0.015); cb.set_label("%p", fontsize=10)
fig.savefig(os.path.join(ROOT, "results", "fig_psi_delta_by_model.png"), dpi=125, facecolor=SURF); plt.close(fig)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(tab.round(2).to_string(index=False))
