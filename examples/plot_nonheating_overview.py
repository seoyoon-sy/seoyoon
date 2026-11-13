"""비난방(실내온도 변동, Wall_TI_2.FTE) 30케이스 야간 평균 표면온도 위치별 프로파일: 모사 대 VOLTRA (모든 패널 y축 폭 8 도, 눈금 2 도).
입력: results/surface_temp_absolute.xlsx 의 비난방_표면_야간평균 시트(모사 보정 0 = 현재 기본 설정)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF, BLUE = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb", "#2f6fd0"
d = pd.read_excel(os.path.join(ROOT, "results/surface_temp_absolute.xlsx"), sheet_name="비난방_표면_야간평균")
cols = [c for c in d.columns if "열" in c]
V = d[d["구분"] == "VOLTRA"].set_index("케이스")[cols]; M = d[d["구분"] == "모사 보정0"].set_index("케이스")[cols]
order = (V.max(axis=1) - V.min(axis=1)).sort_values().index.tolist()
short = lambda n: n.replace("결로 방지 단열재", "결로방지").replace("단열재 조절", "조정").replace("단열재 조정", "조정")
SPAN = 8.0; x = np.arange(12); nr, nc = 4, 8
fig, axs = plt.subplots(nr, nc, figsize=(21, 11), facecolor=SURF, gridspec_kw={"hspace": 0.5, "wspace": 0.22})
for k, n in enumerate(order):
    ax = axs[k // nc, k % nc]; v, m = V.loc[n].values, M.loc[n].values
    lo_, hi_ = min(v.min(), m.min()), max(v.max(), m.max()); assert hi_ - lo_ < SPAN - 0.3, n
    mid = (lo_ + hi_) / 2; ax.set_ylim(mid - SPAN / 2, mid + SPAN / 2); ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.plot(x, m, color=BLUE, lw=1.8, zorder=2); ax.plot(x, v, "o", mfc="none", mec=INK, mew=1.3, ms=5.5, zorder=3)
    ax.set_facecolor(SURF); ax.set_title(short(n), loc="left", fontsize=9.5, color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9cdd1")
    ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=0); ax.tick_params(axis="x", length=3, color="#c9cdd1")
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_xticks(x); ax.set_xticklabels(["0", "", "", "", "", "50", "", "", "", "", "", "500"], fontsize=7.5, color=MUTED)
    ax.text(0.97, 0.05, f"평균오차 {np.abs(m - v).mean():.3f}", transform=ax.transAxes, fontsize=8, color=INK, ha="right")
for k in range(len(order), nr * nc): axs[k // nc, k % nc].axis("off")
h = [plt.Line2D([], [], color=BLUE, lw=2), plt.Line2D([], [], marker="o", mfc="none", mec=INK, mew=1.4, ls="", ms=7)]
fig.legend(h, ["모사", "VOLTRA"], frameon=False, fontsize=10, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.992))
fig.suptitle("비난방(실내온도 변동) 30케이스 야간 평균 표면온도: 모사 대 VOLTRA (왼쪽 위 열교 약함, 오른쪽 아래 강함). 패널마다 y축 위치는 다르고 눈금 폭은 모두 8 ℃, 눈금 2 ℃",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.5, 0.01, "접합부 면에서 거리 mm (0 ~ 500, 12개 위치를 같은 간격으로 표시)", ha="center", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=0.035, right=0.995, top=0.91, bottom=0.05)
fig.savefig(os.path.join(ROOT, "results/fig_voltra_vs_hbcalc_30cases_nonheating.png"), dpi=150, facecolor=SURF)
print("saved")
