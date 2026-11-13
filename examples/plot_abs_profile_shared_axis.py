"""야간 평균 표면온도 위치별 프로파일을 같은 축 눈금으로 다시 그린다.
위 줄: 절대 온도. 패널마다 y축 범위는 다르지만 폭(눈금 간격)은 모두 같아서 간격을 눈으로 비교할 수 있다.
아래 줄: 모사 - VOLTRA 오차, 모든 패널이 같은 y축(+-0.2 도), 목표 +-0.05 도 띠.
데이터: results/surface_temp_absolute.xlsx (난방, 모사 보정 0 = 현재 기본 설정)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb"
BLUE = "#2f6fd0"
d = pd.read_excel("results/surface_temp_absolute.xlsx", sheet_name="난방_표면_야간평균")
cols = [c for c in d.columns if "열" in c]
XT = ["0", "10\n(5)", "20\n(10)", "30\n(20)", "40\n(30)", "50", "75", "100", "150", "200", "300", "500"]
names = ["결로 방지 단열재 0.0001", "결로 방지 단열재 0.014", "단열재 조정 0.045", "결로 방지 단열재 10"]
SPAN, ELIM, TARGET = 4.6, 0.2, 0.05
x = np.arange(len(cols))
fig, axs = plt.subplots(2, 4, figsize=(17, 8.2), facecolor=SURF, gridspec_kw={"hspace": 0.32, "wspace": 0.12})
for j, n in enumerate(names):
    s = d[d["케이스"] == n]
    V = s[s["구분"] == "VOLTRA"][cols].values[0]; M = s[s["구분"] == "모사 보정0"][cols].values[0]
    e = M - V
    a = axs[0, j]; mid = (V.max() + V.min()) / 2
    a.set_ylim(mid - SPAN / 2, mid + SPAN / 2)
    a.plot(x, M, color=BLUE, lw=2, label="모사", zorder=2)
    a.plot(x, V, "o", mfc="none", mec=INK, mew=1.6, ms=7, label="VOLTRA", zorder=3)
    a.set_title(n, loc="left", fontsize=12, color=INK)
    b = axs[1, j]
    b.axhspan(-TARGET, TARGET, color="#dfe9f7", zorder=0)
    b.axhline(0, color=MUTED, lw=0.8, zorder=1)
    b.plot(x, e, color=BLUE, lw=2, marker="o", ms=5, zorder=2)
    b.set_ylim(-ELIM, ELIM)
    b.text(0.03, 0.03, f"평균 절대오차 {np.abs(e).mean():.3f} ℃, 최대 {np.abs(e).max():.3f} ℃", transform=b.transAxes, fontsize=10, color=INK, va="bottom")
    for ax in (a, b):
        ax.set_facecolor(SURF); ax.set_xticks(x); ax.set_xticklabels(XT, fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9cdd1")
        ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0); ax.tick_params(axis="x", length=3, color="#c9cdd1")
        if j > 0 and ax is b: ax.set_yticklabels([])
    if j == 0:
        a.set_ylabel("야간 평균 표면온도 (℃)", color=MUTED, fontsize=10); b.set_ylabel("모사 - VOLTRA (℃)", color=MUTED, fontsize=10)
    b.set_xlabel("접합부 면에서 거리 mm: 엑셀 열 이름 (모사 노드 위치)", fontsize=8.5, color=MUTED)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=10, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99))
fig.suptitle("야간(18~07시) 평균 실내 표면온도: 위 절대값(패널마다 y축 위치는 다르고 눈금 폭은 모두 4.6 ℃), 아래 모사 - VOLTRA 오차(같은 축, 띠 = 목표 ±0.05 ℃)",
             x=0.01, ha="left", fontsize=13.5, color=INK)
fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.09)
fig.savefig("results/fig_abs_profile_heating_sameaxis.png", dpi=200, facecolor=SURF)
print("saved")
