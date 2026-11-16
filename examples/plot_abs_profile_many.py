"""난방 39케이스 모사 - VOLTRA 오차(야간 평균) 한눈에 보기. 모든 케이스를 그린다.
데이터: results/surface_temp_absolute.xlsx (난방, 모사 보정 0 = 현재 기본 설정). 모든 절대온도 패널은 y축 폭이 같다(5.0 도)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF, BLUE, BAND = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb", "#2f6fd0", "#dfe9f7"
d = pd.read_excel("results/surface_temp_absolute.xlsx", sheet_name="난방_표면_야간평균")
cols = [c for c in d.columns if "열" in c]
V = d[d["구분"] == "VOLTRA"].set_index("케이스")[cols]; M = d[d["구분"] == "모사 보정0"].set_index("케이스")[cols]
E = M - V
order = (V.max(axis=1) - V.min(axis=1)).sort_values().index.tolist()
XT = ["0", "10\n(5)", "20\n(10)", "30\n(20)", "40\n(30)", "50", "75", "100", "150", "200", "300", "500"]
x = np.arange(12); SPAN, ELIM, TARGET = 5.0, 0.2, 0.05
short = lambda n: n.replace("결로 방지 단열재", "결로방지").replace("단열재 조절", "조정").replace("단열재 조정", "조정")
def style(ax, show_x):
    ax.set_facecolor(SURF); ax.set_xticks(x); ax.set_xticklabels(XT if show_x else [], fontsize=7.5, color=MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9cdd1")
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0); ax.tick_params(axis="x", length=3, color="#c9cdd1")
    ax.set_axisbelow(True)

def overview(path):
    nr, nc = 5, 8
    fig, axs = plt.subplots(nr, nc, figsize=(20, 11.5), facecolor=SURF, sharey=True, gridspec_kw={"hspace": 0.5, "wspace": 0.08})
    for k, n in enumerate(order):
        ax = axs[k // nc, k % nc]; e = E.loc[n].values
        ax.axhspan(-TARGET, TARGET, color=BAND, zorder=0); ax.axhline(0, color=MUTED, lw=0.7)
        ax.plot(x, e, color=BLUE, lw=1.6, marker="o", ms=3, zorder=2); ax.set_ylim(-ELIM, ELIM)
        ax.yaxis.set_major_locator(MultipleLocator(0.1)); ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_title(short(n), loc="left", fontsize=8, color=INK)
        ax.text(0.04, 0.04, f"평균 {np.abs(e).mean():.3f}", transform=ax.transAxes, fontsize=8, color=INK)
        style(ax, False); ax.set_xticks([0, 5, 11]); ax.set_xticklabels(["0", "50", "500"], fontsize=7, color=MUTED)
    for k in range(len(order), nr * nc): axs[k // nc, k % nc].axis("off")
    fig.suptitle(f"난방 {len(order)}케이스 모사 - VOLTRA 오차(야간 평균, ℃). 같은 y축 ±{ELIM}, 띠 = ±{TARGET}. 왼쪽 위가 열교 약함, 오른쪽 아래가 강함. x축은 접합부 면에서 0, 50, 500 mm",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.92, bottom=0.04)
    fig.savefig(path, dpi=150, facecolor=SURF); plt.close(fig)

overview("results/fig_error_39cases.png")
am = E.abs().mean(axis=1)
print("39케이스 평균 절대오차(위치 평균) 중앙값 %.3f, 0.05 이하 %d개, 최대 %s %.3f" % (am.median(), (am <= 0.05).sum(), am.idxmax(), am.max()))
