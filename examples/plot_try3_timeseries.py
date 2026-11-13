"""TRY3 69케이스(난방 39, 비난방 30) 접합부 0 mm 시간별 표면온도 시계열: VOLTRA 대 모사(현재 기본 설정, 일사 보정 0), 첫 7일.
케이스마다 위 패널은 절대온도(폭 8 도, 눈금 2 도), 아래 패널은 모사 - VOLTRA 오차(폭 +-0.3 도, 띠 +-0.05 도). 회색 배경 = 야간(18 ~ 07시).
입력: results/_abs_cache.pkl (compare_surface_temperature.py 가 만든다). 출력: results/fig_ts_try3_{heating,nonheating}.png"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator
from hbcalc.validate import night_mask
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF, BLUE, GRAY, BAND, NIGHT = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb", "#2f6fd0", "#9aa3ab", "#dfe9f7", "#eef0f5"
H = 168; SPANS = {"난방": 8, "비난방": 12}; ELIM = {"난방": 0.3, "비난방": 0.6}; mk = night_mask(744); t = np.arange(H)
short = lambda n: n.replace("결로 방지 단열재", "결로방지").replace("단열재 조절", "조정").replace("단열재 조정", "조정")
cache = pickle.load(open(os.path.join(ROOT, "results", "_abs_cache.pkl"), "rb"))

def shade(ax):
    for d in range(7):
        ax.axvspan(24 * d - 6, 24 * d + 7, color=NIGHT, lw=0, zorder=0)
    ax.set_xlim(0, H); ax.set_facecolor(SURF)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9cdd1")
    ax.tick_params(labelsize=7, colors=MUTED, length=2); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

for cond, fn in (("난방", "fig_ts_try3_heating.png"), ("비난방", "fig_ts_try3_nonheating.png")):
    names = [n for (c, n) in cache if c == cond]; nc = 6; nr = (len(names) + nc - 1) // nc
    fig = plt.figure(figsize=(20, 3.4 * nr), facecolor=SURF)
    gs = fig.add_gridspec(nr * 2, nc, height_ratios=[3, 1.4] * nr, hspace=0.55, wspace=0.18, left=0.035, right=0.995, top=0.945, bottom=0.03)
    for i, nm in enumerate(names):
        d = cache[(cond, nm)]; v = d["V_surf"][0]; m = d["M_surf_0.0"][0]
        r, c = divmod(i, nc); a = fig.add_subplot(gs[2 * r, c]); b = fig.add_subplot(gs[2 * r + 1, c], sharex=a)
        shade(a); shade(b)
        mid = (max(v[:H].max(), m[:H].max()) + min(v[:H].min(), m[:H].min())) / 2
        sp = SPANS[cond]; assert max(v[:H].max(), m[:H].max()) - min(v[:H].min(), m[:H].min()) < sp - 0.3, nm
        a.set_ylim(mid - sp / 2, mid + sp / 2); a.yaxis.set_major_locator(MultipleLocator(2))
        a.plot(t, v[:H], color=GRAY, lw=3.2, zorder=2); a.plot(t, m[:H], color=BLUE, lw=1.3, zorder=3)
        a.set_title(short(nm), loc="left", fontsize=10, color=INK); plt.setp(a.get_xticklabels(), visible=False)
        e = m - v
        b.axhspan(-0.05, 0.05, color=BAND, lw=0, zorder=1); b.axhline(0, color=BLUE, lw=0.6)
        b.plot(t, e[:H], color=BLUE, lw=1.0, zorder=3); el = ELIM[cond]; assert np.abs(e[:H]).max() < el, (nm, np.abs(e[:H]).max()); b.set_ylim(-el, el); b.yaxis.set_major_locator(MultipleLocator(0.1 if el < 0.4 else 0.2))
        b.set_xticks(range(0, H + 1, 24)); b.set_xticklabels([f"{k}일" for k in range(8)])
        b.text(1.0, 1.04, f"0 mm 야간 평균절대오차 {np.abs(e[mk]).mean():.3f}", transform=b.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
    fig.suptitle(f"{cond} TRY3 {len(names)}케이스 접합부 0 mm 표면온도 시계열(첫 7일): 회색 굵은선 VOLTRA, 파랑 모사. 위 절대값(폭 {SPANS[cond]} ℃, 눈금 2 ℃), 아래 오차(띠 ±0.05 ℃, 폭 ±{ELIM[cond]} ℃). 회색 배경 = 야간(18 ~ 07시)",
                 x=0.035, ha="left", fontsize=12, color=INK)
    fig.savefig(os.path.join(ROOT, "results", fn), dpi=110); plt.close(fig); print(fn, len(names))
