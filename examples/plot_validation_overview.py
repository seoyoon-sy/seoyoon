"""VOLTRA 대 모사(hbcalc) 비교 그림 3장.
1) 난방 39케이스 야간 평균 표면온도 위치별 프로파일(절대값, 모든 패널 y축 폭 6.0 도)
2) 추가 벽체 10종(N1 ~ N6 난방 20도 고정, A ~ D 변형 비난방) 위치별 프로파일
3) 추가 벽체 10종 접합부 0 mm 시간별 시계열(첫 7일)과 오차
입력: results/surface_temp_absolute.xlsx, voltra_results/test_walls/_cmp.pkl, voltra_results/_cmp.pkl
(두 pkl 은 examples/compare_test_wall_results.py, compare_variant_results.py 가 만든다). 모사는 현재 기본 설정(conv_lump='low', 일사 보정 0)."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator
from hbcalc.validate import night_mask
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF, BLUE, BAND, GRAY = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb", "#2f6fd0", "#dfe9f7", "#9aa3ab"
SPAN = 6.0; N = 744; mk = night_mask(N)
XT = ["0", "10\n(5)", "20\n(10)", "30\n(20)", "40\n(30)", "50", "75", "100", "150", "200", "300", "500"]
x = np.arange(12)
short = lambda n: n.replace("결로 방지 단열재", "결로방지").replace("단열재 조절", "조정").replace("단열재 조정", "조정")

def style(ax, xt=True, fs=8):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9cdd1")
    ax.tick_params(axis="y", labelsize=fs, colors=MUTED, length=0); ax.tick_params(axis="x", length=3, color="#c9cdd1", labelsize=fs - 0.5, colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)

def profile(ax, v, m, title, fs=8):
    mid = (max(v.max(), m.max()) + min(v.min(), m.min())) / 2
    assert max(v.max(), m.max()) - min(v.min(), m.min()) < SPAN - 0.3, title
    ax.set_ylim(mid - SPAN / 2, mid + SPAN / 2); ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.plot(x, m, color=BLUE, lw=1.8, zorder=2); ax.plot(x, v, "o", mfc="none", mec=INK, mew=1.3, ms=5.5, zorder=3)
    ax.set_xticks(x); style(ax, fs=fs)
    ax.set_title(title, loc="left", fontsize=fs + 1.5, color=INK)

def legend(fig, lw=2, y=0.995):
    h = [plt.Line2D([], [], color=BLUE, lw=lw), plt.Line2D([], [], marker="o", mfc="none", mec=INK, mew=1.4, ls="", ms=7)]
    fig.legend(h, ["모사", "VOLTRA"], frameon=False, fontsize=10, loc="upper right", ncol=2, bbox_to_anchor=(0.99, y))

# ---------------- 1) 39케이스 ----------------
d = pd.read_excel(os.path.join(ROOT, "results/surface_temp_absolute.xlsx"), sheet_name="난방_표면_야간평균")
cols = [c for c in d.columns if "열" in c]
V = d[d["구분"] == "VOLTRA"].set_index("케이스")[cols]; M = d[d["구분"] == "모사 보정0"].set_index("케이스")[cols]
order = (V.max(axis=1) - V.min(axis=1)).sort_values().index.tolist()
nr, nc = 5, 8
fig, axs = plt.subplots(nr, nc, figsize=(21, 13.5), facecolor=SURF, gridspec_kw={"hspace": 0.5, "wspace": 0.22})
for k, n in enumerate(order):
    ax = axs[k // nc, k % nc]; v, m = V.loc[n].values, M.loc[n].values
    profile(ax, v, m, short(n), fs=8)
    ax.text(0.97, 0.05, f"평균오차 {np.abs(m - v).mean():.3f}", transform=ax.transAxes, fontsize=8, color=INK, ha="right")
    last = (k // nc == nr - 1) or (k + nc >= len(order))
    ax.set_xticklabels(["0", "", "", "", "", "50", "", "", "", "", "", "500"], fontsize=7.5, color=MUTED)
for k in range(len(order), nr * nc): axs[k // nc, k % nc].axis("off")
legend(fig, y=0.992)
fig.suptitle("난방 39케이스 야간 평균 표면온도: 모사 대 VOLTRA (왼쪽 위 열교 약함, 오른쪽 아래 강함). 패널마다 y축 위치는 다르지만 눈금 폭은 모두 6.0 ℃, 눈금 1 ℃",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.text(0.5, 0.008, "접합부 면에서 거리 mm (0 ~ 500, 12개 위치를 같은 간격으로 표시)", ha="center", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=0.035, right=0.995, top=0.925, bottom=0.04)
fig.savefig(os.path.join(ROOT, "results/fig_voltra_vs_hbcalc_39cases.png"), dpi=150, facecolor=SURF); plt.close(fig)

# ---------------- 추가 벽체 데이터 ----------------
tw = pickle.load(open(os.path.join(ROOT, "voltra_results/test_walls/_cmp.pkl"), "rb"))
vr = pickle.load(open(os.path.join(ROOT, "voltra_results/_cmp.pkl"), "rb"))
DESC = {"N1_light": "N1 경량, EPS 120", "N2_heavy": "N2 중량, EPS 220", "N3_thin": "N3 얇은 단열, EPS 80", "N4_weak": "N4 약한 열교(띠 0.002)",
        "N5_noblock": "N5 블록 없음, 띠 0.5", "N6_noblock_insstrip": "N6 블록 없음, 띠 0.04",
        "A_EPS100": "A EPS 100", "B_EPS250": "B EPS 250", "C_partition250": "C 내벽 250", "D_strip20": "D 띠 20, 돌출 120"}
walls = []   # (이름, 조건, Vs, Ms)
for k in tw: walls.append((DESC[k], "난방 20 ℃", tw[k]["junction"]["Vs"], tw[k]["junction"]["Ms"], tw[k]["planar"]))
for k in ("A_EPS100", "B_EPS250", "C_partition250", "D_strip20"): walls.append((DESC[k], "비난방", vr[(k, 0.0)]["Vs"], vr[(k, 0.0)]["Ms"], None))

# ---------------- 2) 추가 벽체 위치별 프로파일 ----------------
fig, axs = plt.subplots(2, 5, figsize=(19, 8.2), facecolor=SURF, gridspec_kw={"hspace": 0.42, "wspace": 0.2})
for k, (nm, cond, Vs, Ms, pl) in enumerate(walls):
    ax = axs[k // 5, k % 5]; v, m = Vs[mk].mean(0), Ms[mk].mean(0)
    profile(ax, v, m, f"{nm}  ({cond})", fs=8.5)
    mae = np.abs(Ms - Vs)[mk].mean()
    ax.text(0.97, 0.05, f"시간별 평균오차 {mae:.3f} ℃\n0 mm 편차 {(m - v)[0]:+.3f} ℃", transform=ax.transAxes, fontsize=8.5, color=INK, ha="right")
    ax.set_xticklabels(XT, fontsize=7.5, color=MUTED)
legend(fig)
fig.suptitle("추가 벽체 10종 야간 평균 표면온도: 모사 대 VOLTRA (모든 패널 y축 눈금 폭 6.0 ℃, 눈금 1 ℃)", x=0.01, ha="left", fontsize=13, color=INK)
fig.text(0.5, 0.01, "접합부 면에서 거리 mm: 엑셀 열 이름 (모사 노드 위치)", ha="center", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=0.04, right=0.995, top=0.89, bottom=0.075)
fig.savefig(os.path.join(ROOT, "results/fig_extra_walls_profile.png"), dpi=170, facecolor=SURF); plt.close(fig)

# ---------------- 3) 접합부 0 mm 시간별 시계열(첫 7일) ----------------
H = 168; t = np.arange(H)
fig, axs = plt.subplots(4, 5, figsize=(19, 11.5), facecolor=SURF, gridspec_kw={"hspace": 0.5, "wspace": 0.2, "height_ratios": [1, 0.6, 1, 0.6]})
EL = 0.3
for k, (nm, cond, Vs, Ms, pl) in enumerate(walls):
    g, j = divmod(k, 5); a, b = axs[2 * g, j], axs[2 * g + 1, j]
    TS, tick = (6.0, 1.0) if cond.startswith("난방") else (10.5, 2.0)   # 비난방은 실내온도가 변해 변동 폭이 큼
    v, m = Vs[:H, 0], Ms[:H, 0]; mid = (max(v.max(), m.max()) + min(v.min(), m.min())) / 2
    assert max(v.max(), m.max()) - min(v.min(), m.min()) < TS - 0.3, nm
    for ax in (a, b):
        for day in range(8):
            ax.axvspan(day * 24 - 6, day * 24 + 7, color="#eef0f3", zorder=0)   # 야간 18시 ~ 07시(전날 18시부터)
    a.set_ylim(mid - TS / 2, mid + TS / 2); a.yaxis.set_major_locator(MultipleLocator(tick))
    a.plot(t, v, color=GRAY, lw=3.2, zorder=2); a.plot(t, m, color=BLUE, lw=1.3, zorder=3)
    a.set_title(f"{nm}  ({cond})", loc="left", fontsize=10, color=INK)
    e = m - v
    b.axhspan(-0.05, 0.05, color=BAND, zorder=1); b.axhline(0, color=MUTED, lw=0.7, zorder=1)
    b.plot(t, e, color=BLUE, lw=1.3, zorder=3); b.set_ylim(-EL, EL); b.yaxis.set_major_locator(MultipleLocator(0.1))
    nmask = mk[:H]
    b.text(0.02, 0.05, f"첫 7일 야간 0 mm 평균절대오차 {np.abs(e[nmask]).mean():.3f} ℃", transform=b.transAxes, fontsize=8.5, color=INK)
    for ax in (a, b):
        style(ax, fs=8); ax.set_xlim(0, H); ax.set_xticks(range(0, H + 1, 24)); ax.set_xticklabels([f"{d}일" for d in range(8)], fontsize=7.5, color=MUTED)
    if j == 0: a.set_ylabel("0 mm 표면온도 (℃)", fontsize=9, color=MUTED); b.set_ylabel("모사 - VOLTRA (℃)", fontsize=9, color=MUTED)
h = [plt.Line2D([], [], color=GRAY, lw=3.2), plt.Line2D([], [], color=BLUE, lw=1.4)]
fig.legend(h, ["VOLTRA", "모사"], frameon=False, fontsize=10, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("추가 벽체 10종 접합부 0 mm 표면온도 시계열(첫 7일): 위 절대값(난방 y축 폭 6 ℃ 눈금 1 ℃, 비난방 폭 10.5 ℃ 눈금 2 ℃), 아래 오차(띠 ±0.05 ℃). 회색 배경이 야간(18 ~ 07시)", x=0.01, ha="left", fontsize=13, color=INK)
fig.subplots_adjust(left=0.05, right=0.995, top=0.925, bottom=0.04)
fig.savefig(os.path.join(ROOT, "results/fig_extra_walls_timeseries.png"), dpi=150, facecolor=SURF); plt.close(fig)

rows = []
for nm, cond, Vs, Ms, pl in walls:
    e = (Ms - Vs)[mk]
    rows.append(dict(벽체=nm, 조건=cond, 접합부MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), MAE_0mm=np.abs(e[:, 0]).mean(), 편향_0mm=e[:, 0].mean(), 최대절대=np.abs(e).max(),
                     평면MAE=(np.abs(pl["Ms"] - pl["Vs"])[mk].mean() if pl else np.nan)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
am = (M - V).abs().mean(axis=1); print("39케이스 위치평균 MAE 중앙값 %.3f, 0.05 이하 %d개" % (am.median(), (am <= 0.05).sum()))
