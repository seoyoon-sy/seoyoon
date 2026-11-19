"""모사 대 VOLTRA 산점도(케이스 x 위치 x 야간 시간별 표면온도, 평균 아님): TRY3 난방 39, 비난방 30케이스에 추가 벽체(N1 ~ N6 난방과 비난방, 변형 A ~ D 비난방)를 함께 그린다.
왼쪽 1:1 산점도, 오른쪽은 같은 점의 오차(모사 - VOLTRA) 대 VOLTRA 온도. 모사는 현재 기본 설정(conv_lump='auto', 일사 보정 0).
입력: results/_abs_cache.pkl, voltra_results/test_walls/*_junction.txt, voltra_results/test_walls_nonheating/*_junction.txt, voltra_results/{A_EPS100,..}.txt
출력: results/fig_scatter_hourly.png, results/accuracy/scatter_hourly_summary.csv"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from dataclasses import replace
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report
import test_walls_def as T
import model_a_def
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb"
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); N = 744; mk = night_mask(N)
R = lambda *p: os.path.join(ROOT, *p)
bcs = {k: Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
       for k, ti in (("난방", "Wall_TI.FTE"), ("비난방", "Wall_TI_2.FTE"))}
nm = lambda a: a[mk]                                  # 야간 시간별 값 (420, 12)
G = {}                                                    # 그룹 -> (V (n,12), M (n,12), 케이스 이름들)
def add(g, name, V, M):
    v, m, n = G.setdefault(g, ([], [], [])); v.append(V); m.append(M); n.append(name)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb"))
for (cond, name), d in cache.items():
    add(f"TRY3 {cond}", name, nm(d["V_surf"][:, :N].T), nm(d["M_surf_0.0"][:, :N].T))
def run(model, bc):
    s = Solver(model); s.run(bc, nsteps=N); return np.array([s.series(f"surf_{p}")[:N] for p in POS]).T
for key in T.WALLS:
    b = T.build(key)
    for cond, d in (("난방", "voltra_results/test_walls"), ("비난방", "voltra_results/test_walls_nonheating")):
        f = R(d, f"{key}_junction.txt")
        if os.path.exists(f):
            a = read_report(f)[1][:N, 7:]; add(f"추가 벽체 {cond} (N1 ~ N6)", f"{key} {cond}", nm(a[:, 0::2]), nm(run(b["mj"], bcs[cond])))
L = lambda eps=180: [Layer("concrete", 200), Layer("EPS", eps), Layer("0.5B cement block", 90), Layer("gypsum", 10)]
VAR = {"A_EPS100": dict(layers=L(100)), "B_EPS250": dict(layers=L(250)), "C_partition250": dict(partition_mm=250), "D_strip20": dict(strip_mm=20, strip_extend_mm=120)}
for key, kw in VAR.items():
    a = read_report(R("voltra_results", f"{key}.txt"))[1][:N, 7:]
    add("변형 A ~ D 비난방", key, nm(a[:, 0::2]), nm(run(build_tjunction(replace(try3_spec(snap_mm=1), **kw)), bcs["비난방"])))
ip = R("results", "accuracy", "independent_test.pkl")
if os.path.exists(ip):
    dd = pickle.load(open(ip, "rb"))
    for cond, g in (("heating", "독립 테스트 T1 ~ T8 난방"), ("nonheating", "독립 테스트 T1 ~ T4 비난방")):
        for (nme, c_), d_ in sorted(dd.items()):
            if c_ == cond: add(g, nme, nm(d_["V"]), nm(d_["voltra"]))
# Model A (사용자 정의 파일 TRY3_INS_160_ver2.vtr, 난방, TRY3 난방 그룹에 포함): 접합부 모사와 VOLTRA 엑셀 시트 'Model A'의 12위치 표면온도
_ma = model_a_def.get(0.0); _xa = pd.ExcelFile(R("voltra_30cases_night420_통합_v2.xlsx")).parse("Model A")
_Va = np.array([_xa[f"{p}mm 표면온도(℃)"].values[:N] for p in (0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500)]).T
add("TRY3 난방", "Model A", nm(_Va), nm(_ma["M_surf"].T))          # TRY3 난방 그룹(39 + Model A = 40케이스)
sty = {"TRY3 난방": ("o", "#2f6fd0", 4.5), "TRY3 비난방": ("^", "#8fb3e8", 4.5),
       "추가 벽체 난방 (N1 ~ N6)": ("s", "#e8590c", 5), "추가 벽체 비난방 (N1 ~ N6)": ("D", "#2b9348", 5), "변형 A ~ D 비난방": ("P", "#7b2cbf", 6),
       "독립 테스트 T1 ~ T8 난방": ("X", "#c2185b", 5), "독립 테스트 T1 ~ T4 비난방": ("*", "#00897b", 6)}
def metr(V, M):
    e = M - V
    return dict(MAE=abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), r=np.corrcoef(V, M)[0, 1], R2=1 - (e ** 2).sum() / ((V - V.mean()) ** 2).sum(), MAPE=100 * (abs(e) / abs(V)).mean(), 최대=abs(e).max(), g05=100 * (abs(e) > 0.05).mean(), g10=100 * (abs(e) > 0.1).mean())
fig = plt.figure(figsize=(14, 10.4), facecolor=SURF); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.46], hspace=0.28, left=0.06, right=0.985, top=0.93, bottom=0.025)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]; axT = fig.add_subplot(gs[1, :]); axT.axis("off"); rows = []
allv = np.concatenate([np.concatenate([x.ravel() for x in v]) for v, _, _ in G.values()]); lim = np.array([np.floor(allv.min()) - 0.3, np.ceil(allv.max()) + 0.3])
VA, MA = [], []
for g, (v, m, n) in G.items():
    V, M = np.concatenate([x.ravel() for x in v]), np.concatenate([x.ravel() for x in m]); mkr, c, ms = sty[g]; VA.append(V); MA.append(M)
    axs[0].plot(V, M, mkr, color=c, ms=2.2, alpha=0.18, mew=0, rasterized=True, label=f"{g} ({len(n)}케이스)")
    axs[1].plot(V, M - V, mkr, color=c, ms=2.2, alpha=0.18, mew=0, rasterized=True)
    e = M - V; rows.append(dict(그룹=g, 케이스수=len(n), 점수=V.size, **metr(V, M), 평균편차=e.mean(), **{"95퍼센타일 절대오차": np.percentile(abs(e), 95)}))
tot = metr(np.concatenate(VA), np.concatenate(MA)); rows.append(dict(그룹="전체", 케이스수=sum(len(x[2]) for x in G.values()), 점수=sum(a.size for a in VA), **tot, 평균편차=(np.concatenate(MA) - np.concatenate(VA)).mean(), **{"95퍼센타일 절대오차": np.percentile(abs(np.concatenate(MA) - np.concatenate(VA)), 95)}))
axs[0].plot(lim, lim, "-", color=INK, lw=1); axs[0].set_xlim(lim); axs[0].set_ylim(lim); axs[0].set_aspect("equal")
axs[0].set_xlabel("VOLTRA 야간 시간별 표면온도 (℃)"); axs[0].set_ylabel("모사 (℃)"); lg = axs[0].legend(frameon=False, loc="upper left", fontsize=8, markerscale=1)
[h.set_alpha(1) for h in lg.legend_handles]
axs[0].text(0.97, 0.04, f"전체 {rows[-1]['점수']:,}점\nr {tot['r']:.5f}   R² {tot['R2']:.5f}\nRMSE {tot['RMSE']:.4f} ℃   MAE {tot['MAE']:.4f} ℃\nMAPE {tot['MAPE']:.3f} %", transform=axs[0].transAxes, ha="right", va="bottom", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
axs[0].set_title("1:1 비교 (검은 선: 일치)", loc="left", fontsize=12, color=INK)
axs[1].axhspan(-0.05, 0.05, color="#dfe9f7", lw=0, zorder=0); axs[1].axhline(0, color=INK, lw=0.8)
axs[1].set_xlim(lim); axs[1].set_ylim(-0.3, 0.3); axs[1].set_xlabel("VOLTRA 야간 시간별 표면온도 (℃)"); axs[1].set_ylabel("모사 - VOLTRA (℃), 띠 = ±0.05")
axs[1].set_title("같은 점의 오차", loc="left", fontsize=12, color=INK)
for ax in axs:
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
# 아래 표: 그룹별 성능
cols = [("그룹", 0.0, "left"), ("케이스", 0.30, "right"), ("점수", 0.37, "right"), ("MAE", 0.43, "right"), ("RMSE", 0.49, "right"), ("r", 0.56, "right"), ("R²", 0.63, "right"), ("MAPE %", 0.70, "right"), ("최대", 0.76, "right"), ("0.05 초과 %", 0.86, "right"), ("0.1 초과 %", 0.96, "right")]
axT.set_xlim(0, 1); axT.set_ylim(0, 1); nr = len(rows) + 1; dy = 1.0 / (nr + 0.6)
for nme, x, ha in cols: axT.text(x if ha == "left" else x + 0.0, 1 - dy * 0.6, nme, ha=ha, va="center", fontsize=10.5, weight="bold", color=INK)
axT.plot([0, 1], [1 - dy * 1.1] * 2, color=INK, lw=1)
for k, r_ in enumerate(rows):
    y = 1 - dy * (k + 1.6); last = r_["그룹"] == "전체"
    vals = [r_["그룹"], f"{r_['케이스수']}", f"{r_['점수']:,}", f"{r_['MAE']:.4f}", f"{r_['RMSE']:.4f}", f"{r_['r']:.5f}", f"{r_['R2']:.5f}", f"{r_['MAPE']:.3f}", f"{r_['최대']:.3f}", f"{r_['g05']:.2f}", f"{r_['g10']:.2f}"]
    for (nme, x, ha), t in zip(cols, vals): axT.text(x, y, t, ha=ha, va="center", fontsize=10, color=INK, weight="bold" if last else "normal")
    if k < len(rows) - 1: axT.plot([0, 1], [y - dy / 2] * 2, color=GRID, lw=0.7)
axT.text(0, 0.0, "단위 ℃(MAE, RMSE, 최대). r = 피어슨 상관계수, R² = 1 - sum(오차²) / sum((V - 평균 V)²), MAPE = mean(|오차| / |V|) x 100 (V는 섭씨 온도라 0 기준이 임의이므로 참고용). 점 하나 = 케이스 x 위치 x 야간 1시간.", fontsize=9, color=MUTED, va="bottom")
fig.suptitle("모사 대 VOLTRA, 케이스 x 위치 x 야간 시간별 표면온도 (현재 기본 설정 voltra, 일사 보정 0, 점 하나 = 1시간)", x=0.01, ha="left", fontsize=14)
fig.savefig(R("results", "fig_scatter_hourly.png"), dpi=130, facecolor=SURF)
pd.DataFrame(rows).round(4).to_csv(R("results", "accuracy", "scatter_hourly_summary.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).round(4).to_string(index=False)); print("범위", lim)
