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
sty = {"TRY3 난방": ("o", "#2f6fd0", 4.5), "TRY3 비난방": ("^", "#8fb3e8", 4.5),
       "추가 벽체 난방 (N1 ~ N6)": ("s", "#e8590c", 5), "추가 벽체 비난방 (N1 ~ N6)": ("D", "#2b9348", 5), "변형 A ~ D 비난방": ("P", "#7b2cbf", 6)}
fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURF); rows = []
allv = np.concatenate([np.concatenate([x.ravel() for x in v]) for v, _, _ in G.values()]); lim = np.array([np.floor(allv.min()) - 0.3, np.ceil(allv.max()) + 0.3])
for g, (v, m, n) in G.items():
    V, M = np.concatenate([x.ravel() for x in v]), np.concatenate([x.ravel() for x in m]); ncase = len(n); mkr, c, ms = sty[g]
    axs[0].plot(V, M, mkr, color=c, ms=2.2, alpha=0.18, mew=0, rasterized=True, label=f"{g} ({len(n)}케이스, {V.size:,}점)")
    axs[1].plot(V, M - V, mkr, color=c, ms=2.2, alpha=0.18, mew=0, rasterized=True)
    e = M - V; rows.append(dict(그룹=g, 케이스수=len(n), 점수=V.size, 평균절대오차=abs(e).mean(), 최대절대오차=abs(e).max(), 평균편차=e.mean(), **{"0.05 초과 %": 100 * (abs(e) > 0.05).mean(), "0.1 초과 %": 100 * (abs(e) > 0.1).mean(), "95퍼센타일 절대오차": np.percentile(abs(e), 95)}))
axs[0].plot(lim, lim, "-", color=INK, lw=1); axs[0].set_xlim(lim); axs[0].set_ylim(lim); axs[0].set_aspect("equal")
axs[0].set_xlabel("VOLTRA 야간 시간별 표면온도 (℃)"); axs[0].set_ylabel("모사 (℃)"); lg = axs[0].legend(frameon=False, loc="upper left", fontsize=8.5, markerscale=1)
[h.set_alpha(1) for h in lg.legend_handles]
axs[0].set_title("1:1 비교 (검은 선: 일치)", loc="left", fontsize=12, color=INK)
axs[1].axhspan(-0.05, 0.05, color="#dfe9f7", lw=0, zorder=0); axs[1].axhline(0, color=INK, lw=0.8)
axs[1].set_xlim(lim); axs[1].set_ylim(-0.6, 0.6); axs[1].set_xlabel("VOLTRA 야간 시간별 표면온도 (℃)"); axs[1].set_ylabel("모사 - VOLTRA (℃), 띠 = ±0.05")
axs[1].set_title("같은 점의 오차", loc="left", fontsize=12, color=INK)
for ax in axs:
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.suptitle("모사 대 VOLTRA, 케이스 x 위치 x 야간 시간별 표면온도 (현재 기본 설정, 일사 보정 0, 점 하나 = 1시간)", x=0.01, ha="left", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(R("results", "fig_scatter_hourly.png"), dpi=130, facecolor=SURF)
pd.DataFrame(rows).round(4).to_csv(R("results", "accuracy", "scatter_hourly_summary.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).round(3).to_string(index=False)); print("범위", lim)
