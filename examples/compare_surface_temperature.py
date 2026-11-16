"""표면온도 자체(절대값 ℃)로 VOLTRA 와 hbcalc 를 비교한다 (차이가 아님).
산출물: results/surface_temp_absolute.xlsx (표), results/fig_*.png (그림)

사용: python examples/compare_surface_temperature.py [출력폴더]
  난방(20도 고정): Wall_TI.FTE + voltra_30cases_night420_통합_v2.xlsx
  비난방        : Wall_TI_2.FTE + Non_heating_Tsi.xlsx
  모사는 VOLTRA 격자, 일사 시각 보정 0 과 -0.25 시간 두 가지.
  엑셀 열 이름 10~40 mm 는 .vtr 출력 노드 순서에 따라 모사 5~30 mm 와 대응시킨다(미확정 가정).
"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.validate import (EXCEL_LABELS, SIM_POS_VOLTRA_ORDER, load_cases, parse_sheet, night_mask)
from hbcalc.presets import try3_voltra_mesh_model, try3_excel_strip_props

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_abs_cache.pkl")
SHIFTS = (0.0, -0.25)
SETS = {"난방": ("Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"),
        "비난방": ("Wall_TI_2.FTE", "Non_heating_Tsi.xlsx")}
N = 744


_BC = {}
def _bcs(ti):
    if ti not in _BC:
        _BC[ti] = {sh: Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=sh) for sh in SHIFTS}
    return _BC[ti]
def _solve(task):
    key, nm, ti, d = task
    eps, xps = parse_sheet(nm)
    lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    rho_, cp_ = try3_excel_strip_props(xps); model = try3_voltra_mesh_model(lam).with_properties(rho={"XPS": rho_}, cp={"XPS": cp_})
    s = Solver(model); bcs = _bcs(ti)
    for sh in SHIFTS:
        s.run(bcs[sh])
        d[f"M_surf_{sh}"] = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER])
        d[f"M_back_{sh}"] = np.array([s.series(f"back_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER])
    return (key, nm), d
def compute():
    import multiprocessing as mp
    tasks = []
    for key, (ti, xlsx) in SETS.items():
        xl, names = load_cases(os.path.join(ROOT, xlsx))
        for nm in names:
            df = xl.parse(nm)
            d = dict(V_surf=np.array([df[f"{p}mm 표면온도(℃)"].values[:N] for p in EXCEL_LABELS]),
                     V_back=np.array([df[f"{p}mm 석고보드 벽체 쪽 표면온도(℃)"].values[:N] for p in EXCEL_LABELS]),
                     T_out=df["실외 온도(℃)"].values[:N], T_in=df["실내 온도(℃)"].values[:N])
            tasks.append((key, nm, ti, d))
    with mp.Pool(4) as pool:
        data = {}
        for k, d in pool.imap(_solve, tasks, chunksize=1):
            data[k] = d; print(k[0], k[1][:40], flush=True)
    return data


def night_mean(a):
    return a[:, night_mask(N)].mean(1)


def main():
    if os.path.exists(CACHE):
        data = pickle.load(open(CACHE, "rb"))
    else:
        data = compute(); pickle.dump(data, open(CACHE, "wb"))
    # ---------------- 엑셀 표 (절대 온도)
    cols_pos = [f"{l}mm 열 (모사 {s}mm)" for l, s in zip(EXCEL_LABELS, SIM_POS_VOLTRA_ORDER)]
    with pd.ExcelWriter(os.path.join(OUT, "surface_temp_absolute.xlsx")) as w:
        for key in SETS:
            for kind, lab in (("surf", "표면"), ("back", "석고보드 벽체쪽")):
                rows = []
                for (k, nm), d in data.items():
                    if k != key: continue
                    for tag, arr in (("VOLTRA", d[f"V_{kind}"]), ("모사 보정0", d[f"M_{kind}_0.0"]), ("모사 보정-0.25", d[f"M_{kind}_-0.25"])):
                        rows.append([nm, tag, *night_mean(arr)])
                pd.DataFrame(rows, columns=["케이스", "구분", *cols_pos]).round(3).to_excel(
                    w, sheet_name=f"{key}_{lab}_야간평균", index=False)
        # 정상상태(첫 시각, 일사 없음) 절대 온도: 시간 요소가 없는 비교
        for key in SETS:
            rows = []
            for (k, nm), d in data.items():
                if k != key: continue
                rows.append([nm, "VOLTRA", *d["V_surf"][:, 0]]); rows.append([nm, "모사", *d["M_surf_0.0"][:, 0]])
            pd.DataFrame(rows, columns=["케이스", "구분", *cols_pos]).round(3).to_excel(
                w, sheet_name=f"{key}_정상상태_t0_표면", index=False)
    print("표 저장")
    plots(data)


def plots(data):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    INK, B, O = "#0b0b0b", "#2a78d6", "#eb6834"      # VOLTRA(검정 점), 보정0(파랑), 보정-0.25(주황)
    x = np.arange(12); xt = [f"{l}\n({s})" if l != s else f"{l}" for l, s in zip(EXCEL_LABELS, SIM_POS_VOLTRA_ORDER)]

    # ---- 그림: 모사 대 VOLTRA 산점도 (모든 케이스 x 위치, 야간 평균)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, sh, c in zip(axs, ("0.0", "-0.25"), (B, O)):
        for key, mk in (("난방", "o"), ("비난방", "^")):
            V = np.concatenate([night_mean(d["V_surf"]) for (k, _), d in data.items() if k == key])
            M = np.concatenate([night_mean(d[f"M_surf_{sh}"]) for (k, _), d in data.items() if k == key])
            ax.plot(V, M, mk, color=c, ms=4, alpha=0.55, label=f"{key} ({len(V)}점)")
        lim = np.array([14.6, 19.9]); ax.plot(lim, lim, "-", color=INK, lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal"); ax.set_xlabel("VOLTRA 야간 평균 표면온도 (℃)"); ax.set_ylabel("모사 (℃)")
        ax.set_title(f"일사 보정 {sh} 시간", loc="left", fontsize=12, color=INK); ax.legend(frameon=False, loc="upper left")
    fig.suptitle("모사 대 VOLTRA, 케이스 x 위치별 야간 평균 표면온도 (검은 선: 일치)", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_abs_scatter.png"), dpi=150); plt.close(fig)
    print("그림 저장")


if __name__ == "__main__":
    main()
