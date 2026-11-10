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
from hbcalc.presets import try3_voltra_mesh_model

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_abs_cache.pkl")
SHIFTS = (0.0, -0.25)
SETS = {"난방": ("Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"),
        "비난방": ("Wall_TI_2.FTE", "Non_heating_Tsi.xlsx")}
N = 744


def compute():
    data = {}
    for key, (ti, xlsx) in SETS.items():
        xl, names = load_cases(os.path.join(ROOT, xlsx))
        bcs = {sh: Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in
               ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=sh)
               for sh in SHIFTS}
        for nm in names:
            eps, xps = parse_sheet(nm)
            lam = {"EPS": eps if eps is not None else 0.032}
            if xps is not None: lam["XPS"] = xps
            model = try3_voltra_mesh_model(lam)
            s = Solver(model)
            df = xl.parse(nm)
            d = dict(V_surf=np.array([df[f"{p}mm 표면온도(℃)"].values[:N] for p in EXCEL_LABELS]),
                     V_back=np.array([df[f"{p}mm 석고보드 벽체 쪽 표면온도(℃)"].values[:N] for p in EXCEL_LABELS]),
                     T_out=df["실외 온도(℃)"].values[:N], T_in=df["실내 온도(℃)"].values[:N])
            for sh in SHIFTS:
                s.run(bcs[sh])
                d[f"M_surf_{sh}"] = np.array([s.series(f"surf_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER])
                d[f"M_back_{sh}"] = np.array([s.series(f"back_{p}")[:N] for p in SIM_POS_VOLTRA_ORDER])
            data[(key, nm)] = d
            print(key, nm[:40], flush=True)
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
        # 대표 케이스 시간별 (난방, 첫 7일)
        nm0 = "결로 방지 단열재 10"
        d = data[("난방", nm0)]
        t = pd.date_range("2026-01-01", periods=N, freq="h")
        ts = {"시각": t, "실외온도": d["T_out"], "실내온도": d["T_in"]}
        for i, (l, s) in enumerate(zip(EXCEL_LABELS, SIM_POS_VOLTRA_ORDER)):
            ts[f"VOLTRA {l}mm열"] = d["V_surf"][i]
            ts[f"모사(보정0) {s}mm"] = d["M_surf_0.0"][i]
            ts[f"모사(보정-0.25) {s}mm"] = d["M_surf_-0.25"][i]
        pd.DataFrame(ts).round(3).to_excel(w, sheet_name="시간별_난방_결로방지10_표면", index=False)
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

    # ---- 그림 1: 위치별 야간 평균 표면온도 (난방 4개 케이스)
    pick = ["결로 방지 단열재 0.0001", "결로 방지 단열재 0.014", "단열재 조정 0.045", "결로 방지 단열재 10"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, nm in zip(axs.ravel(), pick):
        d = data[("난방", nm)]
        v, m0, m1 = night_mean(d["V_surf"]), night_mean(d["M_surf_0.0"]), night_mean(d["M_surf_-0.25"])
        ax.plot(x, m0, "-", color=B, lw=2, label="모사 (일사 보정 0)")
        ax.plot(x, m1, "--", color=O, lw=2, label="모사 (일사 보정 -0.25시간)")
        ax.plot(x, v, "o", color=INK, ms=7, mfc="none", mew=1.6, label="VOLTRA", zorder=5)
        ax.set_title(nm, loc="left", fontsize=12, color=INK)
        ax.set_ylabel("야간 평균 실내 표면온도 (℃)")
        ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=8)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=10, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.995))
    for ax in axs[1]: ax.set_xlabel("접합부 면에서 거리 mm: 엑셀 열 이름 (모사 노드 위치)")
    fig.suptitle("야간(18~07시) 평균 실내 표면온도, 절대값 (난방, 실내 20 ℃ 고정)", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, "fig_abs_profile_heating.png"), dpi=150); plt.close(fig)

    # ---- 그림 2: 시간별 표면온도 (대표 케이스, 1월 첫 7일)
    nm = "결로 방지 단열재 10"; d = data[("난방", nm)]
    h = np.arange(7 * 24)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, i in zip(axs, (0, 4, 11)):
        ax.plot(h, d["M_surf_0.0"][i][:h.size], "-", color=B, lw=1.6, label="모사 (보정 0)")
        ax.plot(h, d["M_surf_-0.25"][i][:h.size], "--", color=O, lw=1.6, label="모사 (보정 -0.25)")
        ax.plot(h, d["V_surf"][i][:h.size], ".", color=INK, ms=4, label="VOLTRA")
        ax.set_ylabel("표면온도 (℃)")
        ax.set_title(f"엑셀 {EXCEL_LABELS[i]} mm 열 (모사 노드 {SIM_POS_VOLTRA_ORDER[i]} mm)", loc="left", fontsize=11, color=INK)
    axs[0].legend(frameon=False, ncol=3, fontsize=9, loc="lower right")
    axs[-1].set_xlabel("1월 1일 0시부터 경과 시간 (h)"); axs[-1].set_xticks(np.arange(0, 169, 24))
    fig.suptitle(f"시간별 실내 표면온도, 절대값 ({nm}, 난방)", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_abs_timeseries.png"), dpi=150); plt.close(fig)

    # ---- 그림 3: 모사 대 VOLTRA 산점도 (모든 케이스 x 위치, 야간 평균)
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
