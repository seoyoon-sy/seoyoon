"""accuracy_report.py 결과(raw.pkl) 집계: 평균 오차, 위치별, 열교 강도별, ΔTs 오차의 Ψ 환산."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc.validate import EXCEL_LABELS
R = ROOT + "/"; OUT = R + "results/accuracy/"
raw = pickle.load(open(OUT + "raw.pkl", "rb"))
tab = pd.read_excel(R + "정상상태_정답.xlsx"); tab.columns = ["case", "Phi", "Phi2", "P2D", "psi"]
from hbcalc.validate import parse_sheet
def key(nm):
    e, x = parse_sheet(nm); return (e if e is not None else 0.032, x if x is not None else 1.6)
psi_by = {key(r.case): r.psi for _, r in tab.iterrows() if not r.case.startswith("Model A")}

rows = []
for (cond, nm), r in raw.items():
    d, S = r["d"], r["S"]
    V0 = S[:, 0].mean(); dTs_V = r["planar"] - V0
    dTs_err = -(d[:, 0].mean())                      # 모사 ΔTs - VOLTRA ΔTs = -(0 mm 평균편차)
    rows.append(dict(조건=cond, 케이스=nm, MAE=np.abs(d).mean(), RMSE=np.sqrt((d**2).mean()), 편향=d.mean(),
                     MAE_0mm=np.abs(d[:, 0]).mean(), 편향_0mm=d[:, 0].mean(), 최대절대=np.abs(d).max(),
                     MAE_열교부0_100=np.abs(d[:, :8]).mean(), MAE_평면부150_500=np.abs(d[:, 8:]).mean(),
                     VOLTRA_dTs=dTs_V, dTs_오차=dTs_err, psi_정답=psi_by.get(r["key"], np.nan)))
T = pd.DataFrame(rows)
# Ψ 환산: 난방 VOLTRA ΔTs -> TRISCO Ψ 3차 다항 (8.17 방법)
H = T[(T.조건 == "난방") & T.psi_정답.notna()]
cf = np.polyfit(H.VOLTRA_dTs, H.psi_정답, 3); f = np.poly1d(cf)
T["psi_VOLTRA곡선"] = f(T.VOLTRA_dTs); T["psi_모사곡선"] = f(T.VOLTRA_dTs + T.dTs_오차)
T["psi_환산오차"] = T.psi_모사곡선 - T.psi_VOLTRA곡선
T["psi_환산오차율%"] = 100 * T.psi_환산오차 / T.psi_VOLTRA곡선
T["열교"] = pd.cut(T.VOLTRA_dTs, [-1, 1, 3, 99], labels=["약함(ΔTs<1)", "중간(1~3)", "강함(≥3)"])
T.to_csv(OUT + "per_case.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print("Ψ 곡선 잔차 RMSE", np.sqrt(((H.psi_정답 - f(H.VOLTRA_dTs))**2).mean()).round(4))
for c in ("난방", "비난방"):
    t = T[T.조건 == c]
    print(f"\n== {c} {len(t)}개: MAE 평균 {t.MAE.mean():.4f} (최소 {t.MAE.min():.3f} 최대 {t.MAE.max():.3f}), RMSE 평균 {t.RMSE.mean():.4f}, "
          f"MAE<=0.05 {int((t.MAE<=0.05).sum())}개, 0 mm MAE 평균 {t.MAE_0mm.mean():.4f}, 최대절대 평균 {t.최대절대.mean():.3f} 최대 {t.최대절대.max():.3f}")
    print(t.groupby("열교", observed=True).agg(개수=("MAE", "size"), MAE=("MAE", "mean"), MAE_0mm=("MAE_0mm", "mean"), 편향_0mm=("편향_0mm", "mean"),
                                         dTs오차=("dTs_오차", "mean"), dTs오차절대=("dTs_오차", lambda x: np.abs(x).mean())).round(4))
    print("Ψ 환산: 평균절대 %.4f, 최대절대 %.4f, 상대 평균절대 %.2f %%, 최대 %.2f %%, 3%% 초과 %d개" % (
        t.psi_환산오차.abs().mean(), t.psi_환산오차.abs().max(), t["psi_환산오차율%"].abs().mean(), t["psi_환산오차율%"].abs().max(), int((t["psi_환산오차율%"].abs() > 3).sum())))
    print(t.groupby("열교", observed=True)["psi_환산오차율%"].agg(lambda x: np.abs(x).mean()).round(2).to_dict())
# 위치별
P = []
for c in ("난방", "비난방"):
    ds = [r["d"] for (cc, _), r in raw.items() if cc == c]
    D = np.concatenate(ds)
    P.append(pd.DataFrame({"조건": c, "엑셀열(mm)": EXCEL_LABELS, "평균편차": D.mean(0), "MAE": np.abs(D).mean(0), "RMSE": np.sqrt((D**2).mean(0))}))
P = pd.concat(P); P.to_csv(OUT + "per_position.csv", index=False, encoding="utf-8-sig")
print("\n", P.round(3).to_string(index=False))
# 시간대
for c in ("난방", "비난방"):
    D = np.stack([r["d"] for (cc, _), r in raw.items() if cc == c])         # 케이스, 420, 12
    hr = np.tile(np.arange(14), 30)
    print(c, "야간 시각별(18시 이후 0~13) 평균편차:", np.round([D[:, hr == h, :].mean() for h in range(14)], 3))
print("\n약한 열교 상세(난방, 열교 약함):"); print(T[(T.조건 == "난방") & (T.열교 == "약함(ΔTs<1)")][["케이스", "VOLTRA_dTs", "MAE", "MAE_0mm", "편향_0mm", "dTs_오차", "psi_VOLTRA곡선", "psi_환산오차", "psi_환산오차율%"]].round(4).to_string(index=False))
