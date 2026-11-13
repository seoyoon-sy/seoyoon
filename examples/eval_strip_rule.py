"""결로 방지 단열재(XPS 판) 열전도율에 따라 대류 배분 가중치 w를 바꾸는 규칙(conv_lump=(w, w))의 TRY3 69케이스 검증 (8.35).
w(lam): log10(lam)에 대한 선형 보간. 점: (1e-4, 0.5)=half(G2, G4), (0.002, 0.65)=N4, (0.04, 0.825)=N6, N5b, (0.5 이상, 1.0)=low(N1 ~ N3, N5, N6b).
기준(low)은 results/accuracy/per_case.csv. 사용: python examples/eval_strip_rule.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary
from hbcalc.validate import *

R = ROOT + "/"; mk = night_mask(744); POS = EXCEL_LABELS; SIMPOS = SIM_POS_VOLTRA_ORDER
PTS_L = np.log10([1e-4, 0.002, 0.04, 0.5]); PTS_W = [0.5, 0.65, 0.825, 1.0]
def w_of(lam): return float(np.interp(np.log10(max(lam, 1e-6)), PTS_L, PTS_W))
if __name__ == "__main__":
    rows = []
    for cond, ti, xlsx in (("난방", "Wall_TI.FTE", "voltra_30cases_night420_통합_v2.xlsx"), ("비난방", "Wall_TI_2.FTE", "Non_heating_Tsi.xlsx")):
        bc = Boundary.from_voltra_files(R + "wall_TO.FTE", R + ti, R + "Wall_HO.FHT", R + "Wall_HI.FHT", R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0)
        xl, names = load_cases(R + xlsx)
        for nm in names:
            e, x = parse_sheet(nm); lam = x if x is not None else 1.6; w = w_of(lam)
            s, m = run_case(nm, bc, mesh="voltra", positions=sorted(set(POS) | set(SIMPOS)), solver_kw=dict(conv_lump=(w, w)))
            df = xl.parse(nm); ss = {p: s.series(f"surf_{p}") for p in set(POS) | set(SIMPOS)}; bb = {p: s.series(f"back_{p}") for p in set(POS) | set(SIMPOS)}
            d, dw = compare_case(df, ss, bb, POS, SIMPOS)
            rows.append(dict(조건=cond, 케이스=nm, XPS=lam, w=w, MAE=np.abs(d).mean(), RMSE=np.sqrt((d ** 2).mean()), 편향_0mm=d[:, 0].mean(), MAE_0mm=np.abs(d[:, 0]).mean(), 최대절대=np.abs(d).max()))
            print(cond, nm, round(w, 3), round(rows[-1]["MAE"], 4), flush=True)
    new = pd.DataFrame(rows); old = pd.read_csv(R + "results/accuracy/per_case.csv", encoding="utf-8-sig")
    j = new.merge(old[["조건", "케이스", "MAE", "RMSE", "편향_0mm", "MAE_0mm", "최대절대"]], on=["조건", "케이스"], suffixes=("_규칙", "_low"))
    j.to_csv(R + "results/accuracy/strip_rule_vs_low.csv", index=False, encoding="utf-8-sig")
    for cond in ("난방", "비난방"):
        q = j[j.조건 == cond]
        print(f"\n{cond} {len(q)}케이스: 평균절대오차 규칙 {q.MAE_규칙.mean():.4f} 대 low {q.MAE_low.mean():.4f}, RMSE 규칙 {q.RMSE_규칙.mean():.4f} 대 {q.RMSE_low.mean():.4f}, "
              f"0 mm 평균절대 {q.MAE_0mm_규칙.mean():.4f} 대 {q.MAE_0mm_low.mean():.4f}, 최대절대오차 최댓값 {q.최대절대_규칙.max():.3f} 대 {q.최대절대_low.max():.3f}")
        print(f"   0.05 이하 케이스: 규칙 {(q.RMSE_규칙<=0.05).sum()}개 대 low {(q.RMSE_low<=0.05).sum()}개 (RMSE 기준), MAE 기준 {(q.MAE_규칙<=0.05).sum()} 대 {(q.MAE_low<=0.05).sum()}")
        for lo, hi, lab in ((0, 0.01, "XPS < 0.01(약함)"), (0.01, 0.5, "0.01 ~ 0.5"), (0.5, 99, "XPS >= 0.5")):
            r = q[(q.XPS >= lo) & (q.XPS < hi)]
            if len(r): print(f"   {lab:16s} {len(r):2d}개: MAE 규칙 {r.MAE_규칙.mean():.4f} 대 low {r.MAE_low.mean():.4f}, 0 mm 편향 규칙 {r.편향_0mm_규칙.mean():+.3f} 대 low {r.편향_0mm_low.mean():+.3f}")
