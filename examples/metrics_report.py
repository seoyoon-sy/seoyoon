"""모사 대 VOLTRA 정확도 지표 모음: MAE, RMSE, R2, MAPE, 최대 절대오차, 0.05 초과 비율.
대상: (1) TRY3 난방 39, 비난방 30케이스 (results/_abs_cache.pkl, 현재 기본 voltra 설정, XPS 판 밀도 25),
      (2) 독립 테스트 T1 ~ T8 (results/accuracy/independent_test.pkl, 설정 5가지).
점 단위 = 케이스 x 12위치 x 야간(18 ~ 07시) 시간별 표면온도. 지표 정의:
  R2 = 1 - sum((M - V)^2) / sum((V - mean(V))^2)   (V = VOLTRA, M = 모사)
  MAPE = mean(|M - V| / |V|) x 100   (V 는 섭씨 온도. 섭씨는 0 이 임의 기준이라 해석에 주의: 아래 ΔTs 지표가 더 의미 있다)
ΔTs 지표: 케이스별 야간 평균 (500 mm 열 - 0 mm 열)을 VOLTRA 와 모사로 구해 케이스 단위로 비교 (학습모델 입력과 같은 종류의 양).
사용: python examples/metrics_report.py   (출력 results/accuracy/metrics_summary.csv)"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc.validate import night_mask
R = lambda *p: os.path.join(ROOT, *p); N = 744; mk = night_mask(N)
SCHEMES = ["half", "low", "이전 표(손격자용)", "새 표(자동 격자용)", "voltra"]

def stats(V, M):
    V, M = np.asarray(V).ravel(), np.asarray(M).ravel(); e = M - V
    return dict(점수=V.size, MAE=abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / ((V - V.mean()) ** 2).sum(), MAPE_pct=100 * (abs(e) / abs(V)).mean(),
                최대절대=abs(e).max(), **{"0.05초과_pct": 100 * (abs(e) > 0.05).mean(), "0.1초과_pct": 100 * (abs(e) > 0.1).mean()})

def dts_stats(Vl, Ml):
    """Vl, Ml: 케이스별 (12, N) 배열 목록 -> 야간 평균 ΔTs(500 mm - 0 mm) 비교"""
    v = np.array([x[11, mk].mean() - x[0, mk].mean() for x in Vl]); m = np.array([x[11, mk].mean() - x[0, mk].mean() for x in Ml]); e = m - v
    return dict(케이스수=len(v), ΔTs_MAE=abs(e).mean(), ΔTs_RMSE=np.sqrt((e ** 2).mean()), ΔTs_R2=1 - (e ** 2).sum() / ((v - v.mean()) ** 2).sum(),
                ΔTs_MAPE_pct=100 * (abs(e) / abs(v)).mean(), ΔTs_최대절대=abs(e).max(), ΔTs_편향=e.mean())

rows = []
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb"))
for cond in ("난방", "비난방"):
    ks = [k for k in cache if k[0] == cond]
    V = np.array([cache[k]["V_surf"][:, mk] for k in ks]); M = np.array([cache[k]["M_surf_0.0"][:, mk] for k in ks])
    rows.append(dict(그룹=f"TRY3 {cond}", 설정="voltra", 범위="12위치 전체", **stats(V, M)))
    rows.append(dict(그룹=f"TRY3 {cond}", 설정="voltra", 범위="0 mm만", **stats(V[:, 0], M[:, 0])))
    rows.append(dict(그룹=f"TRY3 {cond}", 설정="voltra", 범위="케이스 ΔTs", **dts_stats([cache[k]["V_surf"] for k in ks], [cache[k]["M_surf_0.0"] for k in ks])))
ip = R("results", "accuracy", "independent_test.pkl")
if os.path.exists(ip):
    d = pickle.load(open(ip, "rb"))
    for cond, lab in (("heating", "독립 T1 ~ T8 난방"), ("nonheating", "독립 T1 ~ T4 비난방")):
        ks = sorted(k for k in d if k[1] == cond)
        for sc in SCHEMES:
            V = np.array([d[k]["V"].T[:, mk] for k in ks]); M = np.array([d[k][sc].T[:, mk] for k in ks])
            rows.append(dict(그룹=lab, 설정=sc, 범위="12위치 전체", **stats(V, M)))
            rows.append(dict(그룹=lab, 설정=sc, 범위="0 mm만", **stats(V[:, 0], M[:, 0])))
            rows.append(dict(그룹=lab, 설정=sc, 범위="케이스 ΔTs", **dts_stats([d[k]["V"].T for k in ks], [d[k][sc].T for k in ks])))
df = pd.DataFrame(rows); df.to_csv(R("results", "accuracy", "metrics_summary.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
a = df[df.범위 != "케이스 ΔTs"].dropna(axis=1, how="all"); print(a[["그룹", "설정", "범위", "점수", "MAE", "RMSE", "R2", "MAPE_pct", "최대절대", "0.05초과_pct", "0.1초과_pct"]].to_string(index=False))
b = df[df.범위 == "케이스 ΔTs"].dropna(axis=1, how="all"); print(); print(b[["그룹", "설정", "케이스수", "ΔTs_MAE", "ΔTs_RMSE", "ΔTs_R2", "ΔTs_MAPE_pct", "ΔTs_최대절대", "ΔTs_편향"]].to_string(index=False))
