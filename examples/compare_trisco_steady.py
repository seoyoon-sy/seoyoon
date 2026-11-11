"""TRISCO 정상상태 결과(정상상태_정답.xlsx)와 hbcalc 정상상태 비교.
TRISCO 조건(TRY3_Plus_2D.trc): 실내 20 도 hc 1.415, 실외 -3.7 도 hc 16.03, hrb 5.1, 선형 복사.
Φ2D = 평면 벽체(접합부 없음) 열류량, Ψ = (2 Φ - Φ2D) / ΔT 로 표에서 역산됨."""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import parse_sheet

TI, TO, HI, HO, HRB = 20.0, -3.7, 1.415, 16.03, 5.1
DT = TI - TO
df = pd.read_excel(os.path.join(ROOT, "정상상태_정답.xlsx")); df.columns = ["case", "Phi", "Phi2", "P2D", "psi"]
bc = Boundary(np.array([TO]), np.array([TI]), np.array([HO]), np.array([HI]), dt=3600.0)
rows = []
for _, r in df.iterrows():
    nm = r.case
    if nm.startswith("Model A"):
        continue                                   # 구조를 알 수 없음
    e, x = parse_sheet(nm)
    lam = {"EPS": e if e is not None else 0.032}
    if x is not None: lam["XPS"] = x
    m = try3_voltra_mesh_model(lam); m.hrb = HRB
    s = Solver(m); s.run(bc, nsteps=1)
    F = s.zone_flows_into_out()
    inL, inR = F[2][0][0] - F[2][1][0], F[3][0][0] - F[3][1][0]        # 왼쪽/오른쪽 실내에서 구조로 들어오는 순열류
    out = F[1][1][0] - F[1][0][0]                                      # 실외로 나가는 순열류
    # 평면 벽체 해석해 (1D): 폭 2.2 m x 깊이 1 m
    eps_i = lam["EPS"]
    R = 1 / (HI + 0.9 * HRB) + 1 / (HO + 0.9 * HRB) + 0.2 / 1.6 + 0.18 / eps_i + 0.09 / 0.6 + 0.01 / 0.16
    p2d = DT * 2.2 / R
    rows.append(dict(case=nm, Phi_T=r.Phi, P2D_T=r.P2D, psi_T=r.psi, 실내좌=inL, 실내우=inR, 실외유출=out, P2D_h=p2d))
d = pd.DataFrame(rows)
d["Phi_h(우)"] = d["실내우"]; d["Phi_h(평균)"] = 0.5 * (d["실내좌"] + d["실내우"])
d["psi_h"] = (2 * d["Phi_h(평균)"] - d["P2D_h"]) / DT
d["psi_h(우)"] = (2 * d["Phi_h(우)"] - d["P2D_h"]) / DT
pd.set_option("display.width", 220)
print(f"ΔT = {DT} K, 케이스 {len(d)}개 (Model A 제외)")
print("Φ2D 해석해 대 TRISCO 최대 상대차 %.3f %%" % (100 * ((d.P2D_h - d.P2D_T) / d.P2D_T).abs().max()))
for c in ("Phi_h(우)", "Phi_h(평균)", "실외유출"):
    rel = (d[c] / d.Phi_T - 1) * 100
    print(f"Φ {c:10s} / TRISCO 전체열류량 - 1 [%]: 평균 {rel.mean():+.2f}, 범위 {rel.min():+.2f} ~ {rel.max():+.2f}")
for c in ("psi_h", "psi_h(우)"):
    e = d[c] - d.psi_T
    print(f"Ψ {c:10s} - TRISCO: 평균오차 {e.mean():+.4f}, 평균절대 {e.abs().mean():.4f}, 최대절대 {e.abs().max():.4f} W/(m K), 상대 평균절대 {100*(e.abs()/d.psi_T).mean():.2f} %")
d.round(4).to_csv(os.path.join(ROOT, "results", "steady_vs_trisco.csv"), index=False, encoding="utf-8-sig")
print(d[["case", "Phi_T", "Phi_h(평균)", "psi_T", "psi_h"]].round(4).iloc[::4].to_string(index=False))
