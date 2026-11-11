"""TRISCO 정상상태 표(정상상태_정답.xlsx) 재현: TRISCO 자신의 격자(TRY3_Plus.trc)와 조건 사용.
케이스별로 EPS(색 4), XPS(색 7) 열전도율만 바꾼다. 형상계수 배율 vf 를 바꿔 어느 설정이 TRISCO 와 맞는지 본다."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.compat_trisco import model_from_trc
from hbcalc.validate import parse_sheet

TRC = os.path.join(ROOT, "trisco_inputs", "TRY3_Plus.trc")
m0, zv = model_from_trc(TRC)
DT = 23.7
df = pd.read_excel(os.path.join(ROOT, "정상상태_정답.xlsx")); df.columns = ["case", "Phi", "Phi2", "P2D", "psi"]
df = df[~df.case.str.startswith("Model A")].reset_index(drop=True)
b = Boundary(np.array([20.0]), np.array([20.0]), np.array([1.0]), np.array([1.0]),
             zone_values={c: (np.array([zv[c][0]]), np.array([zv[c][1]])) for c in zv})


def run(mode):
    out = []
    for _, r in df.iterrows():
        e, x = parse_sheet(r.case)
        m = m0.with_properties(lam={"colr4": e if e is not None else 0.032, "colr7": x if x is not None else 1.6})
        zmode, scale = mode
        s = Solver(m, zmode=zmode)
        if scale != 1.0: s.set_exchange_areas(s.rad_W * scale)
        s.run(b, nsteps=1); F = s.zone_flows_into_out()
        out.append((F[1][0][0] - F[1][1][0], F[8][0][0] - F[8][1][0]))
    return np.array(out)


if __name__ == "__main__":
    for lab, mode in (("형상계수 사용(mid)", ("mid", 1.0)), ("형상계수 사용(avg)", ("avg", 1.0)), ("형상계수 0 (하늘로만)", ("mid", 0.0))):
        o = run(mode)
        e1 = (o[:, 0] / df.Phi.values - 1) * 100; e2 = (o[:, 1] / df.Phi2.values - 1) * 100
        phi = 0.5 * (o[:, 0] + o[:, 1])
        psi = (2 * phi - df.P2D.values) / DT; pe = psi - df.psi.values
        print(f"{lab:24s} Φ(우)/Φ - 1: 평균 {e1.mean():+.3f} % 범위 {e1.min():+.3f}~{e1.max():+.3f} | Φ(좌)/Φ2 - 1: 평균 {e2.mean():+.3f} % 범위 {e2.min():+.3f}~{e2.max():+.3f} | Ψ 오차 평균 {pe.mean():+.5f}, 평균절대 {np.abs(pe).mean():.5f}, 최대절대 {np.abs(pe).max():.5f}")
        if mode[1] == 0.0:
            d = df.copy(); d["Phi_h"] = o[:, 0]; d["Phi2_h"] = o[:, 1]; d["psi_h"] = psi
            d.round(4).to_csv(os.path.join(ROOT, "results", "steady_vs_trisco_trcmesh_F0.csv"), index=False, encoding="utf-8-sig")
