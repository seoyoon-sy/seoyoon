"""시간 적분 가중(theta)을 바꿔 VOLTRA 비난방 결과(난방 시작 구간)에 맞는지 시험. theta=0.5 가 Crank-Nicolson(현재), 1 은 완전 음해법.
경계 항(대류, 복사, 외기)과 전도 연산자에 같은 theta 를 쓴다. 사용: python examples/diagnose_theta.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import parse_sheet, SIM_POS_VOLTRA_ORDER, night_mask
R = lambda *p: os.path.join(ROOT, *p)
cache = pickle.load(open(R("results", "_abs_cache.pkl"), "rb"))
def run_theta(s, bc, nT, th):
    Isol = bc.irradiance(); s.solar_irradiance = Isol; G, b = s._bc_terms(bc, Isol); Kst = (s.Kc + s.Krad).tocsr(); Cd = s.cap / bc.dt
    C0 = s._conv_coupling_k(0); T = spl.spsolve((Kst + C0 + sp.diags(G[0])).tocsc(), b[0]); out = np.zeros((nT, s.N)); out[0] = T; Cp = C0
    for k in range(1, nT):
        Ck = s._conv_coupling_k(k)
        rhs = Cd * T - (1 - th) * ((Kst + Cp) @ T + G[k - 1] * T) + (1 - th) * b[k - 1] + th * b[k]
        T = spl.splu((sp.diags(Cd + th * G[k]) + th * (Kst + Ck)).tocsc()).solve(rhs); out[k] = T; Cp = Ck
    return out
def series(s, out, nm): ix, iy = s.m.outputs[nm]; return out[:, s.idx[ix, iy]]
if __name__ == "__main__":
    cond = sys.argv[1] if len(sys.argv) > 1 else "비난방"; NH = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ti = "Wall_TI_2.FTE" if cond == "비난방" else "Wall_TI.FTE"
    bc = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
    pick = ["단열재 조정 0.01+결로 방지 단열재 11.2", "결로 방지 단열재 0.0001", "결로 방지 단열재 0.005", "결로 방지 단열재 0.035", "결로 방지 단열재 1.6", "단열재 조정 나등급(0.04)"]
    print(f"{cond}: 시간 가중 theta 별 0 ~ 150 mm 평균 오차 / 평균절대 / 최대절대 (h66 ~ h130), h67 0mm, h90 0mm")
    for n in pick:
        eps, xps = parse_sheet(n); lam = {"EPS": eps if eps is not None else 0.032}
        if xps is not None: lam["XPS"] = xps
        s = Solver(try3_voltra_mesh_model(lam)); V = cache[(cond, n)]["V_surf"][:, :NH]; print(n)
        for th in (0.5, 0.6, 0.75, 1.0):
            out = run_theta(s, bc, NH, th); M = np.array([series(s, out, f"surf_{p}") for p in SIM_POS_VOLTRA_ORDER]); e = (M - V)
            ee = e[:9, 66:131]; print(f"   theta {th:4.2f} 평균 {ee.mean():+.3f} 평균절대 {abs(ee).mean():.3f} 최대절대 {abs(ee).max():.3f}  h67 0mm {e[0,67]:+.3f}  h90 0mm {e[0,90]:+.3f}", flush=True)
