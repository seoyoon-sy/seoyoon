"""3D 솔버(hbcalc.solver3d.Solver3D)를 VOLTRA 정상상태 해 파일(.sol, 모든 격자 노드 x z 평면 21개)과 비교 (CLAUDE.md 8.77).
대상: calib_steady C_E0.032_X0.04 와 X0.0001(시각 0, 정상상태), X4.3(시각 3600 s). 2D voltra 방식과 3D(z 조각 수 여러 개)를 같이 본다.
사용: python examples/compare_sol_solver3d.py"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.solver3d import Solver3D
from hbcalc.presets import try3_model
D = ROOT + "/voltra_results/calib_steady_sol/"
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)


def load(f):
    V = {}
    for l in open(f, "rb").read().decode("cp1252").replace("\r", "").split("\n"):
        r = l.split()
        if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
    return V


for X, f, k in ((0.04, "C_E0.032_X0.040000.sol", 0), (0.0001, "C_E0.032_X0.00010000.sol", 0), (4.3, "C_E0.032_X4.30001.sol", 1)):
    V = load(D + f); m = try3_model({"EPS": 0.032, "XPS": X}, snap_mm=1)
    bc = Boundary.from_voltra_files(*(ROOT + "/" + g for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), diffuse_obstruction=True)
    po = [m.outputs[f"surf_{p}"][:2] for p in POS]
    s2 = Solver(m); s2.run(bc, nsteps=k + 1); idx = s2.idx
    keys = [q for q in V if q[2] == 10 and idx[q[0], q[1]] >= 0]

    def rep(nm, Tmid, Tz=None, extra=""):
        d = np.array([Tmid[idx[q[0], q[1]]] - V[q] for q in keys]); e = np.array([Tmid[idx[i, j]] - V[(i, j, 10)] for i, j in po])
        s = f"  {nm:22s} 중간면 전체 노드 RMSE {np.sqrt((d ** 2).mean()):.4f} 최대 {abs(d).max():.3f} | 12위치 평균절대 {abs(e).mean():.4f} 최대 {abs(e).max():.3f} 0mm {e[0]:+.4f} 500mm {e[-1]:+.4f}"
        print(s + extra, flush=True)
    print(f"X{X:g} (시각 {k * 3600} s), 노드 2D {s2.N}")
    rep("2D voltra", s2.T_all[k])
    for nz in (6, 10, 20):
        t = time.time(); s3 = Solver3D(m, nz=nz); s3.run(bc, nsteps=k + 1); dtm = time.time() - t
        T = s3.T_all[k]; N = s3.N
        Tp = lambda p: T[(min(p, nz - p)) * N:(min(p, nz - p) + 1) * N]       # 접은 평면 p
        extra = ""
        if 20 % nz == 0:                      # VOLTRA 평면(50 mm 간격)과 겹치는 평면에서 끝면 비교
            zz = [p for p in range(nz // 2 + 1)]
            errs = []
            for p in zz:
                iz = p * (20 // nz); dd = np.array([Tp(p)[idx[q[0], q[1]]] - V[(q[0], q[1], iz)] for q in keys]); errs.append(np.sqrt((dd ** 2).mean()))
            extra = f" | 평면별 전체 RMSE(z=0 -> 중간) " + " ".join(f"{x:.4f}" for x in errs)
        rep(f"3D nz={nz} ({dtm:.0f}s)", Tp(nz // 2), extra=extra)
