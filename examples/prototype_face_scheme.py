"""면 단위 표면 열교환 방식 프로토타입 (정상상태 t=0, CLAUDE.md 8.51).
VOLTRA .flx 분석(8.51)에서 확인: 면(선분) 플럭스 = (h_c + h_r)(T_공기 - 면 양 끝 노드 온도 평균), 면 열량은 두 끝 노드에 절반씩.
대류: 면 f(영역 z, 면적 A_f = L x 깊이)에서 노드 a, b 로 가는 항 = (h A_f / 2)(평균T - Ta) 를 두 노드에 절반씩 -> 행렬 (h A_f / 4)[[1,1],[1,1]].
복사: 요소 = 면, 요소 온도 = 양 끝 노드 평균, 면 간 교환면적 W (해석 형상계수), 회색면 흑체망 grey_network 그대로, 요소 열량을 두 끝 노드에 절반씩.
사용: python examples/prototype_face_scheme.py 접두어 EPS XPS  (VOLTRA .sol 0000 과 비교)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.radiation import exchange_area, grey_network
def face_scheme_steady(s, bc, conv=True, rad=True, vwf=None):
    m, N, zt, xe, ye = s.m, s.N, s.m.zt, s.m.xe, s.m.ye
    F = s.faces; nf = len(F)
    nodes = [((s.idx[f[0], f[1]], s.idx[f[0] + 1, f[1]]) if f[2] == "x" else (s.idx[f[0], f[1]], s.idx[f[0], f[1] + 1])) for f in F]
    ext = [((xe[f[0]], xe[f[0] + 1], ye[f[1]]) if f[2] == "x" else (ye[f[1]], ye[f[1] + 1], xe[f[0]])) for f in F]
    A = np.array([f[6] for f in F]); zone = np.array([f[3] for f in F])
    Tz = {}; hz = {}
    for z in s.zone_ids:
        kind = m.zones[z].kind
        Tz[z] = float(bc.T_out[0] if kind == "outdoor" else bc.T_in[0]); hz[z] = float(bc.h_out[0] if kind == "outdoor" else bc.h_in[0])
    rows, cols, vals = [], [], []; b = np.zeros(N)
    if conv:
        for k in range(nf):
            g = hz[zone[k]] * A[k]; a, c = nodes[k]
            for p in (a, c):
                for q in (a, c): rows.append(p); cols.append(q); vals.append(g / 4)
                b[p] += g / 2 * Tz[zone[k]]
    B = sp.csr_matrix((np.full(2 * nf, 0.5), (np.repeat(np.arange(nf), 2), np.array(nodes).ravel())), shape=(nf, N))
    Krad = sp.csr_matrix((N, N))
    if rad:
        W = np.zeros((nf, nf))
        for ia in range(nf):
            if F[ia][2] != "x": continue
            la, ha, pa = ext[ia]
            for ib in range(nf):
                if F[ib][2] != "y" or zone[ib] != zone[ia]: continue
                lb, hb, pb = ext[ib]
                nx2 = 1 if F[ib][7] == "right" else -1; ny1 = 1 if F[ia][7] == "up" else -1
                if ((la + ha) / 2 - pb) * nx2 <= 0 or ((lb + hb) / 2 - pa) * ny1 <= 0: continue
                W[ia, ib] = W[ib, ia] = exchange_area(la, ha, pa, lb, hb, pb, zt, s.zmode)
        E = [dict(eps=f[4]) for f in F]
        Gt, Gs, _ = grey_network(E, W, A, m.hrb)
        Krad = (B.T @ sp.csr_matrix(Gt) @ B).tocsr()
        for z in s.zone_ids: b += Tz[z] * (B.T @ (Gs * (zone == z)))
    K = (s.Kc + Krad + sp.csr_matrix((vals, (rows, cols)), shape=(N, N))).tocsc()
    return spl.spsolve(K, b)
if __name__ == "__main__":
    pre, E_, X_ = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
    V = {}
    for l in open(os.path.join(D, pre + "0000.sol"), "rb").read().decode("cp1252").replace("\r", "").split("\n"):
        r = l.split()
        if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
    m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); cid = m.cid
    bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
    s = Solver(m, conv_lump="half"); s.run(bc, nsteps=1); Tn = s.T_all[0]; Tf = face_scheme_steady(s, bc)
    xs = np.round(m.xe * 1000, 3); yi = int(np.argmin(abs(np.round(m.ye * 1000, 3) - 1480))); POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
    for nm, T0 in (("노드 방식 half(현재)", Tn), ("면 평균 방식(프로토타입)", Tf)):
        keys = [k for k in V if k[2] == 10 and s.idx[k[0], k[1]] >= 0]; d = np.array([T0[s.idx[k[0], k[1]]] - V[k] for k in keys])
        right = [T0[s.idx[m.outputs[f"surf_{p}"][0], m.outputs[f"surf_{p}"][1]]] - V[(m.outputs[f"surf_{p}"][0], m.outputs[f"surf_{p}"][1], 10)] for p in POS]
        left = [T0[s.idx[int(np.argmin(abs(xs - (1000 - p)))), yi]] - V[(int(np.argmin(abs(xs - (1000 - p)))), yi, 10)] for p in (0, 50, 100, 200, 500)]
        print(f"[{nm}] 전체 RMSE {np.sqrt((d ** 2).mean()):.4f} 평균 {d.mean():+.4f} 최대 {abs(d).max():.3f} | 오른쪽 벽 0, 5, 10, 20, 30, 50, 100, 500: " + " ".join(f"{right[POS.index(p)]:+.3f}" for p in (0, 5, 10, 20, 30, 50, 100, 500)) + f" (12위치 RMSE {np.sqrt(np.mean(np.square(right))):.3f}) | 왼쪽 x=1000, 950, 900, 800, 500: " + " ".join(f"{v:+.3f}" for v in left))
