"""VOLTRA 온도(.sol 0000, z 중간)를 면 흑체 노드 망에 넣어 표면 노드 열 평형 잔차를 본다 (CLAUDE.md 8.52).
구조가 VOLTRA와 같으면 모든 노드의 잔차가 0 이어야 한다. 비교: (a) 기존 노드 방식 half, (b) 흑체 노드 망 + 대류 노드, (c) 흑체 노드 망 + 대류 면 평균.
사용: python examples/network_residual_test.py 접두어 EPS XPS"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.radiation import exchange_area
from prototype_face_network import vwf_face_W
from hbcalc.voltra_vwf import read_vwf
pre, E_, X_ = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
V = {}
for l in open(os.path.join(D, pre + "0000.sol"), "rb").read().decode("cp1252").replace("\r", "").split("\n"):
    r = l.split()
    if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); cid = m.cid; nx, ny = cid.shape; xe, ye, zt = m.xe, m.ye, m.zt
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
s = Solver(m, conv_lump="half"); s.run(bc, nsteps=1); N = s.N; F = s.faces; nf = len(F); hrb = m.hrb
T = np.zeros(N)
for (i, j, z), t in V.items():
    if z == 10 and s.idx[i, j] >= 0: T[s.idx[i, j]] = t
nodes = [((s.idx[f[0], f[1]], s.idx[f[0] + 1, f[1]]) if f[2] == "x" else (s.idx[f[0], f[1]], s.idx[f[0], f[1] + 1])) for f in F]
A = np.array([f[6] for f in F]); eps = np.array([f[4] for f in F]); zone = np.array([f[3] for f in F])
Tz = {z: float(bc.T_out[0] if m.zones[z].kind == "outdoor" else bc.T_in[0]) for z in s.zone_ids}; hz = {z: float(bc.h_out[0] if m.zones[z].kind == "outdoor" else bc.h_in[0]) for z in s.zone_ids}
ext = [((xe[f[0]], xe[f[0] + 1], ye[f[1]]) if f[2] == "x" else (ye[f[1]], ye[f[1] + 1], xe[f[0]])) for f in F]
def analytic_W():
    W = np.zeros((nf, nf))
    for ia in range(nf):
        if F[ia][2] != "x": continue
        la, ha, pa = ext[ia]
        for ib in range(nf):
            if F[ib][2] != "y" or zone[ib] != zone[ia]: continue
            lb, hb, pb = ext[ib]; nx2 = 1 if F[ib][7] == "right" else -1; ny1 = 1 if F[ia][7] == "up" else -1
            if ((la + ha) / 2 - pb) * nx2 <= 0 or ((lb + hb) / 2 - pa) * ny1 <= 0: continue
            W[ia, ib] = W[ib, ia] = exchange_area(la, ha, pa, lb, hb, pb, zt, s.zmode)
    return W
surf = np.zeros(N, bool)
for a, b in nodes: surf[a] = surf[b] = True
cond = s.Kc @ T
def residual(W, conv):
    g = eps * A * hrb / (1 - eps); R = hrb * W; Rsky = hrb * A * np.clip(1 - (W / A[:, None]).sum(1), 0, None)
    Bm = sp.lil_matrix((nf, nf)); rhs = np.zeros(nf)
    for k in range(nf):
        a, c = nodes[k]; Bm[k, k] = g[k] + Rsky[k] + R[k].sum(); rhs[k] = g[k] / 2 * (T[a] + T[c]) + Rsky[k] * Tz[zone[k]]
        for k2 in np.nonzero(R[k])[0]: Bm[k, k2] = -R[k, k2]
    B = spl.spsolve(Bm.tocsc(), rhs); r = cond.copy()
    for k in range(nf):
        a, c = nodes[k]; r[a] += g[k] / 2 * (T[a] - B[k]); r[c] += g[k] / 2 * (T[c] - B[k]); gc = hz[zone[k]] * A[k] / 2
        if conv == "node": r[a] += gc * (T[a] - Tz[zone[k]]); r[c] += gc * (T[c] - Tz[zone[k]])
        else: avg = (T[a] + T[c]) / 2; r[a] += gc * (avg - Tz[zone[k]]); r[c] += gc * (avg - Tz[zone[k]])
    return r
print(f"{pre}0000: 표면 노드 {surf.sum()}개, 내부 노드 {(~surf).sum()}개")
G, b = s.G_all[0], s.b_all[0]; r0 = cond + (s.Krad @ T) + G * T - b
print(f"  [현재 노드 방식 half] 표면 노드 열 평형 잔차: RMS {np.sqrt((r0[surf] ** 2).mean()):.4f}, 최대 {abs(r0[surf]).max():.3f} W/m")
Wv = vwf_face_W(s, read_vwf(os.path.join(D, pre + ".vwf"))); Wa = analytic_W(); xy = {s.idx[i, j]: (xe[i] * 1000, ye[j] * 1000) for i in range(nx + 1) for j in range(ny + 1) if s.idx[i, j] >= 0}
for nm, W in (("해석 VF", Wa), ("VOLTRA VF", Wv)):
    for cv in ("node", "face"):
        r = residual(W, cv); o = np.argsort(-abs(r))[:5]
        print(f"  [흑체 노드 망, {nm}, 대류 {'노드' if cv == 'node' else '면 평균'}] 표면 노드 잔차 RMS {np.sqrt((r[surf] ** 2).mean()):.4f}, 최대 {abs(r[surf]).max():.3f} W/m, 합 {r[surf].sum():+.4f}; 큰 노드 {[(round(xy[i][0]), round(xy[i][1]), round(r[i], 3)) for i in o]}")
