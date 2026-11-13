"""VOLTRA식 면 흑체 노드 망 프로토타입 (정상상태 t=0, CLAUDE.md 8.52).
구조: 면 f(선분, 방사율 e, 면적 A_f = L x 깊이)마다 흑체 노드 B_f 를 미지수로 두고 면의 두 끝 격자 노드 a, b 에 각각 g_f/2 로 연결(g_f = e A_f hrb/(1-e)).
 흑체 노드끼리는 R_ff' = hrb W_ff'(면 간 교환면적, 해석 형상계수 또는 VOLTRA .vwf), 먼 곳(영역 공기온도)과 Rsky_f = hrb A_f (1 - 합 F).
 대류는 격자 노드에 면적을 절반씩(half) 붙인 것으로 둔다(면 플럭스 합이 면 평균 법칙과 같으므로 .flx 로는 구분 안 됨).
이렇게 하면 흑체 노드를 거쳐 면의 두 끝 노드가 서로 열을 주고받는다(표면을 따르는 추가 연결 ~ A_f/(4 R_e)).
사용: python examples/prototype_face_network.py 접두어 EPS XPS [vwf]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.radiation import exchange_area
def face_network_steady(s, bc, W=None, conv="node"):
    m, N, zt, xe, ye = s.m, s.N, s.m.zt, s.m.xe, s.m.ye
    F = s.faces; nf = len(F); hrb = m.hrb
    nodes = [((s.idx[f[0], f[1]], s.idx[f[0] + 1, f[1]]) if f[2] == "x" else (s.idx[f[0], f[1]], s.idx[f[0], f[1] + 1])) for f in F]
    ext = [((xe[f[0]], xe[f[0] + 1], ye[f[1]]) if f[2] == "x" else (ye[f[1]], ye[f[1] + 1], xe[f[0]])) for f in F]
    A = np.array([f[6] for f in F]); eps = np.array([f[4] for f in F]); zone = np.array([f[3] for f in F])
    Tz = {z: float(bc.T_out[0] if m.zones[z].kind == "outdoor" else bc.T_in[0]) for z in s.zone_ids}; hz = {z: float(bc.h_out[0] if m.zones[z].kind == "outdoor" else bc.h_in[0]) for z in s.zone_ids}
    if W is None:
        W = np.zeros((nf, nf))
        for ia in range(nf):
            if F[ia][2] != "x": continue
            la, ha, pa = ext[ia]
            for ib in range(nf):
                if F[ib][2] != "y" or zone[ib] != zone[ia]: continue
                lb, hb, pb = ext[ib]; nx2 = 1 if F[ib][7] == "right" else -1; ny1 = 1 if F[ia][7] == "up" else -1
                if ((la + ha) / 2 - pb) * nx2 <= 0 or ((lb + hb) / 2 - pa) * ny1 <= 0: continue
                W[ia, ib] = W[ib, ia] = exchange_area(la, ha, pa, lb, hb, pb, zt, s.zmode)
    g = eps * A * hrb / (1 - eps); R = hrb * W; Fs = (W / A[:, None]).sum(1); Rsky = hrb * A * np.clip(1 - Fs, 0, None)
    M = N + nf; rows, cols, vals = [], [], []
    def add(i, j, v): rows.append(i); cols.append(j); vals.append(v)
    rhs = np.zeros(M)
    for k in range(nf):
        a, c = nodes[k]; Bk = N + k
        for n in (a, c):
            add(n, n, g[k] / 2); add(n, Bk, -g[k] / 2); add(Bk, Bk, g[k] / 2); add(Bk, n, -g[k] / 2)
            gc = hz[zone[k]] * A[k] / 2; rhs[n] += gc * Tz[zone[k]]
            if conv == "node": add(n, n, gc)                                                   # 대류: 노드 온도로, 면적 절반
            else:
                for n2 in (a, c): add(n, n2, gc / 2)                                           # 대류: 면 양 끝 노드 평균 온도로
        add(Bk, Bk, Rsky[k] + R[k].sum()); rhs[Bk] += Rsky[k] * Tz[zone[k]]
        for k2 in np.nonzero(R[k])[0]: add(Bk, N + k2, -R[k, k2])
    Kc = s.Kc.tocoo(); rows += list(Kc.row); cols += list(Kc.col); vals += list(Kc.data)
    sol = spl.spsolve(sp.csr_matrix((vals, (rows, cols)), shape=(M, M)).tocsc(), rhs)
    return sol[:N]
def vwf_face_W(s, vwf, zmid=500.0):
    """VOLTRA .vwf(면 x z 조각)에서 hbcalc 면(선분) 간 교환면적 W (z 중간 조각 원천, z 전체 합 목적지, 상호성 대칭)"""
    import scipy.sparse as sp
    f, Fu = vwf["faces"], vwf["F"].tocoo(); n = len(f); box = np.array([x[4] for x in f]); area = np.array([x[2] for x in f])
    zone = np.zeros(n, int); k = 0
    for z in sorted(vwf["zones"]): zone[k:k + vwf["zones"][z]] = z; k += vwf["zones"][z]
    Wu = sp.csr_matrix((Fu.data * area[Fu.row], (Fu.row, Fu.col)), shape=(n, n)); Wf = Wu + Wu.T
    segs, seg_of = {}, np.zeros(n, int)
    for i, b in enumerate(box):
        key = (zone[i], "x", round(b[2], 1), round(b[0], 1), round(b[1], 1)) if b[2] == b[3] else (zone[i], "y", round(b[0], 1), round(b[2], 1), round(b[3], 1))
        seg_of[i] = segs.setdefault(key, len(segs))
    S = len(segs); mid = (np.abs(box[:, 5] - zmid) < 1e-9) | (np.abs(box[:, 4] - zmid) < 1e-9)
    P = sp.csr_matrix((np.ones(n), (seg_of, np.arange(n))), shape=(S, n)); Ff = sp.diags(1 / area) @ Wf
    Fseg = (Ff[np.where(mid)[0]] @ P.T).toarray(); src = seg_of[np.where(mid)[0]]; Fst = np.zeros((S, S)); cnt = np.zeros(S)
    for r, s_ in enumerate(src): Fst[s_] += Fseg[r]; cnt[s_] += 1
    Fst /= cnt[:, None]; keys = list(segs); length = np.array([abs(k[4] - k[3]) / 1000.0 for k in keys]); Wst = (length * s.m.zt)[:, None] * Fst; Wst = 0.5 * (Wst + Wst.T)
    xe, ye = s.m.xe * 1000, s.m.ye * 1000; F = s.faces; nf = len(F); idx = {}
    for j, k in enumerate(keys): idx[k] = j
    fm = []
    for f_ in F:
        z = f_[3]
        key = (z, "x", round(ye[f_[1]], 1), round(xe[f_[0]], 1), round(xe[f_[0] + 1], 1)) if f_[2] == "x" else (z, "y", round(xe[f_[0]], 1), round(ye[f_[1]], 1), round(ye[f_[1] + 1], 1))
        cand = [kk for kk in idx if kk[0] == key[0] and kk[1] == key[1] and abs(kk[2] - key[2]) < 0.6 and abs(kk[3] - key[3]) < 0.6 and abs(kk[4] - key[4]) < 0.6]
        fm.append(idx[cand[0]] if cand else None)
    W = np.zeros((nf, nf))
    for a in range(nf):
        for b in range(nf):
            if fm[a] is not None and fm[b] is not None and a != b: W[a, b] = Wst[fm[a], fm[b]]
    print("  .vwf 면 대응 실패:", sum(x is None for x in fm), "/", nf)
    return W
if __name__ == "__main__":
    pre, E_, X_ = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
    V = {}
    for l in open(os.path.join(D, pre + "0000.sol"), "rb").read().decode("cp1252").replace("\r", "").split("\n"):
        r = l.split()
        if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
    m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1)
    bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
    s = Solver(m, conv_lump="half"); s.run(bc, nsteps=1); Tn = s.T_all[0]
    from hbcalc.voltra_vwf import read_vwf
    Wv = vwf_face_W(s, read_vwf(os.path.join(D, pre + ".vwf")))
    variants = [("노드 방식 half(현재)", Tn), ("흑체 노드 망 + 해석 VF + 대류 노드", face_network_steady(s, bc)), ("흑체 노드 망 + VOLTRA VF + 대류 노드", face_network_steady(s, bc, W=Wv)),
                ("흑체 노드 망 + 해석 VF + 대류 면 평균", face_network_steady(s, bc, conv="face")), ("흑체 노드 망 + VOLTRA VF + 대류 면 평균", face_network_steady(s, bc, W=Wv, conv="face"))]
    xs = np.round(m.xe * 1000, 3); yi = int(np.argmin(abs(np.round(m.ye * 1000, 3) - 1480))); POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
    for nm, T0 in variants:
        keys = [k for k in V if k[2] == 10 and s.idx[k[0], k[1]] >= 0]; d = np.array([T0[s.idx[k[0], k[1]]] - V[k] for k in keys])
        right = [T0[s.idx[m.outputs[f"surf_{p}"][0], m.outputs[f"surf_{p}"][1]]] - V[(m.outputs[f"surf_{p}"][0], m.outputs[f"surf_{p}"][1], 10)] for p in POS]
        left = [T0[s.idx[int(np.argmin(abs(xs - (1000 - p)))), yi]] - V[(int(np.argmin(abs(xs - (1000 - p)))), yi, 10)] for p in (0, 50, 100, 200, 500)]
        print(f"[{nm}] 전체 RMSE {np.sqrt((d ** 2).mean()):.4f} 평균 {d.mean():+.4f} 최대 {abs(d).max():.3f} | 오른쪽 벽 0, 5, 10, 20, 30, 50, 100, 500: " + " ".join(f"{right[POS.index(p)]:+.3f}" for p in (0, 5, 10, 20, 30, 50, 100, 500)) + f" (12위치 RMSE {np.sqrt(np.mean(np.square(right))):.3f}) | 왼쪽 x=1000, 950, 900, 800, 500: " + " ".join(f"{v:+.3f}" for v in left))
