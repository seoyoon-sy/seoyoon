"""VOLTRA 의 형상계수 하한(vwfc = 1e-4, 이보다 작은 값은 버리고 나머지는 하늘로) 가설 시험: 3D 정상상태(z 20조각 x 50 mm, 조각별 형상계수)에서
cutoff 를 0 과 1e-4 로 바꿔 G3, G4 의 VOLTRA 정상상태(열류, 중간면 z=10 표면온도)와 비교한다.
사용: python examples/diagnose_vf_cutoff_3d.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np
import steady3d_test as S
from diagnose_steady_G import model, TGT, bc, POS
from hbcalc.radiation import grey_network
import scipy.sparse as sp, scipy.sparse.linalg as spl

def solve(key, nz=20, cutoffs=(0.0, 1e-4)):
    m = model(key); zv = {c: None for c in ()}
    s, E, W3, Kc3, Kz, ext, hk, A2 = S.steady3d(m, None, nz=nz)
    N = s.N; nE = len(E)
    A3 = np.array([A2[i] * hk[k] for i in range(nE) for k in range(nz + 1)])
    E3 = [E[i] for i in range(nE) for k in range(nz + 1)]
    node3 = np.array([E[i]["node"] + k * N for i in range(nE) for k in range(nz + 1)])
    zone3 = np.array([E[i]["zone"] for i in range(nE) for k in range(nz + 1)])
    S3 = sp.csr_matrix((np.ones(len(node3)), (node3, np.arange(len(node3)))), shape=(N * (nz + 1), len(node3)))
    # 경계 영역별 공기온도와 대류계수 (정상상태 첫 시각)
    zvals = {}
    for z in s.zone_ids:
        kind = s.m.zones[z].kind
        zvals[z] = ((bc.T_out[0], bc.h_out[0]) if kind == "outdoor" else (bc.T_in[0], bc.h_in[0]))
    out = {}
    for c in cutoffs:
        W = W3.copy()
        if c > 0:
            F = W / A3[:, None]; keep = (F >= c) & (F.T >= c); W = np.where(keep, W, 0.0)
        Gt, Gs, vs = grey_network(E3, W, A3, m.hrb)
        Gc = np.zeros(N * (nz + 1)); b = np.zeros(N * (nz + 1))
        for z, (Ta, h) in zvals.items():
            sel = zone3 == z; Az = S3[:, np.nonzero(sel)[0]] @ A3[sel]
            Gc += h * Az; b += h * Ta * Az + Ta * (S3 @ (Gs * sel))
        P = (Kc3 + Kz + sp.diags(Gc)).tocsc(); lu = spl.splu(P)
        op = spl.LinearOperator((N * (nz + 1),) * 2, matvec=lambda x: P @ x + S3 @ (Gt @ (S3.T @ x)))
        Mp = spl.LinearOperator((N * (nz + 1),) * 2, matvec=lambda x: lu.solve(x))
        T, info = spl.cg(op, b, M=Mp, rtol=1e-11, atol=0, maxiter=500)
        flow = (S3 @ (Gt @ (S3.T @ T))) + Gc * T - b
        T2 = T.reshape(nz + 1, N); f2 = flow.reshape(nz + 1, N)
        zf = {z: -f2[:, s.zone_nodes[z]].sum() for z in s.zone_ids}
        o = [z for z in s.zone_ids if s.m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in s.zone_ids if s.m.zones[z].kind == "indoor")
        Ms = np.array([T2[nz // 2][s.idx[m.outputs[f"surf_{p}"]]] for p in POS])
        out[c] = dict(out=-zf[o], left=zf[ind[0]], right=zf[ind[1]], surf=Ms, kept=(W > 0).sum() / max((W3 > 0).sum(), 1))
    return out

if __name__ == "__main__":
    for key in ("G3_default_sym_strong", "G4_default_sym_weak"):
        r = solve(key); t = TGT[key]
        print(f"\n== {key}: VOLTRA 열류 실외 {t['out']:.3f} 좌 {t['left']:.3f} 우 {t['right']:.3f}")
        for c, v in r.items():
            e = v["surf"] - t["surf"]
            print(f"  3D cutoff {c:g}: 열류 오차 실외 {(v['out']/t['out']-1)*100:+.2f} % 좌 {(v['left']/t['left']-1)*100:+.2f} % 우 {(v['right']/t['right']-1)*100:+.2f} % | 표면 0~100 mm 평균 {e[:8].mean():+.3f} RMSE {np.sqrt((e**2).mean()):.3f} | 남은 형상계수 쌍 {v['kept']*100:.1f} %")
            print("     편차:", np.round(e, 3))
