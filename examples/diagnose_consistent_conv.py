"""가설 시험: VOLTRA 의 경계면이 격자 선분(삼각분할의 변) 단위로 처리된다면, 선분의 대류를 '질량행렬 일치'(consistent, 끝점 사이 선형 보간 적분)
방식 [[1/3,1/6],[1/6,1/3]] 으로 두 끝점에 연결하는 것과 비교한다 (기본은 끝점에 절반씩 집중 = 'half', 또는 'low').
정상상태 t=0, 시험 벽체 6종 표면온도 편차(모사 - VOLTRA)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_test_walls import *
import scipy.sparse as sp, scipy.sparse.linalg as spl

def consistent_dev(key, wc=1.0):
    """wc: 0 이면 집중(half), 1 이면 consistent, 사이 값은 혼합."""
    s = Solver(B[key]["mj"], conv_lump="half"); s.run(bc, nsteps=1)
    n = s.N; rows, cols, vals = [], [], []
    for f, lo, hi in zip(s.faces, s._face_low_node, s._face_high_node):
        zn, area = f[3], f[6]
        kind = s.m.zones[zn].kind
        h = (bc.h_out if kind == "outdoor" else bc.h_in)[0]
        a = h * area
        d, o = a * (0.5 * (1 - wc) + wc / 3.0), a * wc / 6.0
        rows += [lo, hi, lo, hi]; cols += [lo, hi, hi, lo]; vals += [d, d, o, o]
    C = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = (s.Kc + s.Krad + C).tocsc(); b = s.b_all[0]
    x = spl.spsolve(A, b)
    M = np.array([x[s.idx[B[key]["mj"].outputs[f"surf_{p}"]]] for p in POS])
    return M - res[key]["junction"]["Vs"][0]

if __name__ == "__main__":
    print("대류를 consistent 방식으로: 벽체별 0~150 mm 평균 편차 / RMSE(12위치)")
    for wc in (0.0, 0.5, 1.0):
        out = []
        for k in res:
            d = consistent_dev(k, wc); out.append(f"{k[:2]} {d[:9].mean():+.3f}/{np.sqrt((d**2).mean()):.3f}")
        print(f"wc {wc:.1f} |", " | ".join(out))
