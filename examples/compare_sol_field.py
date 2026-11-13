"""VOLTRA 해 파일(.sol)의 모든 격자 노드 온도와 hbcalc 온도장 비교 (CLAUDE.md 8.50). 대상: calib_steady C_E0.032_X4.3, 시각 3600 s(해 파일 ...0001), z 중간 평면(iz = 10).
.sol 형식: D, R 전체노드수 격자노드수 ..., 'N 온도 ix iy iz'(격자 노드), 'N 온도'(표면 계산 노드), T 줄(재료색별 최저, 최고), Q 줄(색별 열류)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_results", "calib_steady_sol", "C_E0.032_X4.30001.sol")
E_, X_ = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.032, 4.3)
L = open(f, "rb").read().decode("cp1252").replace("\r", "").split("\n")
R = L[1].split(); ntot, nbody = int(R[1]), int(R[2])
body = [l.split() for l in L if l.startswith("N") and len(l.split()) == 5]; surf = [float(l.split()[1]) for l in L if l.startswith("N") and len(l.split()) == 2]
V = {(int(r[2]), int(r[3]), int(r[4])): float(r[1]) for r in body}
print(f"전체 노드 {ntot}, 격자 노드 {len(body)}(R줄 {nbody}), 표면 계산 노드 {len(surf)}")
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1)
for cl in ("half", "low"):
    s = Solver(m, conv_lump=cl); s.run(bc, nsteps=2)
    keys = [k for k in V if k[2] == 10 and s.idx[k[0], k[1]] >= 0]
    d = np.array([s.T_all[1, s.idx[k[0], k[1]]] - V[k] for k in keys])
    print(f"[{cl}] z 중간 평면 노드 {len(keys)}개: 모사 - VOLTRA RMSE {np.sqrt((d ** 2).mean()):.4f}, 평균 {d.mean():+.4f}, 최대 절대 {abs(d).max():.3f}")
    # 큰 오차 위치
    o = np.argsort(-abs(d))[:6]; print("   큰 차이:", [(keys[i][0], keys[i][1], round(d[i], 3)) for i in o])

# ---- 위치별 분석 (half)
s = Solver(m, conv_lump="half"); s.run(bc, nsteps=2); cid = m.cid; nx, ny = cid.shape
keys = [k for k in V if k[2] == 10 and s.idx[k[0], k[1]] >= 0]
D = {k: s.T_all[1, s.idx[k[0], k[1]]] - V[k] for k in keys}
def kind(i, j):
    cs = [cid[a, b] for a in (i - 1, i) for b in (j - 1, j) if 0 <= a < nx and 0 <= b < ny]
    return "표면" if (any(c > 0 for c in cs) and any(c < 0 for c in cs)) else "내부"
for kd in ("표면", "내부"):
    v = np.array([D[k] for k in keys if kind(k[0], k[1]) == kd]); print(f"{kd} 노드 {len(v)}개: RMSE {np.sqrt((v ** 2).mean()):.4f}, 평균 {v.mean():+.4f}, 최대 절대 {abs(v).max():.3f}")
print("큰 차이 위치 (x mm, y mm, 차이, 주변 재료 id):")
for k in sorted(D, key=lambda k: -abs(D[k]))[:10]:
    i, j = k[0], k[1]; mats = sorted({int(cid[a, b]) for a in (i - 1, i) for b in (j - 1, j) if 0 <= a < nx and 0 <= b < ny})
    print(f"   ({m.xe[i] * 1000:.0f}, {m.ye[j] * 1000:.0f}) {D[k]:+.3f} {mats}")
print("재료:", {k: v.name for k, v in m.materials.items()}, " zones:", {k: v.kind for k, v in m.zones.items()})
