"""띠 끝, 뿌리 노드 진단 (CLAUDE.md 8.52): VOLTRA 온도(.sol 0000, z 중간)에서 노드 하나의 전도 연결(이웃, 전도도, 온도차, 유량)과 VOLTRA 면 플럭스 입열을 나열한다.
사용: python examples/corner_node_diag.py 접두어 EPS XPS x_mm y_mm [x2 y2 ...]"""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
pre, E_, X_ = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); pts = [(float(sys.argv[i]), float(sys.argv[i + 1])) for i in range(4, len(sys.argv) - 1, 2)]
D = os.path.join(ROOT, "voltra_results", "calib_steady_sol"); rd = lambda p: open(p, "rb").read().decode("cp1252").replace("\r", "").split("\n")
V = {}; Rl = []
for l in rd(os.path.join(D, pre + "0000.sol")):
    r = l.split()
    if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
    if r and r[0] == "Q": Rl.append(l)
print("VOLTRA 색별 열류 Q:", Rl)
Q = collections.defaultdict(dict)
for l in rd(os.path.join(D, pre + "0000.flx")):
    r = l.split()
    if r: Q[(int(r[0]), int(r[1]), int(r[3]))][int(r[2])] = float(r[4])
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); cid = m.cid; nx, ny = cid.shape; xe, ye = m.xe * 1000, m.ye * 1000
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
s = Solver(m, conv_lump="half"); s.run(bc, nsteps=1)
T = np.zeros(s.N)
for (i, j, z), t in V.items():
    if z == 10 and s.idx[i, j] >= 0: T[s.idx[i, j]] = t
inv = {s.idx[i, j]: (i, j) for i in range(nx + 1) for j in range(ny + 1) if s.idx[i, j] >= 0}
Kc = s.Kc.tocsr()
# 면 입열 합계 점검: 모든 면의 q x 길이 (z 20슬라이스 평균) = 색별 W/m 와 비교
tot = collections.Counter()
for (ix, iy, d), zz in Q.items():
    if len(zz) < 20: continue
    L = (xe[ix + 1] - xe[ix]) / 1000 if d in (2, 3) else (ye[iy + 1] - ye[iy]) / 1000
    c = cid[ix, iy]; nb = {0: (ix - 1, iy), 1: (ix + 1, iy), 2: (ix, iy - 1), 3: (ix, iy + 1)}[d]; bcid = cid[nb[0], nb[1]] if 0 <= nb[0] < nx and 0 <= nb[1] < ny else 0
    tot[int(bcid)] += np.mean(list(zz.values())) * L
print("VOLTRA 면 플럭스를 BC 영역별로 합한 값(W/m, z 평균 x 길이): ", {k: round(v, 3) for k, v in tot.items()}, "  (음수 = 영역으로 나감)")
for (x, y) in pts:
    i = int(np.argmin(abs(xe - x))); j = int(np.argmin(abs(ye - y))); n = s.idx[i, j]
    print(f"\n노드 ({xe[i]:.0f}, {ye[j]:.0f}) T = {T[n]:.3f}, 주변 칸 재료 id (왼아래, 오른아래, 왼위, 오른위): {[int(cid[a, b]) if 0 <= a < nx and 0 <= b < ny else None for (a, b) in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j))]}")
    row = Kc.getrow(n); tot_f = 0
    for c, g in zip(row.indices, row.data):
        if c == n: continue
        (ia, ja) = inv[c]; f = -g * (T[c] - T[n]); tot_f += f
        print(f"   이웃 ({xe[ia]:.0f}, {ye[ja]:.0f}) 전도도 {-g:.4f} W/K, 온도차 {T[n] - T[c]:+.3f}, 노드에서 나가는 열 {f:+.4f} W/m")
    print(f"   전도 순유출 합 {tot_f:+.4f} W/m")
