"""VOLTRA 온도장(.sol)을 hbcalc 전도 연산자에 그대로 넣어 내부 노드 열 평형 잔차를 본다 (CLAUDE.md 8.51).
VOLTRA 의 온도가 hbcalc 와 같은 전도 이산화를 만족하면 정상상태 내부 노드(표면이 아닌 노드)의 순 전도 열류 (Kc T)_i 가 0 에 가까워야 한다(z 방향 전도 제외).
사용: python examples/conduction_residual_test.py 접두어 번호 열전도율_EPS 열전도율_XPS"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
pre, k, E_, X_ = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
V = {}
for l in open(os.path.join(D, f"{pre}{k:04d}.sol"), "rb").read().decode("cp1252").replace("\r", "").split("\n"):
    r = l.split()
    if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); cid = m.cid; nx, ny = cid.shape
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
s = Solver(m, conv_lump="half"); s.run(bc, nsteps=1)
T = np.full(s.N, np.nan)
for (i, j, z), t in V.items():
    if z == 10 and s.idx[i, j] >= 0: T[s.idx[i, j]] = t
assert not np.isnan(T).any()
r = s.Kc @ T                                              # 각 노드의 순 전도 유출 (W, 깊이 z_mm 전체 기준)
surf = np.zeros(s.N, bool)
for i in range(nx + 1):
    for j in range(ny + 1):
        cs = [cid[a, b] for a in (i - 1, i) for b in (j - 1, j) if 0 <= a < nx and 0 <= b < ny]
        if s.idx[i, j] >= 0 and any(c > 0 for c in cs) and any(c < 0 for c in cs): surf[s.idx[i, j]] = True
inn = ~surf; xy = {s.idx[i, j]: (m.xe[i] * 1000, m.ye[j] * 1000) for i in range(nx + 1) for j in range(ny + 1) if s.idx[i, j] >= 0}
print(f"{pre}{k:04d}: 내부 노드 {inn.sum()}개 전도 잔차 (Kc T)_i [W, 전체 깊이 기준]: RMS {np.sqrt((r[inn] ** 2).mean()):.4f}, 최대 절대 {abs(r[inn]).max():.4f}, 합 {r[inn].sum():+.4f}")
o = np.argsort(-abs(r)); print("  전체(표면 포함) 큰 잔차 위치 x, y mm, 값, 표면여부:", [(round(xy[i][0]), round(xy[i][1]), round(r[i], 3), bool(surf[i])) for i in o[:8]])
oi = [i for i in o if inn[i]][:8]; print("  내부 노드 큰 잔차:", [(round(xy[i][0]), round(xy[i][1]), round(r[i], 4)) for i in oi])
# 전체 열류 대비 크기: 접합부 총 열류 W
print("  참고 해당 모델 한 방 열류 약 %.1f W" % (np.nan if False else 0.5 * np.abs(r[surf]).sum()))
