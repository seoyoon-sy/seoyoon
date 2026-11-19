"""N5_noblock 접합부(난방) VOLTRA 해 파일(.sol, 24스텝마다 저장)과 hbcalc 온도장 비교 (CLAUDE.md 8.64).
파일 번호 n = 24 n 시간째, z 중간 평면(iz = 10)의 격자 노드 1765개. 사용: python examples/compare_sol_transient.py
확산일사 장애물 차단(diffuse_obstruction) 끔/켬을 같이 비교하고, 재료별 평균 차이와 평면부 단면을 낸다."""
import os, sys, glob, re, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np
from hbcalc import Boundary, Solver
import test_walls_def as T
D = os.path.join(ROOT, "voltra_results", "transient_sol")

def load(f):
    V = {}
    for l in open(f, "rb").read().decode("cp1252").replace("\r", "").split("\n"):
        p = l.split()
        if p and p[0] == "N" and len(p) == 5: V[(int(p[2]), int(p[3]), int(p[4]))] = float(p[1])
    return V

files = {24 * int(re.search(r"save24(\d{4})\.sol", f).group(1)): f for f in glob.glob(os.path.join(D, "*.sol"))}
files.pop(744, None); HS = sorted(files); Vs = {h: load(files[h]) for h in HS}
bc0 = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
m = T.build("N5_noblock")["mj"]; cid = m.cid; nx, ny = cid.shape
def region(i, j): return tuple(sorted({int(cid[a, b]) for a in (i - 1, i) for b in (j - 1, j) if 0 <= a < nx and 0 <= b < ny}))
print("시각(h):", HS, "\n각 칸: RMSE / 부호 있는 평균 (모사 - VOLTRA, z 중간 평면 노드 1765개), 마지막은 전체 최대 절대오차")
for name, do in (("확산 차단 끔(현재 기본)", False), ("확산 차단 켬", True)):
    bc = copy.deepcopy(bc0); bc.diffuse_obstruction = do
    s = Solver(m); s.run(bc, nsteps=744)
    keys = [k for k in Vs[0] if k[2] == 10 and s.idx[k[0], k[1]] >= 0]
    cells, mx = [], 0
    for h in HS:
        d = np.array([s.T_all[h, s.idx[k[0], k[1]]] - Vs[h][k] for k in keys]); cells.append(f"{np.sqrt((d ** 2).mean()):.3f}/{d.mean():+.3f}"); mx = max(mx, abs(d).max())
    print(f"{name:22s}", " ".join(cells), f"| 최대 {mx:.3f}")
    if not do:
        print("  재료별 평균 차이(h=720): ", end="")
        d = {k: s.T_all[720, s.idx[k[0], k[1]]] - Vs[720][k] for k in keys}
        for r in ((1,), (2,), (3,), (-3, 3)):
            v = [x for k, x in d.items() if region(k[0], k[1]) == r]; print(f"{r} n={len(v)} {np.mean(v):+.3f}", end="  ")
        print()
