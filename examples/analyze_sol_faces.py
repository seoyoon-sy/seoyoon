"""VOLTRA 표면 계산 노드(.sol 끝의 'N 온도' 줄, .flx 줄과 같은 순서)와 인접 격자 노드 온도, 플럭스의 관계 분석 (CLAUDE.md 8.51).
각 면(ix, iy, iz, 방향 d)의 양 끝 격자 노드: d=3(위) (ix,iy+1),(ix+1,iy+1) / d=2(아래) (ix,iy),(ix+1,iy) / d=0(왼쪽 변) (ix,iy),(ix,iy+1) / d=1(오른쪽 변) (ix+1,iy),(ix+1,iy+1).
사용: python examples/analyze_sol_faces.py 접두어(예: C_E0.032_X0.04) 번호(예: 0)"""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
pre, k = sys.argv[1], int(sys.argv[2]); stem = f"{pre}{k:04d}"
def read(path): return open(path, "rb").read().decode("cp1252").replace("\r", "").split("\n")
V, calc = {}, []
for l in read(os.path.join(D, stem + ".sol")):
    r = l.split()
    if r and r[0] == "N":
        if len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
        elif len(r) == 2: calc.append(float(r[1]))
F = [l.split() for l in read(os.path.join(D, stem + ".flx")) if l.strip()]
rows = [(int(a), int(b), int(c), int(d), float(q), calc[i]) for i, (a, b, c, d, q) in enumerate(F)]
EN = {3: lambda i, j: ((i, j + 1), (i + 1, j + 1)), 2: lambda i, j: ((i, j), (i + 1, j)), 0: lambda i, j: ((i, j), (i, j + 1)), 1: lambda i, j: ((i + 1, j), (i + 1, j + 1))}
out = []
for ix, iy, iz, d, q, tc in rows:
    if iz in (0, 20): continue                     # 끝 평면 제외(z 반 슬라이스)
    n1, n2 = EN[d](ix, iy)
    if (n1[0], n1[1], iz) in V and (n2[0], n2[1], iz) in V:
        out.append((ix, iy, iz, d, q, tc, V[(n1[0], n1[1], iz)], V[(n2[0], n2[1], iz)]))
a = np.array(out); print(f"{stem}: 면 {len(rows)}개 중 내부 평면 면 {len(a)}개")
q, tc, t1, t2 = a[:, 4], a[:, 5], a[:, 6], a[:, 7]; tm = (t1 + t2) / 2
print("계산 노드 - 양 끝 격자 노드 평균: 평균 %.4f, 절대 평균 %.4f, 최대 %.3f" % ((tc - tm).mean(), abs(tc - tm).mean(), abs(tc - tm).max()))
print("계산 노드가 양 끝 노드 사이에 있는 비율 %.3f" % np.mean((tc >= np.minimum(t1, t2) - 1e-9) & (tc <= np.maximum(t1, t2) + 1e-9)))
for dname, dd in (("실외 면(2)", 2), ("실내 위쪽 면(3)", 3), ("왼쪽 변(0)", 0), ("오른쪽 변(1)", 1)):
    s = a[:, 3] == dd
    if s.sum() < 5: continue
    x = q[s]; y = (tc - tm)[s]; A = np.vstack([x, np.ones_like(x)]).T; co = np.linalg.lstsq(A, y, rcond=None)[0]; res = y - A @ co
    print(f"  {dname} {s.sum()}개: (계산노드 - 끝노드평균) = {co[0]:.5f} x q + {co[1]:+.4f}, 잔차 RMSE {np.sqrt((res ** 2).mean()):.4f} (y 표준편차 {y.std():.4f})")
