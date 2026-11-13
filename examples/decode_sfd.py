"""VOLTRA .sfd 파일 해독 확인 (CLAUDE.md 8.48). 사용: python examples/decode_sfd.py [sfd파일] (기본: voltra_results/sfd/C_E0.032_X4.3_0000.sfd, C_E0.032_X4.3 모델)
형식: 첫 줄 'D 항목수', 이후 'N 노드번호 a b 값'. 노드번호 = (2D 노드 순번) x 21 + z 평면(0 ~ 20), 2D 노드 순번은 재료 셀에 닿는 격자 노드를 x 바깥, y 안쪽 순서로 센 것.
각 항목 = 표면 노드가 인접한 표면 칸 하나(선분 길이 L x z 슬라이스 dz)의 면적 L x dz / 4 (= (L/2) x (dz/2)): 칸 면적을 네 모서리 노드에 같은 몫.
a = 표면 방향(0: BC가 -x쪽, 1: +x쪽, 2: -y쪽, 3: +y쪽), b = 표면 재료 색(VOLTRA 색 3 콘크리트, 6 석고, 8 XPS)."""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc.presets import try3_model
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_results", "sfd", "C_E0.032_X4.3_0000.sfd")
ent = [(int(a[1]), int(a[2]), int(a[3]), float(a[4])) for a in (l.split() for l in open(f) if l.startswith("N"))]
m = try3_model({"EPS": 0.032, "XPS": 4.3}, snap_mm=1); cid = m.cid; nx, ny = cid.shape
exist = np.zeros((nx + 1, ny + 1), bool)
for i in range(nx):
    for j in range(ny):
        if cid[i, j] > 0: exist[i:i + 2, j:j + 2] = True
order = [(i, j) for i in range(nx + 1) for j in range(ny + 1) if exist[i, j]]
xe, ye = m.xe * 1000, m.ye * 1000; MAT = {1: 3, 4: 6, 5: 8}
def edges(i, j):
    out = []
    for a in (i - 1, i):
        for b in (j - 1, j):
            if not (0 <= a < nx and 0 <= b < ny) or cid[a, b] <= 0: continue
            for da, db, ln, nodes in ((a + 1, b, ye[b + 1] - ye[b], {(a + 1, b), (a + 1, b + 1)}), (a - 1, b, ye[b + 1] - ye[b], {(a, b), (a, b + 1)}),
                                      (a, b + 1, xe[a + 1] - xe[a], {(a, b + 1), (a + 1, b + 1)}), (a, b - 1, xe[a + 1] - xe[a], {(a, b), (a + 1, b)})):
                if 0 <= da < nx and 0 <= db < ny and cid[da, db] < 0 and (i, j) in nodes: out.append((ln, MAT[int(cid[a, b])]))
    return out
nodes2d = {n // 21 for n, *_ in ent}; rows = collections.defaultdict(list)
for n, a, b, v in ent:
    if n % 21 == 5: rows[n // 21].append((round(v * 1e6, 1), b))              # z 내부 평면 하나
ok = sum(sorted((round(ln * 50 / 4, 1), mat) for ln, mat in edges(*order[k]) for _ in (0, 1)) == sorted(l) for k, l in rows.items())
print(f"항목 {len(ent)}, 표면 2D 노드 {len(nodes2d)}, 모델 표면 노드 면적(L/2 x dz/2, 두 z 슬라이스)과 재료가 모두 일치한 노드 {ok} / {len(rows)}")
