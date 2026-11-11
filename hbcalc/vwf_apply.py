"""VOLTRA .vwf 의 형상계수를 hbcalc 요소(노드 둘레 셀)의 교환면적으로 바꾼다 (2D, 중간 높이 z=zmid).
면(칸 x z 조각) -> 칸 단위 띠: 원천은 z 중간 조각(들), 목적지는 z 전체 조각 합. 상호성으로 대칭화. 칸 -> 양 끝 노드에 절반씩."""
import numpy as np
import scipy.sparse as sp


def exchange_areas_from_vwf(solver, vwf, zmid=500.0, zone_map=None):
    m = solver.m
    f, Fu = vwf["faces"], vwf["F"].tocoo()
    n = len(f)
    box = np.array([x[4] for x in f]); area = np.array([x[2] for x in f])
    zone = np.zeros(n, int); k = 0
    for z in sorted(vwf["zones"]):
        zone[k:k + vwf["zones"][z]] = z; k += vwf["zones"][z]
    # 하삼각 정보를 포함한 대칭 교환면적 (A_i F_ij 는 상호성으로 대칭): 상삼각만 저장돼 있으므로 전치를 더한다
    Wu = sp.csr_matrix((Fu.data * area[Fu.row], (Fu.row, Fu.col)), shape=(n, n))
    Wf = Wu + Wu.T                                    # 조각 단위 A_i F_ij
    xe, ye = m.xe * 1000.0, m.ye * 1000.0
    ix = lambda v: int(np.argmin(np.abs(xe - v)))
    iy = lambda v: int(np.argmin(np.abs(ye - v)))
    # 칸(띠) 정의: (zone, 축, 평면좌표, 시작, 끝)
    segs, seg_of = {}, np.zeros(n, int)
    for i, b in enumerate(box):
        if b[2] == b[3]: key = (zone[i], "x", b[2], b[0], b[1])       # 가로 면: y 평면, x 구간
        elif b[0] == b[1]: key = (zone[i], "y", b[0], b[2], b[3])     # 세로 면: x 평면, y 구간
        else: raise ValueError("면 방향을 알 수 없음")
        seg_of[i] = segs.setdefault(key, len(segs))
    S = len(segs)
    zlo = np.array([b[4] for b in box])
    # 원천 조각: z 중간(zmid 를 경계로 하는 두 조각)
    mid = (np.abs(box[:, 5] - zmid) < 1e-9) | (np.abs(box[:, 4] - zmid) < 1e-9)
    P = sp.csr_matrix((np.ones(n), (seg_of, np.arange(n))), shape=(S, n))
    # 칸 s 의 중간 조각 F(원천 조각 -> 목적 칸 t) = Σ_m F_{i->(t,m)} ; W_st = 칸면적(깊이 zt) x 평균 F
    Ff = sp.diags(1 / area) @ Wf                       # F_ij
    Fseg = (Ff[np.where(mid)[0]] @ P.T).toarray()      # (중간 조각, 목적 칸)
    src_seg = seg_of[np.where(mid)[0]]
    Fst = np.zeros((S, S)); cnt = np.zeros(S)
    for r, s_ in enumerate(src_seg): Fst[s_] += Fseg[r]; cnt[s_] += 1
    Fst = Fst / cnt[:, None]
    keys = list(segs)
    length = np.array([abs(k[4] - k[3]) / 1000.0 for k in keys])
    Wst = (length * m.zt)[:, None] * Fst
    Wst = 0.5 * (Wst + Wst.T)
    # 요소(노드, zone, 축) 인덱스
    eidx = {}
    for j, e in enumerate(solver.rad_elements): eidx[(e["node"], e["zone"], e["ax"])] = j
    zmap = zone_map or {z: z for z in vwf["zones"]}
    Wel = np.zeros((len(solver.rad_elements),) * 2)
    ends = []
    for key in keys:
        z, ax, plane, a, b = key
        if ax == "x": nodes = [solver.idx[ix(a), iy(plane)], solver.idx[ix(b), iy(plane)]]
        else: nodes = [solver.idx[ix(plane), iy(a)], solver.idx[ix(plane), iy(b)]]
        ends.append([eidx.get((nd, zmap[z], ax)) for nd in nodes])
    for s_ in range(S):
        for t in range(S):
            w = Wst[s_, t]
            if w == 0 or keys[s_][0] != keys[t][0]: continue
            for a in ends[s_]:
                for b in ends[t]:
                    if a is not None and b is not None: Wel[a, b] += w / 4.0
    return Wel
