"""표면 노드의 경계 열류 비교 (CLAUDE.md 8.51). VOLTRA 온도장(.sol)에서
 (a) VOLTRA 면 플럭스(.flx, W/m2, 내부 평면 iz = 10의 두 z 슬라이스 평균) x 면 길이 / 2 를 노드별로 합한 경계 입열 in_V (W/m 깊이)
 (b) hbcalc 경계 식(대류 + 복사)을 같은 VOLTRA 온도에서 계산한 노드별 입열 in_H
 (c) 전도 순유출 (Kc T)  (정상상태에서는 (a)와 같아야 함)
세 값을 영역별로 비교해 hbcalc의 경계 식이 VOLTRA와 어디서 다른지 본다. 사용: python examples/boundary_flux_test.py 접두어 번호 EPS XPS [conv_lump]"""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
pre, k, E_, X_ = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); cl = sys.argv[5] if len(sys.argv) > 5 else "half"
D = os.path.join(ROOT, "voltra_results", "calib_steady_sol"); stem = f"{pre}{k:04d}"
rd = lambda p: open(p, "rb").read().decode("cp1252").replace("\r", "").split("\n")
V = {}
for l in rd(os.path.join(D, stem + ".sol")):
    r = l.split()
    if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
FL = [l.split() for l in rd(os.path.join(D, stem + ".flx")) if l.strip()]
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); cid = m.cid; nx, ny = cid.shape; xe, ye = m.xe, m.ye
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
s = Solver(m, conv_lump=cl)
if len(sys.argv) > 6 and sys.argv[6] == "vwf":                      # VOLTRA 형상계수(.vwf)를 그대로 사용
    from hbcalc.voltra_vwf import read_vwf
    from hbcalc.vwf_apply import exchange_areas_from_vwf
    s.set_exchange_areas(exchange_areas_from_vwf(s, read_vwf(os.path.join(D, pre + ".vwf"))))
    print("형상계수: VOLTRA .vwf 사용")
s.run(bc, nsteps=1)
T = np.full(s.N, np.nan)
for (i, j, z), t in V.items():
    if z == 10 and s.idx[i, j] >= 0: T[s.idx[i, j]] = t
EN = {3: lambda i, j: ((i, j + 1), (i + 1, j + 1), xe[i + 1] - xe[i]), 2: lambda i, j: ((i, j), (i + 1, j), xe[i + 1] - xe[i]),
      0: lambda i, j: ((i, j), (i, j + 1), ye[j + 1] - ye[j]), 1: lambda i, j: ((i + 1, j), (i + 1, j + 1), ye[j + 1] - ye[j])}
q = collections.defaultdict(dict)
for a, b, c, d, f in FL: q[(int(a), int(b), int(d))][int(c)] = float(f)
inV = np.zeros(s.N); face_q = {}
for (ix, iy, d), zz in q.items():
    if 9 in zz and 10 in zz:
        n1, n2, L = EN[d](ix, iy); qa = 0.5 * (zz[9] + zz[10]); face_q[(ix, iy, d)] = qa
        for n in (n1, n2): inV[s.idx[n[0], n[1]]] += qa * L / 2
cond = s.Kc @ T
G, b = s.G_all[0], s.b_all[0]; Krad = s.Krad
inH = b - G * T - Krad @ T                                    # hbcalc 경계 입열 (대류 + 복사), W/m 깊이
surf = np.zeros(s.N, bool); reg = {}
for i in range(nx + 1):
    for j in range(ny + 1):
        cs = [cid[a, c] for a in (i - 1, i) for c in (j - 1, j) if 0 <= a < nx and 0 <= c < ny]
        if s.idx[i, j] >= 0 and any(c > 0 for c in cs) and any(c < 0 for c in cs): surf[s.idx[i, j]] = True; reg[s.idx[i, j]] = (xe[i] * 1000, ye[j] * 1000)
sf = np.where(surf)[0]
print(f"{stem} [{cl}]  표면 노드 {len(sf)}개, 단위 W/m (깊이 1 m 기준)")
print(f"  VOLTRA 자체 일관성: 전도 순유출 - VOLTRA 면 플럭스 입열: RMS {np.sqrt(((cond - inV)[sf] ** 2).mean()):.4f}, 최대 {abs((cond - inV)[sf]).max():.4f}, 합 {(cond - inV)[sf].sum():+.4f}")
print(f"  합계: VOLTRA 입열 {inV[sf].sum():+.3f}, hbcalc 입열(같은 온도) {inH[sf].sum():+.3f}, 전도 유출 {cond[sf].sum():+.3f}")
def region(x, y):
    if y < 1000 + 1e-6 + 0.0: return "실외 벽면"
    if x <= 1000.5 and y > 1000: return "왼쪽 실내 벽(x<=1000)"
    if x >= 1199.5 and y > 1000: return "오른쪽 실내 벽(x>=1200)"
    return "접합부 띠 면"
R = collections.defaultdict(list)
for i in sf:
    x, y = reg[i]; R[region(x, y)].append(i)
for nm, idx in R.items():
    idx = np.array(idx); d = (inH - inV)[idx]
    print(f"  {nm} {len(idx)}개: 입열 VOLTRA {inV[idx].sum():+.3f} hbcalc {inH[idx].sum():+.3f}, 차이(hbcalc - VOLTRA) 합 {d.sum():+.3f}, 상대 {100 * d.sum() / max(abs(inV[idx].sum()), 1e-9):+.2f} %, RMS {np.sqrt((d ** 2).mean()):.4f}")
# 접합부 근처 노드별
o = sf[np.argsort(-abs((inH - inV)[sf]))][:10]
print("  hbcalc - VOLTRA 입열 차이가 큰 노드 (x, y mm, 차이 W/m, VOLTRA 입열, 온도):", [(round(reg[i][0]), round(reg[i][1]), round((inH - inV)[i], 3), round(inV[i], 3), round(T[i], 2)) for i in o])

# ---- 대류와 복사 분리 (VOLTRA는 면 플럭스 합계만 있으므로 대류는 hbcalc 와 같다고 두고 복사 몫을 역산)
conv = np.zeros(s.N)
for z in s.zone_ids:
    kind = s.m.zones[z].kind; h = s.bc.h_in[0] if kind == "indoor" else s.bc.h_out[0]; Ta = s.bc.T_in[0] if kind == "indoor" else s.bc.T_out[0]
    conv += h * s.conv_A[z] * (Ta - T)
radH = inH - conv; radV = inV - conv                               # hbcalc 복사 입열, VOLTRA 플럭스에서 역산한 복사 입열
print("\n대류(양쪽 같다고 가정) 대 복사 몫, 영역별 합 W/m: 대류 | 복사 hbcalc | 복사 VOLTRA 역산 | 복사 상대차")
for nm, idx in R.items():
    idx = np.array(idx)
    if nm == "실외 벽면": continue
    print(f"  {nm}: {conv[idx].sum():+.3f} | {radH[idx].sum():+.3f} | {radV[idx].sum():+.3f} | {100 * (radH[idx].sum() / radV[idx].sum() - 1):+.1f} %")
print(f"  실내 h_in(t0) = {s.bc.h_in[0]:.3f}, 공기 {s.bc.T_in[0]:.2f} ℃, hrb {m.hrb}")

# ---- VOLTRA 자체 일관성이 깨지는 노드 (전도 순유출 - VOLTRA 면 플럭스 입열)
res = (cond - inV)
o = sf[np.argsort(-abs(res[sf]))][:12]
print("\n표면 노드 중 (전도 순유출 - VOLTRA 면 입열) 가 큰 노드 (x, y mm, 값 W/m, VOLTRA 입열, 온도, 주변 칸 재료 id):")
for i in o:
    ii = [(a, c) for a in range(nx + 1) for c in range(ny + 1) if s.idx[a, c] == i][0]
    mats = sorted({int(cid[a, c]) for a in (ii[0] - 1, ii[0]) for c in (ii[1] - 1, ii[1]) if 0 <= a < nx and 0 <= c < ny})
    print(f"   ({xe[ii[0]] * 1000:.0f}, {ye[ii[1]] * 1000:.0f}) {res[i]:+.3f}  입열 {inV[i]:+.3f}  T {T[i]:.2f}  {mats}")
