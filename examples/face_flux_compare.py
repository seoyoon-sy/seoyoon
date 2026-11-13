"""면 단위 플럭스 비교 (CLAUDE.md 8.51): VOLTRA 온도장에서 면마다 q_VOLTRA(.flx, 내부 평면 두 슬라이스 평균)와
q_hbcalc = h(Ta - T평균) + 복사(면 평균 온도, 해석 형상계수 grey_network)를 비교한다. 사용: python examples/face_flux_compare.py 접두어 EPS XPS"""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, scipy.sparse as sp
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model
from hbcalc.radiation import exchange_area, grey_network
pre, E_, X_ = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); D = os.path.join(ROOT, "voltra_results", "calib_steady_sol")
rd = lambda p: open(p, "rb").read().decode("cp1252").replace("\r", "").split("\n")
V = {}
for l in rd(os.path.join(D, pre + "0000.sol")):
    r = l.split()
    if r and r[0] == "N" and len(r) == 5: V[(int(r[2]), int(r[3]), int(r[4]))] = float(r[1])
Q = collections.defaultdict(dict)
for l in rd(os.path.join(D, pre + "0000.flx")):
    r = l.split()
    if r: Q[(int(r[0]), int(r[1]), int(r[3]))][int(r[2])] = float(r[4])
m = try3_model({"EPS": E_, "XPS": X_}, snap_mm=1); bc = Boundary.from_voltra_files(*(os.path.join(ROOT, g) for g in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
s = Solver(m, conv_lump="half"); s.run(bc, nsteps=1); F = s.faces; nf = len(F); xe, ye, zt = m.xe, m.ye, m.zt
ext = [((xe[f[0]], xe[f[0] + 1], ye[f[1]]) if f[2] == "x" else (ye[f[1]], ye[f[1] + 1], xe[f[0]])) for f in F]
A = np.array([f[6] for f in F]); zone = np.array([f[3] for f in F]); W = np.zeros((nf, nf))
for ia in range(nf):
    if F[ia][2] != "x": continue
    la, ha, pa = ext[ia]
    for ib in range(nf):
        if F[ib][2] != "y" or zone[ib] != zone[ia]: continue
        lb, hb, pb = ext[ib]; nx2 = 1 if F[ib][7] == "right" else -1; ny1 = 1 if F[ia][7] == "up" else -1
        if ((la + ha) / 2 - pb) * nx2 <= 0 or ((lb + hb) / 2 - pa) * ny1 <= 0: continue
        W[ia, ib] = W[ib, ia] = exchange_area(la, ha, pa, lb, hb, pb, zt, s.zmode)
Gt, Gs, Fsum = grey_network([dict(eps=f[4]) for f in F], W, A, m.hrb)
def vkey(f):                                   # hbcalc 면 -> .flx 의 (재료 칸 ix, iy, 방향): 위 면은 아래 칸, 오른쪽 면은 왼쪽 칸
    i, j, ax, side = f[0], f[1], f[2], f[7]
    return {"up": (i, j - 1, 3), "down": (i, j, 2), "right": (i - 1, j, 1), "left": (i, j, 0)}[side]
Tavg = np.zeros(nf); qV = np.zeros(nf); ok = np.zeros(nf, bool)
for k, f in enumerate(F):
    i, j, ax = f[0], f[1], f[2]; n1, n2 = ((i, j), (i + 1, j)) if ax == "x" else ((i, j), (i, j + 1))
    if (n1[0], n1[1], 10) not in V: continue
    Tavg[k] = (V[(n1[0], n1[1], 10)] + V[(n2[0], n2[1], 10)]) / 2
    z = Q.get(vkey(f), {})
    if 9 in z and 10 in z: qV[k] = 0.5 * (z[9] + z[10]); ok[k] = True
Ta = np.array([float(bc.T_out[0]) if m.zones[z].kind == "outdoor" else float(bc.T_in[0]) for z in zone]); h = np.array([float(bc.h_out[0]) if m.zones[z].kind == "outdoor" else float(bc.h_in[0]) for z in zone])
Qrad = (Gt @ Tavg) - Gs * Ta                                  # 요소 복사 순유출 W (기준온도 = 영역 공기온도)
qH = h * (Ta - Tavg) - Qrad / A                               # 입열 W/m2
d = (qH - qV)[ok]
print(f"{pre}0000 면 {ok.sum()}개: hbcalc(면 평균 방식, 해석 형상계수) - VOLTRA 면 플럭스: 평균 {d.mean():+.4f}, RMS {np.sqrt((d ** 2).mean()):.4f} W/m2 (VOLTRA 절대 평균 {abs(qV[ok]).mean():.3f})")
for nm, sel in (("실외 면", [m.zones[z].kind == "outdoor" for z in zone]), ("실내 면", [m.zones[z].kind == "indoor" for z in zone])):
    sel = np.array(sel) & ok
    if sel.any(): dd = (qH - qV)[sel]; print(f"  {nm} {sel.sum()}개: 평균 {dd.mean():+.4f}, RMS {np.sqrt((dd ** 2).mean()):.4f}, 상대 평균 {100 * dd.mean() / abs(qV[sel]).mean():+.2f} %")
# 실내 면 중 큰 차이
idx = np.where(ok & (np.array([m.zones[z].kind for z in zone]) == "indoor"))[0]; o = idx[np.argsort(-abs((qH - qV)[idx]))][:10]
print("  실내 면 큰 차이 (x구간 mm, 평면 mm, 방향, qH, qV):", [((round(ext[k][0] * 1000), round(ext[k][1] * 1000)), round(ext[k][2] * 1000), F[k][7], round(qH[k], 2), round(qV[k], 2)) for k in o[:8]])
# 복사 몫 비교: VOLTRA 면 플럭스에서 대류 몫(h(Ta - T평균))을 빼 복사 역산, hbcalc 복사와 비교
radV = qV - h * (Ta - Tavg); radH = -Qrad / A
for nm, kind in (("실내", "indoor"),):
    sel = ok & (np.array([m.zones[z].kind for z in zone]) == kind); print(f"  {nm} 복사 몫: 면적 가중 합 VOLTRA 역산 {np.sum(radV[sel] * A[sel]):+.3f} W, hbcalc {np.sum(radH[sel] * A[sel]):+.3f} W, 상대 {100 * (np.sum(radH[sel] * A[sel]) / np.sum(radV[sel] * A[sel]) - 1):+.2f} %")

# ---- 면 대응 정정 후 전체 비교와 큰 차이 면
d2 = (qH - qV); idx = np.where(ok)[0]; o = idx[np.argsort(-abs(d2[idx]))][:10]
print(f"면 대응 정정 후: 면 {ok.sum()}개 평균 {d2[ok].mean():+.4f}, RMS {np.sqrt((d2[ok] ** 2).mean()):.4f}")
for k in o: print("   면", F[k][2], F[k][7], "구간 mm", (round(ext[k][0] * 1000), round(ext[k][1] * 1000)), "평면 mm", round(ext[k][2] * 1000), "qH", round(qH[k], 3), "qV", round(qV[k], 3), "차이", round(d2[k], 3))
