"""임의의 2D VOLTRA .vtr (T자 접합부형)를 hbcalc 로 예측: 실내 표면온도(야간 평균), ΔTs, 열류, TRISCO 조건 정상상태 Ψ.
사용: python examples/predict_vtr.py 파일.vtr [실내온도파일(기본 Wall_TI.FTE)]
가정: 경계조건 파일은 이 저장소의 것(wall_TO.FTE 등)과 같다고 본다 (파일이 다르면 결과도 달라진다)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.planar import planar_model
from hbcalc.validate import night_mask

path = sys.argv[1]; TI = sys.argv[2] if len(sys.argv) > 2 else "Wall_TI.FTE"
m = model_from_vtr(path)
print("== 읽은 모델:", path, "| 격자", m.shape, f"| 폭 {m.xe[-1]*1000:.0f} mm 높이 {m.ye[-1]*1000:.0f} mm 깊이 {m.zt*1000:.0f} mm | hrb {m.hrb}")
print("재료:", {k: (v.name, v.lam, v.rho, v.cp, v.eps) for k, v in m.materials.items()})
print("경계 영역:", {k: v.kind for k, v in m.zones.items()})
cid = m.cid; nx, ny = cid.shape
outd = [z for z, v in m.zones.items() if v.kind == "outdoor"][0]; ind = sorted(z for z, v in m.zones.items() if v.kind == "indoor")
# 오른쪽 벽: 실내 경계 ind[-1] 아래의 재료 셀 열에서 표면 노드 찾기
xc = int(np.argmax([(cid[i, :] == -ind[-1]).any() for i in range(nx - 1, -1, -1)]))  # 사용 안 함
col = nx - 5                                                                         # 오른쪽 끝 근처 열
js = max(j for j in range(ny) if cid[col, j] > 0) + 1                                 # 표면 격자선 번호
jb_mat = [cid[col, j] for j in range(ny) if cid[col, j] > 0]
layers = []; j0 = 0
for j in range(ny):
    c = cid[col, j]
    if c > 0:
        if layers and layers[-1][0] == c: layers[-1][1] += m.ye[j+1] - m.ye[j]; layers[-1][2] += 1
        else: layers.append([c, m.ye[j+1] - m.ye[j], 1])
layers = [(m.materials[c].name, round(t * 1000, 3), n) for c, t, n in layers]
print("벽 층 (실외 -> 실내):", layers, "| 표면 y =", round(m.ye[js] * 1000, 1), "mm")
# 오른쪽 접합부 면: 실내 영역 경계가 시작하는 가장 왼쪽 x
ix_face = min(i for i in range(nx) if cid[i, js] == -ind[-1])
x_face = m.xe[ix_face]
offs = [(i, (m.xe[i] - x_face) * 1000) for i in range(ix_face, nx + 1)]
want = (0, 50, 100, 150, 200, 300, 500, 1000)
sel = [(i, o) for i, o in offs if any(abs(o - w) < 1e-3 for w in want)]
print("접합부 면 x =", round(x_face * 1000, 1), "mm; 읽을 수 있는 노드 위치(mm):", [round(o) for _, o in sel])
m = replace(m, outputs={**{f"s{round(o)}": (i, js) for i, o in sel}, **{f"b{round(o)}": (i, js - layers[-1][2]) for i, o in sel}})
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", TI, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
N = 720
s = Solver(m); s.run(bc, nsteps=N)
mk = night_mask(N)
mk[714:] = False                                  # 마지막 밤은 720시간에서 잘리므로 뺀다 (29번의 완전한 밤)
pm = planar_model([(n, t) for n, t, _ in layers], {m.materials[c].name: m.materials[c] for c in m.materials if m.materials[c].name in [l[0] for l in layers]},
                  hrb=m.hrb, cells=[n for _, _, n in layers], width=2.2, nx=44)
sp = Solver(pm); sp.run(bc, nsteps=N)
P = sp.series("surf")[:N][mk].mean()
print(f"\n[야간 평균 예측, {mk.sum()}시간]  비열교부(평면 모델) 표면 {P:.3f} ℃")
print("  위치(mm)   표면(℃)   벽체쪽(℃)   표면 - 평면(℃)")
for i, o in sel:
    a = s.series(f"s{round(o)}")[:N][mk].mean(); b = s.series(f"b{round(o)}")[:N][mk].mean()
    print(f"  {o:7.0f}   {a:7.3f}   {b:8.3f}   {a - P:+8.3f}")
s0 = s.series("s0")[:N][mk]; print(f"\n  ΔTs(평면 - 0 mm, 야간 평균) = {P - s0.mean():.3f} ℃ | 0 mm 야간 최저 {s0.min():.2f} 최고 {s0.max():.2f} ℃ | 전체 기간 최저 {s.series('s0')[:N].min():.2f} ℃")
F = s.zone_flows_into_out(); print(f"  야간 평균 열류: 실외 유출 {np.mean([F[outd][1][k]-F[outd][0][k] for k in range(N)]):.2f} W, 실내 유입 합 {np.mean([sum(F[z][0][k]-F[z][1][k] for z in ind) for k in range(N)]):.2f} W")
# TRISCO 조건 정상상태 Ψ
Ti, Te, hi_r, hi_l, ho, hrb = 20.0, -3.7, 1.415, 1.42, 16.03, m.hrb
bz = Boundary(np.array([Ti]), np.array([Ti]), np.array([1.0]), np.array([1.0]),
              zone_values={outd: (np.array([Te]), np.array([ho])), ind[0]: (np.array([Ti]), np.array([hi_l])), ind[-1]: (np.array([Ti]), np.array([hi_r]))})
ss = Solver(m, conv_lump="half"); ss.run(bz, nsteps=1); F = ss.zone_flows_into_out()
phis = [F[z][0][0] - F[z][1][0] for z in ind]; phi = 0.5 * sum(phis)
R = 1 / (1.415 + 0.9 * m.hrb) + 1 / (ho + 0.9 * m.hrb) + sum(t / 1000 / m.materials[c].lam for c, t, _ in [(next(k for k, v in m.materials.items() if v.name == n), t, 0) for n, t, _ in layers])
W = m.xe[-1]; phi2d = (Ti - Te) * W * m.zt / R
psi = (2 * phi - phi2d) / (Ti - Te)
print(f"\n[TRISCO 조건 정상상태, 실내 20 실외 -3.7 ℃, hrb {m.hrb}] 방별 열류 {phis[0]:.3f}, {phis[1]:.3f} W, 평면 Φ2D {phi2d:.3f} W (폭 {W:.3f} m)")
print(f"  선형열관류율 Ψ ≈ {psi:.4f} W/(m K)  (hbcalc half 기준. 격자 의존성과 TRISCO 대비 편차는 약 +-0.003 ~ 0.006 W/(m K) 정도로 추정)")
