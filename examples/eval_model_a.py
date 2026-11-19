"""Model A 평가: 사용자가 보낸 정의 파일(voltra_inputs/TRY3_INS_160_ver2.vtr)을 그대로 읽어 모사하고
(1) VOLTRA 엑셀 시트 'Model A'(난방 20 도, 야간 18 ~ 07시, 12위치 표면온도)와 비교,
(2) TRISCO 정상상태 표 행 'Model A(0.024)'(열류, Ψ)와 비교한다.
Model A 구성(파일에서 확인): 콘크리트 200, EPS 160 mm 열전도율 0.024, 시멘트 블록 90, 석고 10 mm, 내벽 콘크리트 200 mm(XPS 띠 없음),
격자 55 x 60 손격자, 출력 노드 12위치(0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500 mm) x (표면, 석고 뒤쪽).
사용: python examples/eval_model_a.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.validate import night_mask
R = lambda *p: os.path.join(ROOT, *p)
VTR = R("voltra_inputs", "TRY3_INS_160_ver2.vtr"); N = 744; mk = night_mask(N)
POS = (0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500)

m = model_from_vtr(VTR)
xe = np.array(m.xe) * 1000
onods = [m.outputs[f"onod{k}"] for k in range(24)]
offs = [round(xe[i] - 1200) for i, j in onods[0::2]]
assert tuple(offs) == POS, offs                                  # 출력 노드 위치가 엑셀 열 이름과 같은지 확인

# ---- (1) 야간 시간별 표면온도
bc = Boundary.from_voltra_files(*(R(f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
xl = pd.ExcelFile(R("voltra_30cases_night420_통합_v2.xlsx")).parse("Model A")
V = np.array([xl[f"{p}mm 표면온도(℃)"].values[:N] for p in POS]); Vw = np.array([xl[f"{p}mm 석고보드 벽체 쪽 표면온도(℃)"].values[:N] for p in POS])
res = {}
for scheme in ("half", "voltra"):
    s = Solver(m, conv_lump=scheme); s.run(bc, nsteps=N)
    M = np.array([s.series(f"onod{2 * k}")[:N] for k in range(12)]); Mw = np.array([s.series(f"onod{2 * k + 1}")[:N] for k in range(12)])
    res[scheme] = (M, Mw)
print("=== (1) VOLTRA 엑셀 'Model A' (난방, 야간 420시간, 12위치)")
for scheme, (M, Mw) in res.items():
    e = (M - V)[:, mk]; ew = (Mw - Vw)[:, mk]
    print(f"[{scheme:6s}] 표면 평균절대오차 {abs(e).mean():.4f}, RMSE {np.sqrt((e**2).mean()):.4f}, 최대 {abs(e).max():.3f}, 0.05 초과 {100*(abs(e)>0.05).mean():.2f} %,"
          f" 0 mm 편향 {e[0].mean():+.4f}, 위치별 편향 " + " ".join(f"{x:+.3f}" for x in e.mean(1)) + f" | 석고 뒤쪽 평균절대 {abs(ew).mean():.4f}")
    dts_v = V[-1, mk].mean() - V[0, mk].mean(); dts_m = M[-1, mk].mean() - M[0, mk].mean()
    print(f"         ΔTs(500 mm - 0 mm, 야간 평균): VOLTRA {dts_v:.3f}, 모사 {dts_m:.3f}, 오차 {dts_m - dts_v:+.4f}")

# ---- (2) TRISCO 정상상태 (실내 20 도 hc 1.415, 실외 -3.7 도 hc 16.03, hrb 5.1)
TI, TO, HI, HO, HRB = 20.0, -3.7, 1.415, 16.03, 5.1; DT = TI - TO
tb = pd.read_excel(R("정상상태_정답.xlsx")); tb.columns = ["case", "Phi", "Phi2", "P2D", "psi"]; t = tb[tb.case.str.startswith("Model A")].iloc[0]
bc2 = Boundary(np.array([TO]), np.array([TI]), np.array([HO]), np.array([HI]), dt=3600.0)
m.hrb = HRB
print(f"\n=== (2) TRISCO 'Model A(0.024)': 전체열류량 {t.Phi}, 전체열류량 2 {t.Phi2}, Φ2D {t.P2D}, Ψ {t.psi:.4f}")
R2 = 1 / (HI + 0.9 * HRB) + 1 / (HO + 0.9 * HRB) + 0.2 / 1.6 + 0.16 / 0.024 + 0.09 / 0.6 + 0.01 / 0.16
print(f"평면 벽체 해석해 Φ2D = {DT * 2.2 / R2:.4f} W (TRISCO 표 {t.P2D})")
for scheme in ("half", "voltra"):
    s = Solver(m, conv_lump=scheme); s.run(bc2, nsteps=1); F = s.zone_flows_into_out()
    z = sorted(k for k in F if k != 2)                               # 실내 영역 번호(1, 8)
    ins = [F[k][0][0] - F[k][1][0] for k in z]; out = F[2][1][0] - F[2][0][0]
    phi = 0.5 * sum(ins); psi = (2 * phi - t.P2D) / DT
    print(f"[{scheme:6s}] 방별 순유입(영역 {z}) {ins[0]:.4f}, {ins[1]:.4f} W, 실외 순유출 {out:.4f} W | 평균 Φ {phi:.4f} (TRISCO 평균 {0.5*(t.Phi+t.Phi2):.4f}, {100*(phi/(0.5*(t.Phi+t.Phi2))-1):+.2f} %)"
          f" | Ψ {psi:.4f} (TRISCO {t.psi:.4f}, 오차 {psi - t.psi:+.4f})")
