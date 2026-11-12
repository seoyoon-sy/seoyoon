"""VOLTRA .vtr 에 적힌 출력 노드(onod)와 같은 위치의 온도 예측 + 영역 열류. VOLTRA 리포트(Report Output)와 같은 시각(0 ~ calc 초, 1시간 간격)으로 CSV 를 만든다.
사용: python examples/predict_vtr_outputs.py 파일.vtr [출력csv] [실내온도파일]
가정: 경계조건 파일은 이 저장소와 같다(wall_TO.FTE, 실내 Wall_TI.FTE 20도 고정 등)."""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0] + "_prediction_hourly.csv"
TI = sys.argv[3] if len(sys.argv) > 3 else "Wall_TI.FTE"
raw = open(path, "rb").read()
try: text = raw.decode("utf-8")
except UnicodeDecodeError: text = raw.decode("cp949", "replace")
L = text.replace("\r", "").split("\n")
calc = int([l for l in L if l.startswith("calc=")][0].split("=")[1]); step = int([l for l in L if l.startswith("step=")][0].split("=")[1])
nz = len([l for l in L if l.startswith("zwid=")]); onod = [tuple(int(v) for v in l.split("=")[1].split()) for l in L if l.startswith("onod=")]
m = model_from_vtr(path)
n = calc // step + 1
print(f"{path}: 격자 {m.shape}, 출력 노드 {len(onod)}개, 시각 {n}개 (0 ~ {calc} s), z 칸 {nz}")
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", TI, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
s = Solver(m); s.run(bc, nsteps=n)
cols = {"time_s": np.arange(n) * step}
for no, ix, iy, iz in [(o[0], o[1], o[2], o[3]) for o in onod]:
    x, y = m.xe[ix] * 1000, m.ye[iy] * 1000
    note = "" if iz == nz // 2 else f"  (z 격자 번호 {iz}: 2D 모델은 중간면만 계산하므로 중간면 값)"
    cell = m.cid[min(ix, m.cid.shape[0] - 1), min(iy, m.cid.shape[1] - 1)]
    print(f"  출력 노드 {no + 1}: 격자선 (x {ix}, y {iy}) = ({x:.1f}, {y:.1f}) mm{note}")
    cols[f"onod{no + 1}_x{ix}_y{iy}"] = s.node_series(ix, iy)[:n]
F = s.zone_flows_into_out()
for z in sorted(F, key=lambda z: (m.zones[z].kind != "outdoor", z)):
    cols[f"colour{z}_into"] = F[z][0][:n]; cols[f"colour{z}_out"] = F[z][1][:n]
df = pd.DataFrame(cols); df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.4f")
print(df.iloc[[0, 1, 2]].round(3).to_string(index=False)); print("저장:", out)
mk = np.zeros(n, bool)
for d in range(30):
    mk[18 + 24 * d: 18 + 24 * d + 14] = True
mk[calc // 3600:] = False
print("야간 평균(18 ~ 07시, 완전한 밤만):", {k: round(float(v[mk].mean()), 3) for k, v in cols.items() if k.startswith("onod")})
