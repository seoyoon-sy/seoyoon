"""G1 ~ G4 VOLTRA 정상상태(리포트 첫 행) 와 hbcalc(half, low) 비교: 열류(순, 들어옴, 나감), 좌우비, 표면온도.
사용: python examples/compare_steady_G_results.py [결과폴더]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc.voltra_report import read_report
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_results")
d = pd.read_csv(os.path.join(ROOT, "vtr_cases_steady", "expected_steady_hbcalc.csv"))
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
V = {}
for key in d.케이스.unique():
    r = read_report(os.path.join(DIR, key + ".txt"))[1][0]
    V[key] = dict(out=r[2] - r[1], left=r[3] - r[4], right=r[5] - r[6], left_in=r[3], left_out=r[4], surf=r[7:][0::2], back=r[7:][1::2])
np.set_printoptions(linewidth=200)
for key, v in V.items():
    print(f"\n== {key}: VOLTRA 순 열류 실외 {v['out']:.3f} 좌 {v['left']:.3f} 우 {v['right']:.3f} (좌우비 {v['left']/v['right']:.4f}), 왼쪽 실내 들어옴 {v['left_in']:.2f} 나감 {v['left_out']:.2f} W")
    for _, x in d[d.케이스 == key].iterrows():
        M = np.array([x[f"표면_{p}mm"] for p in POS]); e = M - v["surf"]
        print(f"  hbcalc {x.대류배분:4s} 열류 오차 실외 {(x.실외유출W/v['out']-1)*100:+.2f} % 좌 {(x.왼쪽실내유입W/v['left']-1)*100:+.2f} % 우 {(x.오른쪽실내유입W/v['right']-1)*100:+.2f} % 좌우비 {x.왼쪽실내유입W/x.오른쪽실내유입W:.4f} | 표면 RMSE {np.sqrt((e**2).mean()):.3f} 0~100mm 평균 {e[:8].mean():+.3f}")
        print("     편차:", np.round(e, 3))
    print("  VOLTRA 표면:", np.round(v["surf"], 2))
print("\n격자 효과 (굵은 격자 - 세밀 격자), 표면온도 12위치")
for a, b, lab in (("G3_default_sym_strong", "G1_fine_sym_strong", "강함"), ("G4_default_sym_weak", "G2_fine_sym_weak", "약함")):
    dv = V[a]["surf"] - V[b]["surf"]
    print(f"  {lab} VOLTRA:", np.round(dv, 3), f"| 열류 실외 {V[a]['out']-V[b]['out']:+.3f} W")
    for cl in ("half", "low"):
        ma = d[(d.케이스 == a) & (d.대류배분 == cl)].iloc[0]; mb = d[(d.케이스 == b) & (d.대류배분 == cl)].iloc[0]
        dm = np.array([ma[f"표면_{p}mm"] - mb[f"표면_{p}mm"] for p in POS])
        print(f"  {lab} hbcalc {cl:4s}:", np.round(dm, 3), f"| 열류 실외 {ma.실외유출W-mb.실외유출W:+.3f} W")
