"""변형 벽체 VOLTRA 결과(리포트 txt)와 hbcalc 비교. 사용: python examples/compare_variant_results.py [결과폴더]
.vtr 와 같은 조건: 템플릿 기본 물성(EPS 0.032, XPS 1.6), 실내온도 Wall_TI_2.FTE(비난방, 템플릿 기본)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.voltra_report import read_report

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_results")
L = lambda eps=180: [Layer("concrete", 200), Layer("EPS", eps), Layer("0.5B cement block", 90), Layer("gypsum", 10)]
CASES = {"A_EPS100": dict(layers=L(100)), "B_EPS250": dict(layers=L(250)),
         "C_partition250": dict(partition_mm=250), "D_strip20": dict(strip_mm=20, strip_extend_mm=120)}
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
N = 744
mk = night_mask(N)
bcs = {sh: Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI_2.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=sh) for sh in (0.0, -0.25)}
res = {}
for key, kw in CASES.items():
    _, a = read_report(os.path.join(DIR, f"{key}.txt"))
    V = a[:N, 7:]                                   # (744, 24): 노드 1..24 = 위치 12개 x (표면, 벽체쪽)
    Vs, Vb = V[:, 0::2], V[:, 1::2]
    model = build_tjunction(replace(try3_spec(snap_mm=1), **kw))
    s = Solver(model)
    for sh, bc in bcs.items():
        s.run(bc, nsteps=N)
        Ms = np.array([s.series(f"surf_{p}")[:N] for p in POS]).T
        Mb = np.array([s.series(f"back_{p}")[:N] for p in POS]).T
        res[(key, sh)] = dict(Vs=Vs, Vb=Vb, Ms=Ms, Mb=Mb, V=a[:N])
pickle_path = os.path.join(DIR, "_cmp.pkl")
import pickle; pickle.dump(res, open(pickle_path, "wb"))

pd.set_option("display.width", 220)
for sh in (0.0, -0.25):
    print(f"\n===== 일사 시각 보정 {sh} 시간 =====")
    rows = []
    for key in CASES:
        d = res[(key, sh)]
        e, ew = d["Ms"] - d["Vs"], d["Mb"] - d["Vb"]
        rows.append(dict(케이스=key, 표면RMSE=np.sqrt((e[mk] ** 2).mean()), 벽체쪽RMSE=np.sqrt((ew[mk] ** 2).mean()),
                         최대절대오차=np.abs(e[mk]).max(), 전체시간RMSE=np.sqrt((e ** 2).mean())))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("야간 평균 표면온도 편차(모사-VOLTRA) 위치별 ℃  위치(mm):", POS)
    for key in CASES:
        d = res[(key, sh)]; print(f"  {key:15s}", np.round((d['Ms'] - d['Vs'])[mk].mean(0), 3))
print("\n정상상태 t=0 표면온도 편차(모사-VOLTRA), 보정 무관:")
for key in CASES:
    d = res[(key, 0.0)]; print(f"  {key:15s}", np.round(d['Ms'][0] - d['Vs'][0], 3))
