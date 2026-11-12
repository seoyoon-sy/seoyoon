"""G1 ~ G4 VOLTRA 정상상태를 목표로 hbcalc 설정(대류 배분, 형상계수 z 처리, 복사 배율)을 훑는다. 열류 오차(%)와 표면온도 편차를 함께 본다.
사용: python examples/diagnose_steady_G.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.voltra_report import read_report
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
KEYS = ["G1_fine_sym_strong", "G2_fine_sym_weak", "G3_default_sym_strong", "G4_default_sym_weak"]
TGT = {}
for k in KEYS:
    r = read_report(os.path.join(ROOT, "voltra_results", k + ".txt"))[1][0]
    TGT[k] = dict(out=r[2] - r[1], left=r[3] - r[4], right=r[5] - r[6], surf=r[7:][0::2])
MODELS = {}
def model(k):
    if k not in MODELS:
        from dataclasses import replace
        from hbcalc import build_tjunction
        from hbcalc.compat_trisco import read_trc
        from hbcalc.presets import try3_spec
        trc = {"G1": "T1_fine_sym_strong", "G2": "T2_fine_sym_weak", "G3": "T3_default_sym_strong", "G4": "T4_default_sym_weak"}[k[:2]]
        xw, yw = read_trc(os.path.join(ROOT, "trisco_cases", trc + ".trc"))[:2]
        spec = replace(try3_spec(), x_edges_mm=tuple(np.r_[0, np.cumsum(xw)]), y_edges_mm=tuple(np.r_[0, np.cumsum(yw)]))
        MODELS[k] = build_tjunction(spec).with_properties(lam={"XPS": 1.6 if "strong" in k else 0.0001})
    return MODELS[k]
def run(k, rad=1.0, **kw):
    s = Solver(model(k), **kw)
    if rad != 1.0: s.set_exchange_areas(s.rad_W * rad)
    s.run(bc, nsteps=1); F = s.zone_flows_into_out()
    o = [z for z in F if s.m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in F if s.m.zones[z].kind == "indoor")
    fl = (F[o][1][0] - F[o][0][0], F[ind[0]][0][0] - F[ind[0]][1][0], F[ind[1]][0][0] - F[ind[1]][1][0])
    M = np.array([s.series(f"surf_{p}")[0] for p in POS])
    return fl, M
if __name__ == "__main__":
    cfgs = [("half mid", dict(conv_lump="half")), ("half avg", dict(conv_lump="half", zmode="avg")), ("half inf", dict(conv_lump="half", zmode="inf")),
            ("low  mid", dict(conv_lump="low")), ("low  avg", dict(conv_lump="low", zmode="avg")),
            ("half mid rad.9", dict(conv_lump="half", rad=0.9)), ("half mid rad.85", dict(conv_lump="half", rad=0.85)), ("half mid rad.8", dict(conv_lump="half", rad=0.8)),
            ("half avg rad.95", dict(conv_lump="half", zmode="avg", rad=0.95))]
    print("각 칸: 실외 열류 오차 % / 표면 0~100 mm 평균 편차 / 표면 RMSE (모사 - VOLTRA)")
    print(f"{'설정':16s}", " | ".join(f"{k[:2]:>18s}" for k in KEYS))
    for lab, kw in cfgs:
        row = []
        for k in KEYS:
            fl, M = run(k, **dict(kw)); e = M - TGT[k]["surf"]
            row.append(f"{(fl[0]/TGT[k]['out']-1)*100:+.2f}%/{e[:8].mean():+.3f}/{np.sqrt((e**2).mean()):.3f}")
        print(f"{lab:16s}", " | ".join(f"{r:>18s}" for r in row))
