"""시험 벽체 .vtr 생성: 열전도율, 밀도, 비열(과 두께)을 모두 바꾼 새 벽체마다
 (1) 2D 접합부 .vtr (VOLTRA 2D, TRY3_Plus_QTest2.vtr 형식) 와 (2) 평면 벽체 .vtr (비열교부 기준, TRY3_Plus_2D.vtr 형식) 를 만든다.
같은 조건(난방 20도 고정 Wall_TI.FTE)의 hbcalc 야간 평균 표면온도를 expected_hbcalc.csv 로 함께 낸다.
사용: python examples/make_test_walls.py [출력폴더]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.model import Material
from hbcalc.planar import planar_model, planar_cells
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.write_vtr import write_vtr, write_planar_vtr

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_inputs", "test_walls")
os.makedirs(OUT, exist_ok=True)
TPL_J = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr")
TPL_P = os.path.join(ROOT, "voltra_inputs", "TRY3_Plus_2D.vtr")
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
M = lambda nm, lam, rho, cp: Material(nm, lam, rho, cp)          # 방사율 0.9, 일사반사율 0.1 은 템플릿 값 그대로
# 이름: (설명, 층 두께 mm [콘크리트, EPS, 블록, 석고], 내벽 mm, 띠 mm, 띠 돌출 mm, 물성 {재료: (열전도율, 밀도, 비열)})
WALLS = {
    "N1_light": ("경량 구조: 얇은 콘크리트, EPS 120, 중간 열교", [150, 120, 90, 12], 150, 10, 170,
                 dict(concrete=(1.0, 1600, 1000), EPS=(0.040, 15, 1450), block=(0.45, 900, 1050), gypsum=(0.25, 800, 1000), XPS=(0.3, 700, 1200))),
    "N2_heavy": ("중량 구조: 두꺼운 콘크리트, EPS 220, 강한 열교", [250, 220, 100, 15], 250, 15, 170,
                 dict(concrete=(2.3, 2400, 880), EPS=(0.030, 30, 1500), block=(0.9, 1900, 840), gypsum=(0.21, 700, 1100), XPS=(0.8, 1500, 900))),
    "N3_thin": ("얇은 단열: EPS 80, 강한 열교", [180, 80, 90, 10], 180, 10, 170,
                dict(concrete=(1.4, 2000, 840), EPS=(0.038, 20, 1300), block=(0.35, 1100, 920), gypsum=(0.18, 850, 950), XPS=(1.0, 1800, 1000))),
    "N4_weak": ("약한 열교: 띠 열전도율 0.002", [200, 160, 90, 10], 200, 10, 170,
                dict(concrete=(1.9, 2300, 850), EPS=(0.035, 22, 1400), block=(0.7, 1500, 900), gypsum=(0.20, 750, 1050), XPS=(0.002, 40, 1400))),
}
NAMES = dict(concrete="concrete", EPS="EPS", block="0.5B cement block", gypsum="gypsum", XPS="XPS")
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
mk = night_mask(len(bc.T_out))
rows, desc = [], []
for key, (text, th, part, strip, ext, props) in WALLS.items():
    mats = {NAMES[k]: M(NAMES[k], *v) for k, v in props.items()}
    layers = [Layer("concrete", th[0]), Layer("EPS", th[1]), Layer("0.5B cement block", th[2]), Layer("gypsum", th[3])]
    spec = replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=strip, strip_extend_mm=ext, materials=mats)
    mj = build_tjunction(spec)
    pj = write_vtr(mj, TPL_J, os.path.join(OUT, f"{key}_junction.vtr"), indoor_bc="heating")
    pl = [(NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th)]
    pp, cells = write_planar_vtr(pl, {n: mats[n] for n, _ in pl}, TPL_P, os.path.join(OUT, f"{key}_planar.vtr"), indoor_bc="heating")
    sj = Solver(mj); sj.run(bc)
    mp = planar_model(pl, {n: mats[n] for n, _ in pl}, cells=cells, width=2.2, nx=44)
    sp = Solver(mp); sp.run(bc)
    r = {"벽체": key, "설명": text, "접합부 격자": f"{mj.shape[0]}x{mj.shape[1]}", "평면 칸 수(층별)": "-".join(map(str, cells))}
    r["평면_표면"] = sp.series("surf")[mk].mean(); r["평면_벽체쪽"] = sp.series("back")[mk].mean()
    for p in POS: r[f"표면_{p}mm"] = sj.series(f"surf_{p}")[mk].mean()
    for p in POS: r[f"벽체쪽_{p}mm"] = sj.series(f"back_{p}")[mk].mean()
    r["dTs(평면-0mm)"] = r["평면_표면"] - r["표면_0mm"]
    rows.append(r); print(key, mj.shape, cells, round(r["dTs(평면-0mm)"], 3), flush=True)
pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "expected_hbcalc.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([{"벽체": k, "설명": v[0], "층 두께 mm (콘크리트, EPS, 블록, 석고)": "/".join(map(str, v[1])), "내벽 mm": v[2], "띠 mm": v[3], "띠 돌출 mm": v[4],
               **{f"{NAMES[m]} (열전도율, 밀도, 비열)": "/".join(f"{x:g}" for x in vals) for m, vals in v[5].items()}} for k, v in WALLS.items()]
             ).to_csv(os.path.join(OUT, "wall_definitions.csv"), index=False, encoding="utf-8-sig")
print("완료:", OUT)
