"""N6 과보정 원인 분리용 정상상태 시험 .vtr 두 개(8.33).
N6b: N6 구성(블록 없음, 콘크리트 180, EPS 100, 석고 10, 내벽 180)에 N5의 띠 재료(열전도율 0.5, 밀도 1200, 비열 1000).
N5b: N5 구성(블록 없음, 콘크리트 220, EPS 160, 석고 12, 내벽 200)에 N6의 띠 재료(열전도율 0.04, 밀도 40, 비열 1450).
난방 20도 고정(Wall_TI.FTE), calc=3600(첫 스텝이 정상상태). 정상상태라 밀도와 비열은 결과에 영향이 없다.
같은 조건 hbcalc 첫 스텝 표면온도와 영역 열류를 expected_hbcalc_steady.csv 로 낸다(N5, N6 원본 포함).
사용: python examples/make_strip_split_cases.py [출력폴더]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.model import Material
from hbcalc.presets import try3_spec
from hbcalc.write_vtr import write_vtr
import test_walls_def as T

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_inputs", "strip_split")
os.makedirs(OUT, exist_ok=True)
TPL = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr")
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
STRIP = {"N5": (0.5, 1200, 1000), "N6": (0.04, 40, 1450)}

def make(key, strip_from):
    text, th, part, strip, ext, props = T.WALLS[key]
    props = dict(props); props["XPS"] = STRIP[strip_from]
    mats = {T.NAMES[k]: T.M(T.NAMES[k], *v) for k, v in props.items()}
    layers = [Layer(T.NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    spec = replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=strip, strip_extend_mm=ext, materials=mats)
    return build_tjunction(spec)

CASES = {"N6b_strip0.5": ("N6_noblock_insstrip", "N5", "N6 구성 + 띠 열전도율 0.5"), "N5b_strip0.04": ("N5_noblock", "N6", "N5 구성 + 띠 열전도율 0.04")}
ORIG = {"N5_noblock(원본)": ("N5_noblock", "N5"), "N6_noblock_insstrip(원본)": ("N6_noblock_insstrip", "N6")}
rows = []
def run(name, m, desc):
    for cl in ("half", "low"):
        s = Solver(m, conv_lump=cl); s.run(bc, nsteps=1); F = s.zone_flows_into_out()
        o = [z for z in F if s.m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in F if s.m.zones[z].kind == "indoor")
        r = {"케이스": name, "설명": desc, "격자": f"{m.shape[0]}x{m.shape[1]}", "대류배분": cl,
             "실외유출W": F[o][1][0] - F[o][0][0], "왼쪽실내유입W": F[ind[0]][0][0] - F[ind[0]][1][0], "오른쪽실내유입W": F[ind[1]][0][0] - F[ind[1]][1][0]}
        for p in POS: r[f"표면_{p}mm"] = s.series(f"surf_{p}")[0]
        rows.append(r)
for key, (wall, sf, desc) in CASES.items():
    m = make(wall, sf); write_vtr(m, TPL, os.path.join(OUT, f"{key}.vtr"), calc_s=3600, indoor_bc="heating"); run(key, m, desc); print(key, m.shape, flush=True)
for name, (wall, sf) in ORIG.items():
    run(name, make(wall, sf), "원본(VOLTRA 결과 있음)")
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "expected_hbcalc_steady.csv"), index=False, encoding="utf-8-sig")
print("완료:", OUT)
