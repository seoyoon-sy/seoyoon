"""다음 VOLTRA 시험 파일 (CLAUDE.md 8.36).
A) voltra_inputs/next_steady/: 정상상태(calc 3600, 난방 20도 고정) 3개. XPS 판 열전도율과 블록 유무가 대류 배분 보정에 미치는 영향 분리.
   N6c_block_strip0.04: N6 구성에 시멘트 블록 90 mm를 추가(판 열전도율 0.04), T3nb_strip0.04: TRY3 구성에서 블록을 뺌(판 0.04),
   N5w_strip0.005: N5 구성에 판 열전도율 0.005.
B) voltra_inputs/test_walls_nonheating/: N1 ~ N6 비난방(실내온도 변동 Wall_TI_2.FTE) 744시간 접합부와 평면 .vtr 12개. 같은 조건 hbcalc 야간 평균 표면온도 expected_hbcalc.csv.
사용: python examples/make_next_voltra_tests.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec, TRY3_MATERIALS
from hbcalc.validate import night_mask
from hbcalc.write_vtr import write_vtr, write_planar_vtr
import test_walls_def as T

TPL_J = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr"); TPL_P = os.path.join(ROOT, "voltra_inputs", "TRY3_Plus_2D.vtr")
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
OA = os.path.join(ROOT, "voltra_inputs", "next_steady"); OB = os.path.join(ROOT, "voltra_inputs", "test_walls_nonheating")
os.makedirs(OA, exist_ok=True); os.makedirs(OB, exist_ok=True)
bcH = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
bcN = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI_2.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)

def wall_model(key, layers_mm=None, strip=None, add_block=None):
    text, th, part, st, ext, props = T.WALLS[key]
    props = dict(props)
    if strip is not None: props["XPS"] = strip
    mats = {T.NAMES[k]: T.M(T.NAMES[k], *v) for k, v in props.items()}
    th = list(th)
    if add_block:
        th[2] = add_block; mats["0.5B cement block"] = TRY3_MATERIALS["0.5B cement block"]
    layers = [Layer(T.NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    return build_tjunction(replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=st, strip_extend_mm=ext, materials=mats))

rows = []
def steady(name, m, desc):
    write_vtr(m, TPL_J, os.path.join(OA, name + ".vtr"), calc_s=3600, indoor_bc="heating")
    for cl in ("half", "low", "auto"):
        s = Solver(m, conv_lump=cl); s.run(bcH, nsteps=1); F = s.zone_flows_into_out()
        o = [z for z in F if s.m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in F if s.m.zones[z].kind == "indoor")
        r = {"케이스": name, "설명": desc, "격자": f"{m.shape[0]}x{m.shape[1]}", "대류배분": cl, "실외유출W": F[o][1][0] - F[o][0][0], "왼쪽실내유입W": F[ind[0]][0][0] - F[ind[0]][1][0], "오른쪽실내유입W": F[ind[1]][0][0] - F[ind[1]][1][0]}
        for p in POS: r[f"표면_{p}mm"] = s.series(f"surf_{p}")[0]
        rows.append(r)
    print(name, m.shape, flush=True)

steady("N6c_block_strip0.04", wall_model("N6_noblock_insstrip", add_block=90), "N6 구성 + 시멘트 블록 90 mm, 판 열전도율 0.04")
t3 = build_tjunction(try3_spec(snap_mm=1, layers=[Layer("concrete", 200), Layer("EPS", 180), Layer("gypsum", 10)])).with_properties(lam={"XPS": 0.04})
steady("T3nb_strip0.04", t3, "TRY3 구성에서 블록을 뺌(콘크리트 200, EPS 180, 석고 10), 판 열전도율 0.04")
steady("N5w_strip0.005", wall_model("N5_noblock", strip=(0.005, 40, 1400)), "N5 구성, 판 열전도율 0.005")
pd.DataFrame(rows).round(4).to_csv(os.path.join(OA, "expected_hbcalc_steady.csv"), index=False, encoding="utf-8-sig")

mk = night_mask(len(bcN.T_out)); rows = []
for key in T.WALLS:
    b = T.build(key)
    write_vtr(b["mj"], TPL_J, os.path.join(OB, f"{key}_junction.vtr"), indoor_bc="nonheating")
    write_planar_vtr(b["pl"], b["mats"], TPL_P, os.path.join(OB, f"{key}_planar.vtr"), indoor_bc="nonheating")
    sj = Solver(b["mj"]); sj.run(bcN); sp = Solver(b["mp"]); sp.run(bcN)
    r = {"벽체": key, "평면_표면": sp.series("surf")[mk].mean(), "평면_벽체쪽": sp.series("back")[mk].mean()}
    for p in POS: r[f"표면_{p}mm"] = sj.series(f"surf_{p}")[mk].mean()
    r["dTs(평면-0mm)"] = r["평면_표면"] - r["표면_0mm"]; rows.append(r); print(key, flush=True)
pd.DataFrame(rows).round(3).to_csv(os.path.join(OB, "expected_hbcalc.csv"), index=False, encoding="utf-8-sig")
print("완료")
