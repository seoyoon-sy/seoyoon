"""물성만 바꾼 시험 벽체 .vtr: 기하(격자, 두께, 블록, 출력 노드)는 사용자 템플릿 그대로이고 열전도율, 밀도, 비열 줄과
실내온도 경계(난방 20도 Wall_TI.FTE)만 바뀐다. 접합부 = TRY3_Plus_QTest2.vtr, 평면 = TRY3_Plus_2D.vtr.
사용: python examples/make_property_walls.py [출력폴더]"""
import os, sys, difflib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from hbcalc import Boundary, Solver
from hbcalc.model import Material
from hbcalc.planar import planar_model
from hbcalc.presets import try3_voltra_mesh_model
from hbcalc.validate import night_mask
from hbcalc.write_vtr import write_vtr, write_planar_vtr

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "voltra_inputs", "property_walls")
os.makedirs(OUT, exist_ok=True)
TPL_J = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr"); TPL_P = os.path.join(ROOT, "voltra_inputs", "TRY3_Plus_2D.vtr")
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
LAYERS = [("concrete", 200), ("EPS", 180), ("0.5B cement block", 90), ("gypsum", 10)]     # TRY3 두께 그대로
CELLS = [10, 9, 2, 1]                                                                    # 사용자 평면 템플릿의 층별 칸 수
# 이름: (설명, {재료: (열전도율, 밀도, 비열)}) -- 두께는 TRY3 그대로
SETS = {
    "P1_light": ("경량 재료", {"concrete": (1.0, 1600, 1000), "EPS": (0.040, 15, 1450), "0.5B cement block": (0.45, 900, 1050), "gypsum": (0.25, 800, 1000), "XPS": (0.3, 700, 1200)}),
    "P2_heavy": ("중량 재료", {"concrete": (2.3, 2400, 880), "EPS": (0.030, 30, 1500), "0.5B cement block": (0.9, 1900, 840), "gypsum": (0.21, 700, 1100), "XPS": (0.8, 1500, 900)}),
    "P3_mid": ("중간 재료", {"concrete": (1.4, 2000, 840), "EPS": (0.038, 20, 1300), "0.5B cement block": (0.35, 1100, 920), "gypsum": (0.18, 850, 950), "XPS": (1.0, 1800, 1000)}),
    "P4_weak": ("약한 열교(띠 열전도율 0.002)", {"concrete": (1.9, 2300, 850), "EPS": (0.035, 22, 1400), "0.5B cement block": (0.7, 1500, 900), "gypsum": (0.20, 750, 1050), "XPS": (0.002, 40, 1400)}),
}
_text = lambda p: (lambda b: b.decode("utf-8") if _ok(b) else b.decode("cp949"))(open(p, "rb").read())
def _ok(b):
    try: b.decode("utf-8"); return True
    except UnicodeDecodeError: return False

def diff_lines(a, b):
    d = [l for l in difflib.unified_diff(_text(a).split("\r\n"), _text(b).split("\r\n"), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return d

bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
mk = night_mask(len(bc.T_out)); rows, defs = [], []
for key, (text, props) in SETS.items():
    lam = {k: v[0] for k, v in props.items()}; rho = {k: v[1] for k, v in props.items()}; cp = {k: v[2] for k, v in props.items()}
    mj = try3_voltra_mesh_model().with_properties(lam=lam, rho=rho, cp=cp)
    fj = os.path.join(OUT, f"{key}_junction.vtr"); write_vtr(mj, TPL_J, fj, indoor_bc="heating")
    mats = {n: Material(n, *props[n]) for n, _ in LAYERS}
    fp = os.path.join(OUT, f"{key}_planar.vtr"); write_planar_vtr(LAYERS, mats, TPL_P, fp, cells=CELLS, indoor_bc="heating")
    dj, dp = diff_lines(TPL_J, fj), diff_lines(TPL_P, fp)
    kinds = lambda d: sorted({l[1:].split("=")[0].strip() for l in d})
    print(key, "접합부 바뀐 줄", len(dj), kinds(dj), "| 평면 바뀐 줄", len(dp), kinds(dp), flush=True)
    sj = Solver(mj); sj.run(bc)
    mp = planar_model(LAYERS, mats, cells=CELLS, width=2.2, nx=44); sp = Solver(mp); sp.run(bc)
    r = {"벽체": key, "설명": text, "평면_표면": sp.series("surf")[mk].mean(), "평면_벽체쪽": sp.series("back")[mk].mean()}
    for p in POS: r[f"표면_{p}mm"] = sj.series(f"surf_{p}")[mk].mean()
    for p in POS: r[f"벽체쪽_{p}mm"] = sj.series(f"back_{p}")[mk].mean()
    r["dTs(평면-0mm)"] = r["평면_표면"] - r["표면_0mm"]; rows.append(r)
    defs.append({"벽체": key, "설명": text, **{f"{m} (열전도율, 밀도, 비열)": "/".join(f"{x:g}" for x in v) for m, v in props.items()}})
pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "expected_hbcalc.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(defs).to_csv(os.path.join(OUT, "wall_definitions.csv"), index=False, encoding="utf-8-sig")
print("완료:", OUT)
