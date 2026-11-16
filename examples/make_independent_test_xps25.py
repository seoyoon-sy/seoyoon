"""독립 테스트 T1 ~ T8 재생성: 원본(make_independent_test.py)과 난수와 모든 값이 같고 XPS 판 밀도 25, 비열 1300만 다르다.
원본은 열전도율 0.06 이상 판의 밀도를 600 ~ 2200으로 임의로 정해서 사용자의 TRY3 케이스(항상 25)와 달랐다(2026-11-16 정정).
출력: voltra_inputs/independent_test_xps25/. 사용: python examples/make_independent_test_xps25.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, build_tjunction
from hbcalc.model import Material
from hbcalc.presets import try3_spec
from hbcalc.write_vtr import write_vtr

SEED = 20261113
OUT = os.path.join(ROOT, "voltra_inputs", "independent_test_xps25"); os.makedirs(OUT, exist_ok=True)
TPL_J = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr")
NAMES = dict(concrete="concrete", EPS="EPS", block="0.5B cement block", gypsum="gypsum", XPS="XPS")
rng = np.random.default_rng(SEED)
r2 = lambda x: float(f"{x:.2g}")                                   # 유효숫자 2자리 (VOLTRA 입력이 읽기 쉽게)
logu = lambda a, b: r2(np.exp(rng.uniform(np.log(a), np.log(b))))
step = lambda a, b, s: int(rng.choice(np.arange(a, b + 1, s)))
XPS_BINS = [(1e-4, 5e-4), (5e-4, 3e-3), (3e-3, 1.2e-2), (1.2e-2, 3e-2), (3e-2, 8e-2), (8e-2, 3e-1), (3e-1, 1.0), (1.0, 3.0)]
order = rng.permutation(8)                                           # 블록 유무와 XPS 구간의 짝이 섞이도록
WALLS = []
for i in range(8):
    block = i % 2 == 0
    lo, hi = XPS_BINS[order[i]]; lx = logu(lo, hi)
    th = [step(150, 260, 10), step(80, 240, 10), step(90, 100, 10) if block else 0, step(10, 15, 1)]
    props = dict(concrete=(logu(1.0, 2.3), step(1600, 2400, 100), step(840, 1000, 20)),
                 EPS=(r2(rng.uniform(0.024, 0.045)), step(15, 30, 1), step(1300, 1500, 50)),
                 gypsum=(r2(rng.uniform(0.16, 0.25)), step(700, 900, 50), step(840, 1100, 20)))
    if block: props["block"] = (logu(0.35, 0.9), step(900, 1900, 100), step(840, 1050, 30))
    props["XPS"] = (lx, 40 if lx < 0.06 else step(600, 2200, 100), step(1300, 1450, 50) if lx < 0.06 else step(900, 1200, 100))
    props["XPS"] = (lx, 25, 1300)                         # 사용자 확인: 결로 방지 단열재 밀도 25, 비열 1300 (열전도율만 바꿈)
    WALLS.append((f"T{i + 1}", th, step(150, 250, 10), step(10, 15, 1), step(120, 200, 10), props, block))

rows = []
for name, th, part, strip, ext, props, block in WALLS:
    mats = {NAMES[k]: Material(NAMES[k], *v) for k, v in props.items()}
    layers = [Layer(NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    spec = replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=strip, strip_extend_mm=ext, materials=mats)
    m = build_tjunction(spec)
    write_vtr(m, TPL_J, os.path.join(OUT, f"{name}_heating.vtr"), indoor_bc="heating")
    if int(name[1:]) <= 4: write_vtr(m, TPL_J, os.path.join(OUT, f"{name}_nonheating.vtr"))
    r = dict(벽체=name, 블록=("있음" if block else "없음"), 격자=f"{m.shape[0]}x{m.shape[1]}", 콘크리트_mm=th[0], EPS_mm=th[1], 블록_mm=th[2], 석고_mm=th[3],
             내벽_mm=part, 판_두께_mm=strip, 판_돌출_mm=ext)
    for k, v in props.items(): r[f"{k}_열전도율"], r[f"{k}_밀도"], r[f"{k}_비열"] = v
    rows.append(r); print(name, r["블록"], m.shape, "XPS", props["XPS"][0], "EPS", th[1], props["EPS"][0], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "wall_definitions.csv"), index=False, encoding="utf-8-sig")
