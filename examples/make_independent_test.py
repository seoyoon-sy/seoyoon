"""독립 테스트셋 .vtr 생성. 보정 규칙(conv_lump 'auto', strip_weight)을 고정한 뒤 한 번만 평가하기 위한 새 벽체 8개.
규칙을 맞추는 데는 쓰지 않는다(보정용은 voltra_inputs/calib_steady). 모사 예측값은 일부러 계산하지 않는다(규칙을 고정한 뒤 평가할 때 처음 계산).
- 난수 시드 고정(SEED). 결로 방지 단열재(XPS 판) 열전도율은 구간별로 나눠서 약한 열교 ~ 강한 열교를 모두 포함(약점인 0.0001 ~ 0.3 구간 비중 큼).
- 블록 있음 4개, 없음 4개. 두께, 열전도율, 밀도, 비열, 내벽 두께, 띠 두께와 돌출을 모두 무작위로 정함.
- 파일: voltra_inputs/independent_test/{이름}_heating.vtr (난방 20도 고정 Wall_TI.FTE, 744시간) 8개,
        {이름}_nonheating.vtr (Wall_TI_2.FTE) 4개(T1 ~ T4). 접합부, 자동 격자(snap 1 mm), 출력 노드 12위치(템플릿과 같음).
사용: python examples/make_independent_test.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, build_tjunction
from hbcalc.model import Material
from hbcalc.presets import try3_spec
from hbcalc.write_vtr import write_vtr

SEED = 20261113
OUT = os.path.join(ROOT, "voltra_inputs", "independent_test"); os.makedirs(OUT, exist_ok=True)
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
