"""시험 벽체 정의 (make_test_walls.py 와 compare_test_wall_results.py 가 함께 쓴다)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dataclasses import replace
from hbcalc import Layer, build_tjunction
from hbcalc.model import Material
from hbcalc.planar import planar_model, planar_cells
from hbcalc.presets import try3_spec

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
    # 시멘트 블록 없는 벽체(블록 두께 0): 콘크리트 - EPS - 석고. 띠는 EPS 시작면(콘크리트 안쪽 면)에서 시작한다.
    "N5_noblock": ("블록 없음: 콘크리트 220, EPS 160, 석고 12, 보통 열교", [220, 160, 0, 12], 200, 10, 170,
                   dict(concrete=(2.0, 2300, 880), EPS=(0.034, 20, 1450), gypsum=(0.21, 900, 1000), XPS=(0.5, 1200, 1000))),
    "N6_noblock_insstrip": ("블록 없음: 콘크리트 180, EPS 100, 석고 10, 띠가 단열재급(열전도율 0.04)", [180, 100, 0, 10], 180, 10, 170,
                        dict(concrete=(1.65, 2200, 880), EPS=(0.037, 18, 1400), gypsum=(0.25, 900, 1000), XPS=(0.04, 40, 1450))),
}
NAMES = dict(concrete="concrete", EPS="EPS", block="0.5B cement block", gypsum="gypsum", XPS="XPS")


def build(key):
    """반환: dict(mj=접합부 모델, mp=평면 모델(템플릿과 같은 칸 수), pl=평면 층 목록, mats=재료, cells=층별 칸 수, text=설명)."""
    text, th, part, strip, ext, props = WALLS[key]
    mats = {NAMES[k]: M(NAMES[k], *v) for k, v in props.items()}
    layers = [Layer(NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    spec = replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=strip, strip_extend_mm=ext, materials=mats)
    pl = [(NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    cells = planar_cells(pl, 20.0)
    return dict(mj=build_tjunction(spec), mp=planar_model(pl, {n: mats[n] for n, _ in pl}, cells=cells, width=2.2, nx=44),
                pl=pl, mats={n: mats[n] for n, _ in pl}, cells=cells, text=text)
