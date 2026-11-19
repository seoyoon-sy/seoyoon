"""Model A (TRY3_INS_160_ver2.vtr) 모사. 구성: 콘크리트 200, EPS 160 mm(열전도율 0.024), 시멘트 블록 90, 석고 10 mm, 내벽 콘크리트 200 mm(XPS 띠 없음).
접합부는 .vtr를 그대로 읽고(손격자 55 x 60), 비열교부는 같은 층 구성의 평면 벽체 모델이다.
get(shift): 난방(Wall_TI.FTE) 744시간 -> dict(M_surf (12, N) 엑셀 열 순서 0, 10, ..., 500 mm, M_planar (N,), T_out, T_in)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.planar import planar_model
from hbcalc.presets import TRY3_MATERIALS
VTR = os.path.join(ROOT, "voltra_inputs", "TRY3_INS_160_ver2.vtr"); N = 744
LAYERS = [("concrete", 200), ("EPS", 160), ("0.5B cement block", 90), ("gypsum", 10)]

def _bc(shift, ti="Wall_TI.FTE"):
    return Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=shift)

def get(shift=0.0):
    bc = _bc(shift)
    m = model_from_vtr(VTR); s = Solver(m); s.run(bc, nsteps=N)
    M = np.array([s.series(f"onod{2 * k}")[:N] for k in range(12)])
    mats = dict(TRY3_MATERIALS); mats["EPS"] = replace(mats["EPS"], lam=0.024)
    sp = Solver(planar_model(LAYERS, mats, hrb=5.25)); sp.run(bc, nsteps=N)
    return dict(M_surf=M, M_planar=sp.series("surf")[:N], T_out=bc.T_out[:N], T_in=bc.T_in[:N])
