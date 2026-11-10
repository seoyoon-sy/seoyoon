"""VOLTRA 결과 엑셀과 hbcalc 결과를 야간(18~07시) 기준으로 비교. 새 코드(hbcalc)만 사용한다.

시트 이름에서 물성을 읽는다:  "결로 방지 단열재 X" -> XPS 열전도율,  "단열재 조정 Y" -> EPS 열전도율 (기본 0.032).
"""
import re
import numpy as np
import pandas as pd
from .boundary import Boundary
from .solver import Solver
from .presets import try3_model, try3_voltra_mesh_model

EXCEL_LABELS = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]    # 엑셀 열 이름(mm)
SIM_POS_VOLTRA_ORDER = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]   # VOLTRA onod 순서의 실제 위치(mm)


def parse_sheet(name):
    eps = xps = None
    m = re.search(r"결로 방지 단열재\s*([0-9.]+)", name)
    if m: xps = float(m.group(1))
    m = (re.search(r"단열재 (?:조정|조절)\s*[가-힣]*등급?\(?\s*([0-9.]+)", name)
         or re.search(r"단열재 (?:조정|조절)\s*[가-힣]*\s*([0-9.]+)", name))
    if m: eps = float(m.group(1))
    return eps, xps


def night_mask(n):
    m = np.zeros(n, bool)
    for d in range(30):
        m[18 + 24 * d: 18 + 24 * d + 14] = True
    return m


def load_cases(xlsx, skip=("Model A", "변동_시간대분포", "Sheet1", "면형 확인용 시트(분석 금지)")):
    xl = pd.ExcelFile(xlsx)
    return xl, [s for s in xl.sheet_names if s not in skip and s.strip() not in skip]


def compare_case(df, sim_surf, sim_back, labels, sim_pos, n=744):
    """df: 엑셀 시트. sim_surf/back: {위치mm: 시계열}. labels[i] 열을 sim_pos[i] 위치와 비교."""
    n = min(n, len(df)); m = night_mask(n)
    S = np.array([df[f"{p}mm 표면온도(℃)"].values[:n] for p in labels]).T
    W = np.array([df[f"{p}mm 석고보드 벽체 쪽 표면온도(℃)"].values[:n] for p in labels]).T
    M = np.array([sim_surf[p][:n] for p in sim_pos]).T
    Mw = np.array([sim_back[p][:n] for p in sim_pos]).T
    return (M - S)[m], (Mw - W)[m]


def run_case(name, bc, mesh="voltra", positions=None, base_lam=None, solver_kw=None, **geom):
    eps, xps = parse_sheet(name)
    lam = {"EPS": eps if eps is not None else 0.032}
    if xps is not None: lam["XPS"] = xps
    if base_lam: lam.update({k: v for k, v in base_lam.items() if k not in lam})
    mk = try3_voltra_mesh_model if mesh == "voltra" else try3_model
    if positions is not None:
        geom["offsets_mm"] = tuple(positions)
    model = mk(lam, **geom)
    s = Solver(model, **(solver_kw or {}))
    s.run(bc)
    return s, model
